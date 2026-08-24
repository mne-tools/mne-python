"""W5: HDF5 chunk-cache tuning + zarr v3 local-read verdict (interleaved).

Measures 300 random 2 s windows (seed 99) per pass; arms alternate within one
process for `--rounds` rounds (order re-shuffled per round, seeded) so machine
drift hits every arm equally. Persistent handles (training-loop pattern).

Usage: python bench_cache.py [--rounds 7] [--n-win 300] [--seed 99]
Writes cache_results_<stamp>.json next to this file.
"""

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data_big"
H5 = DATA / "bench_big.h5"
ZARR_WIN = DATA / "bench_big_win.zarr"
ZARR_T10S = DATA / "bench_big_t10s.zarr"

MB = 1024 * 1024
DEF_NSLOTS = 521          # HDF5 default hash slots
DEF_NBYTES = MB           # HDF5 default raw-chunk cache


def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def make_prime(n):
    n = max(3, int(n) | 1)
    while not is_prime(n):
        n += 2
    return n


def build_arms(meta, starts):
    win = int(2 * meta["sfreq"])
    nt = meta["n_times"]
    cols, ks = set(), set()
    for s in starts:
        cols.update({s // 512, (s + win - 1) // 512})
        ks.update({s // 5120, (s + win - 1) // 5120})
    fit_win_bytes = len(cols) * 128 * 512 * 4            # exact bytes to hold all touched win chunks
    fit_t10_bytes = len(ks) * 128 * 5120 * 4             # ... t10 chunks (all channels touched per window)

    arms = []
    # --- h5py t10s dataset: cache-size arms ---------------------------------
    t = "t10s"
    arms.append(dict(name=f"h5_{t}_default", ds=t,
                     kw=dict(rdcc_nbytes=DEF_NBYTES, rdcc_nslots=DEF_NSLOTS)))
    arms.append(dict(name=f"h5_{t}_fit", ds=t,
                     kw=dict(rdcc_nbytes=fit_t10_bytes, rdcc_nslots=make_prime(2 * len(ks) * 128))))
    arms.append(dict(name=f"h5_{t}_big512", ds=t,
                     kw=dict(rdcc_nbytes=512 * MB, rdcc_nslots=make_prime(65537))))
    arms.append(dict(name=f"h5_{t}_ns10103", ds=t,
                     kw=dict(rdcc_nbytes=DEF_NBYTES, rdcc_nslots=10103)))
    arms.append(dict(name=f"h5_{t}_ns65537", ds=t,
                     kw=dict(rdcc_nbytes=DEF_NBYTES, rdcc_nslots=65537)))
    # --- h5py win dataset: control + same treatment -------------------------
    arms.append(dict(name="h5_win_default_CTRL", ds="win",
                     kw=dict(rdcc_nbytes=DEF_NBYTES, rdcc_nslots=DEF_NSLOTS)))
    arms.append(dict(name="h5_win_fit", ds="win",
                     kw=dict(rdcc_nbytes=fit_win_bytes, rdcc_nslots=make_prime(2 * len(cols)))))
    arms.append(dict(name="h5_win_big512", ds="win",
                     kw=dict(rdcc_nbytes=512 * MB, rdcc_nslots=make_prime(65537))))
    arms.append(dict(name="h5_win_ns10103", ds="win",
                     kw=dict(rdcc_nbytes=DEF_NBYTES, rdcc_nslots=10103)))
    arms.append(dict(name="h5_win_ns65537", ds="win",
                     kw=dict(rdcc_nbytes=DEF_NBYTES, rdcc_nslots=65537)))
    # --- file driver variants (once each, interleaved like the rest) --------
    arms.append(dict(name="h5_t10s_sec2", ds="t10s",
                     kw=dict(rdcc_nbytes=DEF_NBYTES, rdcc_nslots=DEF_NSLOTS),
                     note="explicit sec2 (default) driver"))
    arms.append(dict(name="h5_t10s_core", ds="t10s", kw=dict(),
                     open_extra=dict(driver="core", backing_store=False),
                     note="core driver: whole file read into RAM at open"))
    # --- zarr ----------------------------------------------------------------
    arms.append(dict(name="zarr_win_default", zarr=str(ZARR_WIN), zcfg={}))
    arms.append(dict(name="zarr_win_c1", zarr=str(ZARR_WIN),
                     zcfg={"async.concurrency": 1}))
    arms.append(dict(name="zarr_win_c32", zarr=str(ZARR_WIN),
                     zcfg={"async.concurrency": 32}))
    arms.append(dict(name="zarr_t10s_default", zarr=str(ZARR_T10S), zcfg={}))
    return arms, dict(fit_win_bytes=fit_win_bytes, fit_t10s_bytes=fit_t10_bytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--n-win", type=int, default=300)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    import h5py

    meta = json.loads((DATA / "meta.json").read_text())
    sfreq, nt = meta["sfreq"], meta["n_times"]
    width = int(2 * sfreq)                       # 1024 samples
    starts = np.random.default_rng(args.seed).integers(
        0, nt - width - 1, size=args.n_win).astype(int)

    arms, fits = build_arms(meta, starts)
    print(f"{len(arms)} arms x {args.rounds} interleaved rounds x "
          f"{args.n_win} windows | fit(win)={fits['fit_win_bytes']/MB:.1f} MiB "
          f"fit(t10s)={fits['fit_t10s_bytes']/MB:.1f} MiB")

    # ---- open persistent handles -------------------------------------------
    import zarr

    handles = {}
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for arm in arms:
        if "zarr" not in arm:
            extra = arm.get("open_extra", {})
            f = h5py.File(H5, "r", **arm["kw"], **extra)
            handles[arm["name"]] = f[arm["ds"]]
        else:
            arr = zarr.open_array(store=arm["zarr"], mode="r")
            handles[arm["name"]] = arr
    rss_open = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_unit = 1024 * 1024 if sys.platform == "darwin" else 1024  # macOS: bytes; Linux: KiB
    print(f"peak RSS after all opens: {rss_open/rss_unit:.0f} MiB "
          f"(delta {((rss_open-rss0)/rss_unit):.0f} MiB incl. core-driver RAM copy)")

    def pass_h5(ds):
        acc = 0.0
        for s in starts:
            acc += float(ds[:, s:s + width].astype(np.float64).sum())
        return acc

    def pass_zarr(arr, cfg):
        acc = 0.0
        cm = zarr.config.set(cfg) if cfg else None
        if cm is not None:
            cm.__enter__()
        try:
            for s in starts:
                acc += float(arr[:, s:s + width].astype(np.float64).sum())
        finally:
            if cm is not None:
                cm.__exit__(None, None, None)
        return acc

    def run_once(arm):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter_ns()
            if "zarr" in arm:
                chk = pass_zarr(handles[arm["name"]], arm["zcfg"])
            else:
                chk = pass_h5(handles[arm["name"]])
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
        finally:
            gc.enable()
        return dt_ms, chk

    # ---- warmup (page cache + arm-local caches) ----------------------------
    order_rng = np.random.default_rng(12345)
    print("warmup...", flush=True)
    ref_chk = None
    for arm in arms:
        ms, chk = run_once(arm)
        if ref_chk is None:
            ref_chk = chk
        assert abs(chk - ref_chk) < 1e-6 * abs(ref_chk), f"checksum mismatch {arm['name']}"

    # ---- interleaved rounds -------------------------------------------------
    times = {a["name"]: [] for a in arms}
    wall = []
    for r in range(args.rounds):
        order = list(arms)
        order_rng.shuffle(order)
        t_r0 = time.perf_counter()
        for arm in order:
            ms, _ = run_once(arm)
            times[arm["name"]].append(ms)
        wall.append(time.perf_counter() - t_r0)
        done = ", ".join(f"{a['name'].split('_', 1)[0]}:{times[a['name']][-1]:.0f}ms"
                         for a in arms[:3])
        print(f"round {r+1}/{args.rounds} done ({wall[-1]:.1f}s) {done}", flush=True)

    # ---- summarize ----------------------------------------------------------
    res = dict(stamp=time.strftime("%Y%m%d-%H%M%S"), machine="Apple M4 Max, 36 GB",
               rounds=args.rounds, n_win=args.n_win, seed=args.seed,
               peak_rss_mib=rss_open / rss_unit, fits=fits, arms={}, control_rounds=None)
    print(f"\n{'arm':<24}{'us/win median':>14}{'IQR':>18}{'min..max':>18}")
    ctrl_med = None
    for a in arms:
        v = np.array(times[a["name"]]) * 1000.0 / args.n_win   # us/win
        med = float(np.median(v))
        q1, q3 = np.percentile(v, [25, 75])
        res["arms"][a["name"]] = dict(us_per_win=[float(x) for x in v],
                                      median=med, iqr=float(q3 - q1),
                                      min=float(v.min()), max=float(v.max()),
                                      kind="zarr" if "zarr" in a else "h5",
                                      note=a.get("note", ""))
        flag = ""
        if "CTRL" in a["name"]:
            ctrl_med = med
        print(f"{a['name']:<24}{med:>11.1f}  [{np.percentile(v,25):>7.1f},{q3:>7.1f}]"
              f"  {v.min():>7.1f}..{v.max():<7.1f}{flag}")

    # drift check: round-by-round of control arm
    cv = np.array(times["h5_win_default_CTRL"]) * 1000.0 / args.n_win
    res["control_rounds"] = [float(x) for x in cv]
    print(f"\nCONTROL h5_win per-round µs/win: {[round(x,1) for x in cv]}")
    print(f"CONTROL spread max/min = {cv.max()/cv.min():.3f}")
    out = HERE / f"cache_results_{res['stamp']}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
