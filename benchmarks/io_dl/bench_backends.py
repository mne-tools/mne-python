"""Compare storage backends / provider libraries on identical signal data.

Works on any preset dir (data/ or data_big/). Sections:

FULL   : end-to-end load of the entire recording through each backend
WINDOWS: 300 shuffled windows (win = 2 s) -- the DL training pattern

Backends:
  mne_<fmt>       : MNE readers (preload True/False)
  edfio           : edfio.read_edf eager + .data; lazy variant for windows
  h5_<layout>     : h5py datasets (t10s / win / full chunk layouts)
  zarr_<layout>   : zarr arrays (t10s / win)
  npz             : numpy .npz (whole-array)
  floor_memmap    : raw float32 .bin via np.memmap

Usage:
    python benchmarks/io_dl/bench_backends.py [--dir data] [--repeats 3]
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

TREE_ROOT = HERE.resolve().parents[1].parent
import sys  # noqa: E402

sys.path.insert(0, str(TREE_ROOT))

import mne  # noqa: E402

mne.set_log_level("ERROR")


def timed(fn, repeats=3, warmup=1):
    for _ in range(warmup):
        fn()
    out = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            fn()
            out.append((time.perf_counter_ns() - t0) / 1e6)
    finally:
        gc.enable()
    return float(np.median(out))


def rand_starts(rng, n_times, win, n):
    return rng.integers(0, n_times - win - 1, size=n).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data_big")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-win", type=int, default=300)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    d = HERE / args.dir
    meta = json.loads((d / "meta.json").read_text())
    n_ch, sfreq, n_times = meta["n_ch"], meta["sfreq"], meta["n_times"]
    base = d.name.replace("data", "bench")
    win = int(2 * sfreq)
    rng = np.random.default_rng(args.seed)
    starts = rand_starts(rng, n_times, win, args.n_win)

    results = {}
    print(f"\n=== {d.name}: {n_ch} ch, {sfreq:.0f} Hz, {meta['dur']:.0f} s | "
          f"win={win} ({win/sfreq:.1f} s) x {args.n_win} ===")

    # ---------------- FULL loads -------------------------------------------
    print(f"\n{'backend':<24}{'full load':>12} {'MB/s':>8} | "
          f"{'300 windows':>12} {'us/win':>8}")
    rows = []

    def record(name, full_fn=None, win_fn=None, full_bytes=None):
        f_ms = timed(full_fn, args.repeats) if full_fn else None
        w_ms = timed(win_fn, max(args.repeats, 3)) if win_fn else None
        mbps = full_bytes / 1e6 / (f_ms / 1000) if (full_bytes and f_ms) else None
        usw = w_ms * 1000 / args.n_win if w_ms else None
        results[name] = dict(
            full_ms=f_ms, win_ms=w_ms, mbps=mbps, us_per_win=usw,
        )
        print(f"{name:<24}"
              f"{f_ms if f_ms else float('nan'):>9.1f}ms "
              f"{mbps if mbps else float('nan'):>7.0f} | "
              f"{w_ms if w_ms else float('nan'):>11.1f}ms "
              f"{usw if usw else float('nan'):>7.1f}")

    # floors -----------------------------------------------------------------
    binp = d / f"{base}.bin"
    shape = (n_ch, n_times)
    nbytes_f32 = n_ch * n_times * 4

    def mm_full():
        a = np.memmap(binp, "<f4", "r", shape=shape)
        s = float(a.astype(np.float64).sum())
        del a
        return s

    def mm_wins():
        mm = np.memmap(binp, "<f4", "r", shape=shape)
        acc = 0.0
        for s in starts:
            acc += float(mm[:, s : s + win].astype(np.float64).sum())
        del mm
        return acc

    record("floor_memmap", mm_full, mm_wins, nbytes_f32)

    # MNE ------------------------------------------------------------------
    from mne.io import read_raw_bdf, read_raw_brainvision, read_raw_edf, read_raw_fif

    mne_readers = {
        "mne_edf": (lambda pl: read_raw_edf(d / f"{base}.edf", preload=pl),
                    None),
        "mne_bdf": (lambda pl: read_raw_bdf(d / f"{base}.bdf", preload=pl),
                    None),
        "mne_bv": (lambda pl: read_raw_brainvision(d / f"{base}.vhdr", preload=pl),
                   nbytes_f32),
        "mne_fif": (lambda pl: read_raw_fif(d / f"{base}_raw.fif", preload=pl),
                    nbytes_f32),
    }
    for name, (reader, fb) in mne_readers.items():
        path_exists = True
        try:
            reader(False)
        except Exception:
            path_exists = False
        if not path_exists:
            continue

        def full(pl=True, reader=reader):
            r = reader(pl)
            s = float(r.get_data().sum())
            del r
            return s

        def wins(reader=reader, preloaded=False):
            r = reader(preloaded)
            acc = 0.0
            for s in starts:
                acc += float(r.get_data(start=s, stop=s + win).sum())
            del r
            return acc

        record(name, full, wins, fb or nbytes_f32)

        if name in ("mne_edf", "mne_fif"):
            w_ms = timed(lambda: wins(preloaded=True), max(args.repeats, 3))
            usw = w_ms * 1000 / args.n_win
            results[name + "_preloaded"] = dict(full_ms=None, win_ms=w_ms,
                                                mbps=None, us_per_win=usw)
            print(f"{name + '_preloaded':<24}"
                  f"{'--':>12} {'--':>8} | {w_ms:>11.1f}ms {usw:>7.1f}")

    # EEGLAB (.set/.fdt) ------------------------------------------------------
    set_path = d / f"{base}.set"
    if set_path.exists():
        from mne.io import read_raw_eeglab

        def set_full():
            r = read_raw_eeglab(set_path, preload=True)
            s = float(r.get_data().sum())
            del r
            return s

        def set_wins():
            r = read_raw_eeglab(set_path, preload=False)
            acc = 0.0
            for s in starts:
                acc += float(r.get_data(start=int(s), stop=int(s) + win).sum())
            del r
            return acc

        record("mne_set", set_full, set_wins, nbytes_f32)

    # NWB (time-major HDF5; window = contiguous row block) --------------------
    nwb_path = d / f"{base}.nwb"
    if nwb_path.exists():

        def nwb_full():
            from pynwb import NWBHDF5IO

            with NWBHDF5IO(nwb_path, "r") as io:
                a = np.asarray(io.read().acquisition["ElectricalSeries"].data[:])
                s = float(a.astype(np.float64).sum())
            return s

        def nwb_wins():
            from pynwb import NWBHDF5IO

            io = NWBHDF5IO(nwb_path, "r")
            arr = io.read().acquisition["ElectricalSeries"].data
            acc = 0.0
            for s in starts:
                w = np.asarray(arr[s : s + win]).T.astype(np.float64)
                acc += float(w.sum())
            io.close()
            return acc

        record("nwb", nwb_full, nwb_wins, nbytes_f32)

    # edfio ------------------------------------------------------------------
    try:
        from edfio import read_bdf, read_edf

        def edfio_full_eager():
            e = read_edf(d / f"{base}.edf", lazy_load_data=False)
            s = sum(float(sig.data.sum()) for sig in e.signals)
            return s

        def edfio_lazy_wins():
            from edfio.edf_signal import _calculate_gain_and_offset

            e = read_edf(d / f"{base}.edf", lazy_load_data=True)
            acc = 0.0
            for s in starts:
                t0s, t1s = s / sfreq, (s + win) / sfreq
                tot = 0.0
                for sg in e.signals:
                    dg = sg.get_digital_slice(t0s, t1s)
                    gain, offset = _calculate_gain_and_offset(
                        sg.digital_min, sg.digital_max,
                        sg.physical_min, sg.physical_max,
                    )
                    tot += float(((dg + offset) * gain).sum())
                acc += tot
            return acc

        record("edfio_edf_eager", edfio_full_eager, None,
               meta["sizes"].get("edf", 0))
        record("edfio_edf_lazy_wins", None, edfio_lazy_wins, None)
    except Exception as exc:  # noqa: BLE001
        print(f"edfio section skipped: {exc!r}")

    # npz ----------------------------------------------------------------------
    npzp = d / f"{base}.npz"
    if npzp.exists():

        def npz_full():
            a = np.load(npzp)["data"]
            s = float(a.astype(np.float64).sum())
            del a
            return s

        record("npz", npz_full, None, nbytes_f32)

    # hdf5 layouts --------------------------------------------------------------
    h5p = d / f"{base}.h5"
    if h5p.exists():
        import h5py

        with h5py.File(h5p, "r") as f:
            layouts = list(f.keys())
        for lay in layouts:

            def h5_full(lay=lay):
                with h5py.File(h5p, "r") as f:
                    a = f[lay][:]
                    s = float(a.astype(np.float64).sum())
                return s

            def h5_wins(lay=lay):
                f = h5py.File(h5p, "r")[lay]
                acc = 0.0
                for s in starts:
                    acc += float(f[:, s : s + win].astype(np.float64).sum())
                f.file.close()
                return acc

            record(f"h5_{lay}", h5_full, h5_wins, nbytes_f32)

    # zarr layouts ----------------------------------------------------------------
    for lay in ("t10s", "win"):
        zp = d / f"{base}_{lay}.zarr"
        if not zp.exists():
            continue
        import zarr

        arr = zarr.open_array(store=str(zp), mode="r")

        def z_full(arr=arr):
            a = arr[:]
            s = float(a.astype(np.float64).sum())
            del a
            return s

        def z_wins(arr=arr):
            acc = 0.0
            for s in starts:
                acc += float(arr[:, s : s + win].astype(np.float64).sum())
            return acc

        record(f"zarr_{lay}", z_full, z_wins, nbytes_f32)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = HERE / f"backends-{args.dir}-{stamp}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
