"""Benchmark MNE-Python raw IO under deep-learning access patterns.

Run generate_data.py first. Then:

    python benchmarks/io_dl/bench_io.py            # full suite
    python benchmarks/io_dl/bench_io.py --quick    # smoke test
    python benchmarks/io_dl/bench_io.py --curve    # fixed-overhead curve only

Scenarios
---------
open_meta        : read_raw_X(preload=False)   -- header/metadata parse only
full_load        : read_raw_X(preload=True)    -- end-to-end load
seq_1s           : sequential 1 s get_data() over whole file (streaming/eval)
rand_windows     : 300 shuffled 2 s get_data() calls, preload=False
                   (the canonical DL training-loop pattern)
preloaded_windows: same windows on an already-preloaded raw (access-only cost)
floors           : numpy memmap / np.fromfile equivalents on identical bytes

Every number is median of N repetitions (min shown too), GC disabled.
"""

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# Always benchmark THIS working tree, regardless of interpreter resolution
# (script dir / cwd / PYTHONPATH pitfalls).
TREE_ROOT = HERE.resolve().parents[1]
sys.path.insert(0, str(TREE_ROOT))

import mne

mne.set_log_level("ERROR")
from mne.io import read_raw_edf, read_raw_fif, read_raw_brainvision  # noqa: E402


# ---------------------------------------------------------------- utilities
def timed(fn, repeats=5, warmup=1):
    """Return list of runtimes in ms."""
    for _ in range(warmup):
        fn()
    out = []
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            fn()
            out.append((time.perf_counter_ns() - t0) / 1e6)
    finally:
        if was_enabled:
            gc.enable()
    return np.asarray(out)


def fmt(ms):
    med = np.median(ms)
    mn = ms.min()
    return f"{med:9.2f} / {mn:8.2f}"


def row(label, unit, val_ms, extra=""):
    print(f"  {label:<34}{fmt(val_ms)} {unit:<12}{extra}")


# ---------------------------------------------------------------- readers
def make_readers(data_dir):
    return {
        "edf": lambda preload: read_raw_edf(
            data_dir / "bench.edf", preload=preload, verbose="ERROR"
        ),
        "brainvision": lambda preload: read_raw_brainvision(
            data_dir / "bench.vhdr", preload=preload, verbose="ERROR"
        ),
        "fif": lambda preload: read_raw_fif(
            data_dir / "bench_raw.fif", preload=preload, verbose="ERROR"
        ),
    }


def rand_starts(rng, n_times, win, n_win):
    return rng.integers(0, n_times - win - 1, size=n_win)


# ---------------------------------------------------------------- scenarios
def bench_format(name, reader, meta, cfg):
    n_times, sfreq = meta["n_times"], meta["sfreq"]
    res = {}

    # S0: metadata-only open
    res["open_meta"] = timed(lambda: reader(preload=False), cfg["repeats"])
    # S1: full load
    res["full_load"] = timed(lambda: reader(preload=True), cfg["repeats"])

    # Steady-state access patterns: Raw objects are opened ONCE (as DL
    # dataset classes do); we time only the data-access loop.

    # S3/S4: random 2 s windows <-- canonical DL training pattern
    rng = np.random.default_rng(cfg["seed"])
    starts = [int(s) for s in rand_starts(rng, n_times, cfg["win"], cfg["n_win"])]

    def rwin(raw):
        acc = 0.0
        for s in starts:
            acc += float(raw.get_data(start=s, stop=s + cfg["win"]).sum())
        return acc

    def seq(raw_nop):
        acc = 0.0
        for i in range(int(meta["dur"])):
            acc += float(
                raw_nop.get_data(start=i * int(sfreq), stop=(i + 1) * int(sfreq)).sum()
            )
        return acc

    # S2: sequential streaming over 1 s chunks, no preload
    raw_seq = reader(preload=False)
    res["seq_1s"] = timed(lambda: seq(raw_seq), max(3, cfg["repeats"] - 2))

    raw_rand = reader(preload=False)
    res["rand_windows"] = timed(lambda: rwin(raw_rand), cfg["repeats"])

    raw_pre = reader(preload=True)
    res["preloaded_windows"] = timed(lambda: rwin(raw_pre), cfg["repeats"])
    return res


def bench_floors(meta, cfg):
    """Numpy-only baselines on the identical float32 payload."""
    n_ch, n_times = meta["n_ch"], meta["n_times"]
    bin_path = HERE / "data" / "bench.bin"
    rng = np.random.default_rng(cfg["seed"])
    starts = rand_starts(rng, n_times, cfg["win"], cfg["n_win"])
    shape = (n_ch, n_times)
    res = {}

    def mm_windows():
        mm = np.memmap(bin_path, dtype="<f4", mode="r", shape=shape)
        acc = 0.0
        for s in starts:
            w = mm[:, s : s + cfg["win"]]
            acc += float(w.astype(np.float64).sum())
        del mm
        return acc

    res["floor_memmap"] = timed(mm_windows, cfg["repeats"])

    itemsize = 4

    def fromfile_windows():
        acc = 0.0
        with open(bin_path, "rb", buffering=0) as f:
            for s in starts:
                f.seek(s * n_ch * itemsize)
                b = np.fromfile(f, "<f4", n_ch * cfg["win"])
                w = b.reshape(n_ch, -1).T.astype(np.float64)
                acc += float(w.sum())
        return acc

    res["floor_fromfile"] = timed(fromfile_windows, cfg["repeats"])

    def full_fromfile():
        a = np.fromfile(bin_path, "<f4").reshape(shape).T.astype(np.float64)
        return float(a.sum())

    res["floor_full_read"] = timed(full_fromfile, cfg["repeats"])
    return res


def bench_micro(cfg):
    """Explicit copies that show up in DL pipelines."""
    data_dir = HERE / "data"
    raw = read_raw_edf(data_dir / "bench.edf", preload=False, verbose="ERROR")
    res = {}
    res["deepcopy_info"] = timed(lambda: __import__("copy").deepcopy(raw.info), 20, 3)
    res["info_copy_method"] = timed(lambda: raw.info.copy(), 20, 3)
    res["raw_copy"] = timed(lambda: raw.copy(), 10, 2)

    def save_fif():
        raw.save(HERE / "data" / "_tmp_save.fif", overwrite=True, verbose="ERROR")

    res["save_to_fif"] = timed(save_fif, 5, 1)
    (HERE / "data" / "_tmp_save.fif").unlink(missing_ok=True)
    return res


def bench_curve(readers, meta, cfg):
    """Time-per-call vs window length: exposes fixed per-call overhead."""
    sizes_s = [0.125, 0.5, 2, 8, 32]
    rng = np.random.default_rng(cfg["seed"])
    print("\n=== per-get_data() latency (ms/call), preload=False ===")
    print(f"{'window':>8} | " + " | ".join(f"{k:>12}" for k in readers))
    rows = []
    for wlen in sizes_s:
        win = int(wlen * meta["sfreq"])
        vals = []
        for name, reader in readers.items():
            starts = rand_starts(rng, meta["n_times"], win, cfg["n_win"])
            raw = reader(preload=False)

            def run():
                for s in starts:
                    raw.get_data(start=int(s), stop=int(s) + win)

            ms_per_call = np.median(timed(run, 3, 1)) / cfg["n_win"]
            vals.append(ms_per_call)
        rows.append((wlen, vals))
        print(f"{wlen:>7.3f}s | " + " | ".join(f"{v:12.3f}" for v in vals))
    return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--n-win", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    cfg = dict(
        seed=args.seed,
        repeats=args.repeats or (2 if args.quick else 5),
        n_win=args.n_win or (50 if args.quick else 300),
        win=512,  # 2 s @ 256 Hz, typical DL window
    )

    meta = json.loads((HERE / "data" / "meta.json").read_text())
    dur = meta["dur"]
    readers = make_readers(HERE / "data")

    print(f"\nmne {mne.__version__} | numpy {np.__version__} | "
          f"{platform.machine()} | py {platform.python_version()}")
    print(f"mne loaded from: {Path(mne.__file__).parent}")
    print(f"data: {meta['n_ch']} ch, {meta['sfreq']:.0f} Hz, {dur:.0f} s | "
          f"win={cfg['win']} samples ({cfg['win']/meta['sfreq']:.1f} s), "
          f"n_win={cfg['n_win']}")

    hdr = f"{'scenario':<36}{'median/min (ms)':<24}{'unit':<12}"
    all_results = {"config": cfg, "meta": {k: v for k, v in meta.items() if k != "ch_names"}}

    if not args.curve:
        print(f"\n=== floors (numpy on identical float32 bytes) ===\n{hdr}")
        floors = bench_floors(meta, cfg)
        for k, v in floors.items():
            row(k, "ms", v)
        all_results.update({k: v.tolist() for k, v in floors.items()})

        for name, reader in readers.items():
            print(f"\n=== format: {name} ===\n{hdr}")
            r = bench_format(name, reader, meta, cfg)
            for k, v in r.items():
                n_calls = cfg["n_win"] if "windows" in k else (int(dur) if k == "seq_1s" else 1)
                extra = f"({v.mean()/max(n_calls,1)*1000:8.1f} us/call)" if n_calls > 1 else ""
                row(k, "ms", v, extra)
            all_results[f"fmt_{name}"] = {k: v.tolist() for k, v in r.items()}

        print(f"\n=== micro (copies) ===\n{hdr}")
        mic = bench_micro(cfg)
        for k, v in mic.items():
            row(k, "ms", v)
        all_results["micro"] = {k: v.tolist() for k, v in mic.items()}

    bench_curve(readers, meta, cfg)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = HERE / f"results-{stamp}.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
