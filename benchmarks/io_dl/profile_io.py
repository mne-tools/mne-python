"""Profile MNE IO hot paths and trace every deepcopy on them.

    python benchmarks/io_dl/profile_io.py            # all profiles
    python benchmarks/io_dl/profile_io.py --only windows

Sections
--------
A. cProfile of steady-state random-window access (preloaded and not)
B. cProfile of full_load per format
C. cProfile of raw.save() to fif (DL caching path)
D. deepcopy tracer: counts + cumulative time + call sites of
   copy.deepcopy during common DL pipeline operations
"""

import argparse
import cProfile
import io as _io
import pstats
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# Always profile THIS working tree.
TREE_ROOT = HERE.resolve().parents[1]
sys.path.insert(0, str(TREE_ROOT))

import mne

mne.set_log_level("ERROR")
from mne.io import read_raw_brainvision, read_raw_edf, read_raw_fif  # noqa: E402

READERS = {
    "edf": lambda pl: read_raw_edf(HERE / "data" / "bench.edf", preload=pl),
    "brainvision": lambda pl: read_raw_brainvision(
        HERE / "data" / "bench.vhdr", preload=pl
    ),
    "fif": lambda pl: read_raw_fif(HERE / "data" / "bench_raw.fif", preload=pl),
}


def show(prof, n=22, sort="cumulative"):
    s = _io.StringIO()
    st = pstats.Stats(prof, stream=s)
    st.sort_stats(sort).print_stats(n)
    txt = s.getvalue()
    # trim the header noise
    lines = [ln for ln in txt.splitlines()]
    start = next(i for i, ln in enumerate(lines) if "ncalls" in ln) - 1
    print("\n".join(lines[start : start + n + 2]))


def profile_windows(fmt, preloaded, n_win=200):
    rng = np.random.default_rng(7)
    meta_wins = 512
    import json

    meta = json.loads((HERE / "data" / "meta.json").read_text())
    starts = rng.integers(0, meta["n_times"] - meta_wins - 1, size=n_win)
    raw = READERS[fmt](preloaded)

    def loop():
        for s in starts:
            raw.get_data(start=int(s), stop=int(s) + meta_wins)

    label = f"{fmt} windows preloaded={preloaded} ({n_win} calls)"
    print(f"\n--- A: cProfile {label} ---")
    prof = cProfile.Profile()
    prof.enable()
    loop()
    prof.disable()
    show(prof)


def profile_full_load(fmt):
    print(f"\n--- B: cProfile {fmt} full_load(preload=True) ---")
    prof = cProfile.Profile()
    prof.enable()
    READERS[fmt](True)
    prof.disable()
    show(prof)


def profile_save():
    raw = READERS["edf"](False)
    out = HERE / "data" / "_tmp_save.fif"
    print("\n--- C: cProfile raw.save() -> fif ---")
    prof = cProfile.Profile()
    prof.enable()
    raw.save(out, overwrite=True)
    prof.disable()
    out.unlink(missing_ok=True)
    show(prof)


# ------------------------------------------------------------------ tracer
@contextmanager
def trace_deepcopy():
    """Log every deepcopy call reachable from loaded mne modules.

    Patches copy.deepcopy AND every module attribute that holds a direct
    reference (from `from copy import deepcopy` bindings), so call sites
    inside mne._fiff / mne.io are all captured.
    """
    import copy as _copy
    import sys

    records = []
    real = _copy.deepcopy

    def spy(x=None, memo=None, *args, **kwargs):
        t0 = time.perf_counter_ns()
        out = real(x, memo) if memo is not None else real(x)
        dt = (time.perf_counter_ns() - t0) / 1e6
        frame = sys._getframe(1)
        fname = frame.f_code.co_filename.split("mne-python/")[-1].split(
            "mne_python_more_io_speed/"
        )[-1]
        site = f"{fname}:{frame.f_lineno}"
        records.append((type(x).__name__, site, dt))
        return out

    patched = []
    _copy.deepcopy = spy
    for name, mod in list(sys.modules.items()):
        if mod is None or name.split(".")[0] != "mne":
            continue
        try:
            attrs = vars(mod)
        except TypeError:
            continue
        hit = False
        for k, v in list(attrs.items()):
            if v is real:
                attrs[k] = spy
                hit = True
        if hit:
            patched.append(mod)
    try:
        yield records
    finally:
        _copy.deepcopy = real
        for mod in patched:
            try:
                for k, v in list(vars(mod).items()):
                    if v is spy:
                        vars(mod)[k] = real
            except TypeError:
                pass


def report(records, title):
    print(f"\n--- D: deepcopy trace: {title} ---")
    if not records:
        print("  (no deepcopy calls)")
        return
    agg = {}
    for typ, site, dt in records:
        k = (typ, site)
        n, tot = agg.get(k, (0, 0.0))
        agg[k] = (n + 1, tot + dt)
    total = sum(t for *_, t in records)
    print(f"  total calls={len(records)}  total={total:.2f} ms")
    for (typ, site), (n, tot) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"  {tot:8.3f} ms  x{n:<4} {typ:<12} {site}")


def info_copy_scaling():
    """How does Info copy cost scale with channel count / montage?"""
    import copy as _copy

    print("\n--- D2: Info deepcopy scaling (us per copy) ---")
    print(f"{'n_ch':>6} {'montage':>8} {'deepcopy(us)':>13} {'copy()(us)':>12}")
    for n_ch in (32, 64, 128, 256, 512):
        ch_names = [f"EEG{i:03d}" for i in range(n_ch)]
        info = mne.create_info(ch_names, 256.0, "eeg")
        for with_montage in (False, True):
            if with_montage:
                mon = mne.channels.make_standard_montage("standard_1005")
                keep = list(mon.ch_names)[:n_ch]
                info = mne.create_info(keep, 256.0, "eeg")
                info.set_montage(mon)
            n_rep = 50
            t0 = time.perf_counter_ns()
            for _ in range(n_rep):
                _copy.deepcopy(info)
            t_dc = (time.perf_counter_ns() - t0) / n_rep / 1e3
            t0 = time.perf_counter_ns()
            for _ in range(n_rep):
                info.copy()
            t_cp = (time.perf_counter_ns() - t0) / n_rep / 1e3
            print(f"{n_ch:>6} {str(with_montage):>8} {t_dc:>13.1f} {t_cp:>12.1f}")


def trace_pipeline_ops():
    """Deepcopy accounting for ops a DL preprocessing/training loop does."""
    raw_e = READERS["edf"](False)
    raw_b = READERS["brainvision"](False)
    raw_f = READERS["fif"](False)

    with trace_deepcopy() as rec:
        for r in (raw_e, raw_b, raw_f):
            READERS[r.filenames[0].suffix.strip(".") == "edf"
                    and "edf" or
                    ("brainvision" if r.filenames[0].suffix == ".vhdr" else "fif")](True)
    report(rec, "read_raw_X(preload=True) x3")

    with trace_deepcopy() as rec:
        raw = READERS["edf"](True)
        raw.crop(tmax=60.0)
        raw.set_eeg_reference("average")
        raw.notch_filter([50.0])
        raw.resample(128)
    report(rec, "load + crop + reref + notch + resample (EDF)")

    with trace_deepcopy() as rec:
        raw = READERS["edf"](False).load_data()
        epochs = mne.Epochs(raw, mne.make_fixed_length_events(raw, duration=1.0, id=1),
                            tmin=0.0, tmax=1.99, baseline=None, preload=False,
                            reject_by_annotation=False)
        for ep in epochs[:20]:
            pass
    report(rec, "Epochs creation + iterate 20 (EDF)")
    info_copy_scaling()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["windows", "load", "save", "copies"])
    args = ap.parse_args()
    what = args.only

    if what in (None, "windows"):
        for fmt in READERS:
            profile_windows(fmt, False)
        for fmt in READERS:
            profile_windows(fmt, True)
    if what in (None, "load"):
        for fmt in READERS:
            profile_full_load(fmt)
    if what in (None, "save"):
        profile_save()
    if what in (None, "copies"):
        trace_pipeline_ops()


if __name__ == "__main__":
    main()
