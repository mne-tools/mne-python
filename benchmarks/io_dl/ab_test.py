"""Clean interleaved A/B with explicit environment control.

Arm INSTALLED: site-packages mne (unpatched release copy)
Arm TREE:      working tree mne (patched)

Also reports component breakdown inside each process.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

TREE = "/Users/bruaristimunha/Projects/libraries/mne_python/mne_python_more_io_speed"
HERE = Path(__file__).parent

CHILD = r'''
import gc, json, os, sys, time
import numpy as np
which = sys.argv[1]
# `python -c` puts cwd ('') first on sys.path; remove anything that could
# resolve to the working tree unless this is the TREE arm
TREE = "{tree}"
if which == "tree":
    if TREE not in sys.path:
        sys.path.insert(0, TREE)
else:
    sys.path[:] = [p for p in sys.path if p not in ("", ".", TREE)]
    os.environ.pop("PYTHONPATH", None)
import mne
mne.set_log_level("ERROR")
fmt, pl = sys.argv[2], sys.argv[3] == "1"
READERS = {
    "edf": lambda p_: mne.io.read_raw_edf("{here}/data/bench.edf", preload=p_),
    "brainvision": lambda p_: mne.io.read_raw_brainvision("{here}/data/bench.vhdr", preload=p_),
    "fif": lambda p_: mne.io.read_raw_fif("{here}/data/bench_raw.fif", preload=p_),
}
raw = READERS[fmt](pl)
rng = np.random.default_rng(0)
starts = rng.integers(0, raw.n_times - 513, size=4000).astype(int)
stops = starts + 512

def bench(fn, warmup=300):
    for s in starts[:warmup]:
        fn(int(s))
    ts = []
    gc.disable()
    for s, e in zip(starts[warmup:], stops[warmup:]):
        t0 = time.perf_counter_ns()
        fn(s)
        ts.append(time.perf_counter_ns() - t0)
    gc.enable()
    arr = np.asarray(ts) / 1e3
    return dict(med=float(np.median(arr)), p10=float(np.percentile(arr, 10)))

out = dict(which=which, file=mne.__file__)
out["get_data_only"] = bench(lambda s: raw.get_data(start=s, stop=s + 512))
out["get_data_sum"] = bench(lambda s: raw.get_data(start=s, stop=s + 512).sum())
mm = np.memmap("{here}/data/bench.bin", dtype="<f4", mode="r", shape=(64, raw.n_times))
out["memmap_copy"] = bench(lambda s: mm[:, s:s + 512].astype(np.float64).sum())
print(json.dumps(out))
'''.replace("{tree}", TREE).replace("{here}", str(HERE))


FMT = os.environ.get("AB_FMT", "brainvision")
PRELOAD = os.environ.get("AB_PRELOAD", "1")


def run(which):
    env = {k: v for k, v in os.environ.items()}
    out = subprocess.run([sys.executable, "-c", CHILD, which, FMT, PRELOAD],
                         capture_output=True, text=True, env=env)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-800:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    acc = {"INSTALLED": [], "TREE": []}
    for i in range(rounds):
        row = {}
        for key, which in (("INSTALLED", "installed"), ("TREE", "tree")):
            r = run(which)
            acc[key].append(r)
            row[key] = r
            src = "TREE" if "more_io_speed" in r["file"] else "site-packages"
            print(f"round {i+1} {key:<10} src={src:<13} "
                  f"data={r['get_data_only']['med']:6.1f}us  "
                  f"data+sum={r['get_data_sum']['med']:6.1f}us  "
                  f"floor={r['memmap_copy']['med']:5.1f}us")
    print("\n=== best-of medians ===")
    for key in ("INSTALLED", "TREE"):
        best = min(acc[key], key=lambda r: r["get_data_only"]["med"])
        g, gs, f = best["get_data_only"]["med"], best["get_data_sum"]["med"], best["memmap_copy"]["med"]
        print(f"{key:<10} get_data={g:6.1f}us  get_data+sum={gs:6.1f}us  memmap-floor={f:5.1f}us")


if __name__ == "__main__":
    main()
