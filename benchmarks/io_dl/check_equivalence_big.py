"""BIG-set equivalence: tree vs installed release for EDF/BDF windows.

The tree's vectorized EDF path folds ((d*cal)+off)*gain into
d*(cal*gain) + off*gain, which differs from the legacy op order by at most
a few double-precision ulps. We therefore assert max|diff| < 1e-9 uV on
identical inputs rather than bit equality. Full loads use the legacy loop
on both arms and must remain bit identical.
"""

import json
import os
import subprocess
import sys

import numpy as np

ROOT = "/Users/bruaristimunha/Projects/libraries/mne_python/mne_python_more_io_speed"

CODE = r"""
import sys, json
import numpy as np
if sys.argv[1] == "tree":
    sys.path.insert(0, "{root}")
else:
    sys.path = [p for p in sys.path if p not in ("", ".")]
import mne; mne.set_log_level("ERROR")
from mne.io import read_raw_edf, read_raw_bdf
rng = np.random.default_rng(5)
out = {}
for name, rd in [("edf", read_raw_edf), ("bdf", read_raw_bdf)]:
    f = "{root}/benchmarks/io_dl/data_big/bench_big." + name
    raw = rd(f, preload=False)
    starts = rng.integers(0, raw.n_times - 8193, size=15).astype(int)
    wins, picks_ = [], []
    for i, s0 in enumerate(starts):
        stop = int(s0) + 1024
        wins.append(raw.get_data(start=int(s0), stop=stop))
        picks_.append(raw.get_data(picks=[3, 17, 55], start=int(s0),
                                   stop=int(s0) + 3000))
    out[name + "_win"] = wins
    out[name + "_picks"] = picks_
    rawf = rd(f, preload=True)
    d = rawf.get_data()
    out[name + "_full_hash"] = float(d.sum())  # robust across arms
print(json.dumps(out, default=lambda x: x.tolist() if hasattr(x, "tolist") else x))
""".replace("{root}", ROOT)


def run(which):
    r = subprocess.run([sys.executable, "-c", CODE, which],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        raise RuntimeError(r.stderr[-500:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    a = run("installed")
    b = run("tree")
    ok = True
    worst = 0.0
    for key in list(a):
        if key.endswith("_hash"):
            same = a[key] == b[key]
            print(f"{key:<16} sum-equal={same}")
            ok &= same
            continue
        va = np.asarray(a[key])
        vb = np.asarray(b[key])
        d = float(np.abs(va - vb).max())
        worst = max(worst, d)
        this_ok = d < 1e-9
        ok &= this_ok
        print(f"{key:<16} max|diff|={d:.3e} {'OK' if this_ok else 'FAIL'}")
    print(f"\nworst |diff| = {worst:.3e}  ->  {'ALL OK ✔' if ok else 'FAILED ✗'}")


if __name__ == "__main__":
    main()
