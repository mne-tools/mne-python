"""Compare get_data outputs: working-tree mne vs site-packages mne (oracle).

Run pair-wise via subprocess so both versions load in isolation.
"""

import hashlib
import json
import sys


def which_mne():
    import os

    if os.environ.get("USE_TREE_MNE") == "1":
        sys.path.insert(0, "/Users/bruaristimunha/Projects/libraries/"
                           "mne_python/mne_python_more_io_speed")
    import mne

    mne.set_log_level("ERROR")
    return mne, mne.__file__


def digests():
    mne, src = which_mne()
    here = __file__.rsplit("/", 1)[0]
    readers = {
        "edf": lambda: mne.io.read_raw_edf(f"{here}/data/bench.edf", preload=False),
        "brainvision": lambda: mne.io.read_raw_brainvision(
            f"{here}/data/bench.vhdr", preload=False
        ),
        "fif": lambda: mne.io.read_raw_fif(f"{here}/data/bench_raw.fif", preload=False),
    }
    out = {"mne_file": src}
    import numpy as np

    for name, reader in readers.items():
        raw_np = reader()
        raw_p = reader().load_data()
        cases = {}
        # full read
        d = raw_p.get_data()
        cases["full"] = hashlib.md5(d.tobytes()).hexdigest()[:12]
        # windows incl. block boundaries
        accs = []
        for s0, s1 in [(0, 512), (255, 769), (100000, 100512), (76288, 76800)]:
            a = raw_np.get_data(start=s0, stop=s1)
            b = raw_p.get_data(start=s0, stop=s1)
            accs.append(float((a - b).sum()))
            cases[f"win_{s0}"] = hashlib.md5(a.tobytes()).hexdigest()[:12]
        cases["win_delta"] = float(sum(abs(x) for x in accs))
        # tmin/tmax path
        a = raw_np.get_data(tmin=10.0, tmax=12.0)
        b = raw_p.get_data(tmin=10.0, tmax=12.0)
        cases["tmin_tmax"] = hashlib.md5(a.tobytes()).hexdigest()[:12]
        cases["tmin_tmax_delta"] = float(np.abs(a - b).sum())
        # picks variants
        cases["picks_int"] = hashlib.md5(
            raw_p.get_data(picks=[1, 5, 7], start=0, stop=256).tobytes()
        ).hexdigest()[:12]
        cases["picks_str"] = hashlib.md5(
            raw_p.get_data(picks=["EEG001", "EEG004"], start=0, stop=256).tobytes()
        ).hexdigest()[:12]
        cases["picks_slice"] = hashlib.md5(
            raw_p.get_data(picks=slice(2, 8), start=0, stop=256).tobytes()
        ).hexdigest()[:12]
        # negative index semantics
        cases["neg_idx"] = hashlib.md5(
            raw_p.get_data(picks=[-1], start=0, stop=64).tobytes()
        ).hexdigest()[:12]
        out[name] = cases
    return out


if __name__ == "__main__":
    res = digests()
    print(json.dumps(res, indent=2))
