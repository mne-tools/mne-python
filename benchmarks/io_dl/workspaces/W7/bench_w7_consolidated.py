"""W7 consolidated single-session pass: all writers + floors share machine state.

Run 2-3 times; report medians-of-medians. ~30 s per pass.
"""

import gc
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
TREE = BENCH.parents[1]
sys.path.insert(0, str(TREE))

import mne  # noqa: E402

mne.set_log_level("ERROR")

DATA = BENCH / "data"
DATA_BIG = BENCH / "data_big"
TMP = Path(tempfile.mkdtemp(prefix="w7c_"))


def med_ms(fn, reps):
    ts = []
    for _ in range(reps):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
        gc.enable()
    return float(np.median(ts))


def main():
    out = {}
    small = np.fromfile(DATA / "bench.bin", dtype=np.float32)
    big = np.fromfile(DATA_BIG / "bench_big.bin", dtype=np.float32)
    fpath = TMP / "f.bin"

    # floors
    out["floor_small"] = med_ms(lambda: small.tofile(fpath), 9)
    out["floor_big"] = med_ms(lambda: big.tofile(fpath), 5)

    # FIF small (preloaded)
    from mne.io import read_raw_edf

    raw_s = read_raw_edf(DATA / "bench.edf", preload=True)
    o = TMP / "s.fif"
    out["fif_small"] = med_ms(
        lambda: raw_s.save(o, fmt="single", overwrite=True), 9)
    size_small = o.stat().st_size

    # FIF big
    raw_b = read_raw_edf(DATA_BIG / "bench_big.edf", preload=True)
    ob = TMP / "b.fif"
    out["fif_big"] = med_ms(
        lambda: raw_b.save(ob, fmt="single", overwrite=True), 5)
    size_big = ob.stat().st_size
    del raw_b

    # BV (installed pybv; caller copy included, subtract separately)
    import pybv

    ch128 = [f"EEG{i:03d}" for i in range(128)]
    volts = big.reshape(128, -1) * np.float32(1e-6)

    def w_bv():
        pybv.write_brainvision(data=volts.copy(), folder_out=str(TMP),
                               fname_base="v", sfreq=512, ch_names=ch128,
                               fmt="binary_float32", resolution=1e-7,
                               unit="µV", overwrite=True)

    w_bv()  # warm
    for p in TMP.glob("v.*"):
        p.unlink()
    out["bv_copy_only"] = med_ms(lambda: volts.copy(), 9)
    out["bv_installed"] = med_ms(w_bv, 5)
    for p in TMP.glob("v.*"):
        p.unlink()

    # EDF end-to-end (construct+convert+write) and write-only
    from edfio import Edf, EdfSignal

    d64 = big.reshape(128, -1).astype(np.float64)
    pmin, pmax = float(d64.min()) - 1.0, float(d64.max()) + 1.0
    chn = [f"EEG{i:03d}" for i in range(128)]

    def mk():
        return Edf(signals=[
            EdfSignal(data=d64[i].copy(), sampling_frequency=512.0,
                      physical_range=(pmin, pmax), label=chn[i],
                      physical_dimension="uV") for i in range(128)
        ])

    oe = TMP / "e.edf"
    obj = mk()

    def w_e():
        obj.write(oe)

    w_e()
    oe.unlink()
    out["edf_write_only"] = med_ms(w_e, 5)
    out["edf_e2e"] = med_ms(lambda: mk().write(oe), 5)
    oe.unlink()

    # BV-layout floor incl. transposition
    out["floor_bv_layout"] = med_ms(
        lambda: volts.ravel(order="F").tofile(TMP / "fl.eeg"), 5)
    (TMP / "fl.eeg").unlink()

    shutil.rmtree(TMP, ignore_errors=True)

    mb = lambda x: x / 1e6
    print(f"\n=== W7 single-session pass ===")
    print(f"{'measurement':<22} {'payload':>9} {'median':>9} {'MB/s':>7}")
    rows = [
        ("floor_small", mb(small.nbytes)),
        ("floor_big", mb(big.nbytes)),
        ("fif_small", mb(size_small)),
        ("fif_big", mb(size_big)),
        ("bv_copy_only", mb(big.nbytes)),
        ("bv_installed", mb(big.nbytes)),
        ("edf_write_only", 236.0),
        ("edf_e2e", 236.0),
        ("floor_bv_layout", mb(big.nbytes)),
    ]
    for k, payload in rows:
        ms = out[k]
        rate = payload / (ms / 1e3)
        print(f"{k:<22} {payload:>8.1f}M {ms:>8.2f} {rate:>7.0f}")

    b = np.array([size_small, size_big], float)
    t = np.array([out["fif_small"], out["fif_big"]])
    slope = (t[1] - t[0]) / (b[1] - b[0])  # ms per byte
    fixed = t[0] - slope * b[0]
    print(f"\nFIF two-point fit: fixed={fixed:.2f} ms/file, "
          f"marginal={1.0 / slope / 1e6 * 1e3:.0f} MB/s")
    print(f"FIF small fixed-overhead share: {100 * fixed / t[0]:.0f}%")
    print(f"FIF big vs floor: {out['floor_big'] / out['fif_big'] * 100:.0f}% of"
          f" disk floor")
    print(f"BV installed vs its-layout floor: "
          f"{out['floor_bv_layout'] / out['bv_installed'] * 100:.0f}%")
    (HERE / "consolidated.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
