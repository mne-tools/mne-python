"""W7: decompose write-path costs (FIF save, pybv BrainVision, edfio EDF) vs np.tofile floor.

    python benchmarks/io_dl/workspaces/W7/bench_w7_write.py            # everything
    python benchmarks/io_dl/workspaces/W7/bench_w7_write.py --only fif
    python benchmarks/io_dl/workspaces/W7/bench_w7_write.py --only bv
    python benchmarks/io_dl/workspaces/W7/bench_w7_write.py --only edf
    python benchmarks/io_dl/workspaces/W7/bench_w7_write.py --only floor

Method: medians over repeated runs, GC off, outputs written to /tmp and deleted.
Decomposition via accumulator-wrapping of mne internals (rebinding every module
attribute that holds the function, like profile_io.py's deepcopy tracer) and one
cProfile run for hotspot ranking. No library edits.
"""

import argparse
import cProfile
import copy as _copy
import gc
import io as _io
import json
import pstats
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

print("mne loaded from:", mne.__file__)
mne.set_log_level("ERROR")

DATA = BENCH / "data"
DATA_BIG = BENCH / "data_big"
TMP = Path(tempfile.mkdtemp(prefix="w7_write_"))

RESULTS = {}


def mb(x):
    return x / 1e6


def median_ms(fn, reps, *args, **kwargs):
    ts = []
    out = None
    for _ in range(reps):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1e3
        gc.enable()
        ts.append(dt)
    return float(np.median(ts)), ts, out


# --------------------------------------------------------------------------
# accumulator instrumentation: rebind every module attribute holding `func`
# --------------------------------------------------------------------------
class Acc:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.ms = 0.0
        self.calls = 0
        self.extra = 0  # generic counter (bytes for _write)

    def report(self):
        return f"{self.name:<28} {self.calls:>8} calls {self.ms:>9.2f} ms"


class Instrument:
    """Wrap target functions with timing accumulators; restore afterwards."""

    def __init__(self):
        self.accs = {}
        self.patched = []  # (holder, attrname, original)
        self.in_payload = False  # toggled by _write_raw_buffer spy
        self.payload_bytes = 0
        self.meta_bytes = 0

    def _make_spy(self, name, func):
        acc = self.accs[name]
        inst = self

        if name == "_write":
            def spy(fid, data, kind, data_size, FIFFT_TYPE, dtype):
                t0 = time.perf_counter_ns()
                r = func(fid, data, kind, data_size, FIFFT_TYPE, dtype)
                acc.ms += (time.perf_counter_ns() - t0) / 1e6
                acc.calls += 1
                try:
                    nb = int(np.asarray(data).nbytes) + 16
                except Exception:
                    nb = 16
                acc.extra += nb
                if inst.in_payload:
                    inst.payload_bytes += nb
                else:
                    inst.meta_bytes += nb
                return r
        elif name == "_write_raw_buffer":
            def spy(fid, buf, cals, fmt):
                inst.in_payload = True
                try:
                    t0 = time.perf_counter_ns()
                    r = func(fid, buf, cals, fmt)
                    acc.ms += (time.perf_counter_ns() - t0) / 1e6
                    acc.calls += 1
                finally:
                    inst.in_payload = False
                return r
        elif name == "deepcopy":
            real = func

            def spy(x=None, memo=None, *a, **kw):
                t0 = time.perf_counter_ns()
                r = real(x, memo) if memo is not None else real(x)
                acc.ms += (time.perf_counter_ns() - t0) / 1e6
                acc.calls += 1
                return r
        else:
            def spy(*args, **kwargs):
                t0 = time.perf_counter_ns()
                r = func(*args, **kwargs)
                acc.ms += (time.perf_counter_ns() - t0) / 1e6
                acc.calls += 1
                return r

        spy.__name__ = f"spy_{name}"
        return spy

    def __enter__(self):
        import mne.annotations as ann_mod
        import mne._fiff.meas_info as meas_info
        import mne._fiff.write as fiff_write
        import mne.io.base as io_base
        import mne.utils.check as check_mod

        targets = {
            "_check_fname": check_mod._check_fname,
            "check_fname": check_mod.check_fname,
            "_write": fiff_write._write,
            "write_meas_info": meas_info.write_meas_info,
            "_annotations_starts_stops": ann_mod._annotations_starts_stops,
            "_write_annotations": ann_mod._write_annotations,
            "_write_raw_buffer": io_base._write_raw_buffer,
            "deepcopy": _copy.deepcopy,
        }
        for name, func in targets.items():
            self.accs[name] = Acc(name)
            spy = self._make_spy(name, func)
            # rebind in every loaded module that references the function object
            n = 0
            for modname, mod in list(sys.modules.items()):
                if mod is None or modname.split(".")[0] != "mne":
                    continue
                try:
                    attrs = vars(mod)
                except TypeError:
                    continue
                for k, v in list(attrs.items()):
                    if v is func:
                        attrs[k] = spy
                        self.patched.append((mod, k, func))
                        n += 1
            # deepcopy lives in stdlib too
            if name == "deepcopy":
                import copy as c2

                if c2.deepcopy is func:
                    c2.deepcopy = spy
                    self.patched.append((c2, "deepcopy", func))
        return self

    def __exit__(self, *exc):
        for holder, k, orig in reversed(self.patched):
            try:
                vars(holder)[k] = orig
            except TypeError:
                setattr(holder, k, orig)
        self.patched.clear()

    def dump(self, total_ms, payload_mb):
        print(f"  {'component':<28} {'calls':>8} {'ms':>9} {'%total':>7}")
        for a in self.accs.values():
            pct = 100 * a.ms / total_ms if total_ms else 0
            print(f"  {a.name:<28} {a.calls:>8} {a.ms:>9.2f} {pct:>6.1f}%")
        print(f"  tag bytes: payload={mb(self.payload_bytes):.1f} MB, "
              f"meta={mb(self.meta_bytes):.3f} MB")
        other = total_ms - sum(a.ms for a in self.accs.values())
        print(f"  {'(unattributed glue/loop)':<28} {'':>8} {other:>9.2f} "
              f"{100 * other / total_ms:>6.1f}%")


def show(prof, n=18, sort="tottime"):
    s = _io.StringIO()
    st = pstats.Stats(prof, stream=s)
    st.sort_stats(sort).print_stats(n)
    txt = s.getvalue()
    lines = txt.splitlines()
    start = next(i for i, ln in enumerate(lines) if "ncalls" in ln) - 1
    print("\n".join(lines[start : start + n + 2]))


# --------------------------------------------------------------------------
# FIF save decomposition
# --------------------------------------------------------------------------
def bench_fif():
    from mne.io import read_raw_edf

    meta = json.loads((DATA / "meta.json").read_text())
    ch_names = [f"EEG{i:03d}" for i in range(meta["n_ch"])]

    raw_small = read_raw_edf(DATA / "bench.edf", preload=True)
    out_small = TMP / "w7_small_raw.fif"
    n_rep = 7
    med, ts, _ = median_ms(
        lambda: raw_small.save(out_small, fmt="single", overwrite=True), n_rep
    )
    size_small = out_small.stat().st_size
    for p in TMP.glob("w7_small*"):
        p.unlink()
    rate = size_small / (med / 1e3) / 1e6
    RESULTS["fif_small"] = dict(file_mb=mb(size_small), ms=med,
                                mbps=rate, reps_ms=ts)
    print(f"\n[FIF small] {mb(size_small):.1f} MB in {med:.1f} ms "
          f"(median of {n_rep}) -> {rate:.0f} MB/s")
    print("  reps ms:", ", ".join(f"{t:.1f}" for t in ts))

    # proj=True variant (exercises info deepcopy path)
    med_proj, _, _ = median_ms(
        lambda: raw_small.save(TMP / "w7p_raw.fif", fmt="single",
                               proj=True, overwrite=True), 3
    )
    (TMP / "w7p_raw.fif").unlink(missing_ok=True)
    print(f"[FIF small proj=True] {med_proj:.1f} ms (vs {med:.1f} default)")
    RESULTS["fif_small_proj"] = dict(ms=med_proj)

    # instrumented small save (one rep; overhead ~us/call)
    with Instrument() as ins:
        t0 = time.perf_counter()
        raw_small.save(out_small, fmt="single", overwrite=True)
        tot = (time.perf_counter() - t0) * 1e3
    out_small.unlink()
    print(f"\n[FIF small decomposed] total {tot:.1f} ms (instrumented)")
    ins.dump(tot, mb(size_small))
    RESULTS["fif_small_decomp"] = {
        a.name: dict(calls=a.calls, ms=a.ms) for a in ins.accs.values()
    } | {"payload_bytes": ins.payload_bytes, "meta_bytes": ins.meta_bytes}

    # cProfile cross-check
    prof = cProfile.Profile()
    prof.enable()
    raw_small.save(out_small, fmt="single", overwrite=True)
    prof.disable()
    out_small.unlink()
    print("\n[FIF small cProfile top tottime]")
    show(prof, n=16)

    # ---- BIG ----
    print("\nloading BIG edf (preload) ...", flush=True)
    t0 = time.perf_counter()
    raw_big = read_raw_edf(DATA_BIG / "bench_big.edf", preload=True)
    print(f"  loaded in {time.perf_counter() - t0:.1f} s")
    out_big = TMP / "w7_big_raw.fif"
    med_b, ts_b, _ = median_ms(
        lambda: raw_big.save(out_big, fmt="single", overwrite=True), 3
    )
    size_big = out_big.stat().st_size
    out_big.unlink()
    rate_b = size_big / (med_b / 1e3) / 1e6
    RESULTS["fif_big"] = dict(file_mb=mb(size_big), ms=med_b, mbps=rate_b,
                              reps_ms=ts_b)
    print(f"[FIF big] {mb(size_big):.1f} MB in {med_b:.1f} ms "
          f"(median of 3) -> {rate_b:.0f} MB/s")

    with Instrument() as ins:
        t0 = time.perf_counter()
        raw_big.save(out_big, fmt="single", overwrite=True)
        tot_b = (time.perf_counter() - t0) * 1e3
    out_big.unlink()
    print(f"[FIF big decomposed] total {tot_b:.1f} ms (instrumented)")
    ins.dump(tot_b, mb(size_big))
    RESULTS["fif_big_decomp"] = {
        a.name: dict(calls=a.calls, ms=a.ms) for a in ins.accs.values()
    }

    del raw_big

    # two-point fixed/marginal fit: t(bytes) = fixed + bytes/marginal_rate
    b = np.array([size_small, size_big], dtype=float)
    t = np.array([med, med_b])
    A = np.vstack([np.ones_like(b), b]).T
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    fixed_ms, per_byte = coef
    RESULTS["fif_fit"] = dict(fixed_ms=float(fixed_ms),
                              marginal_mbps=float(1e3 / per_byte / 1e6))
    print(f"\n[FIF linear fit] fixed ≈ {fixed_ms:.1f} ms/file; marginal ≈ "
          f"{1e3 / per_byte / 1e6:.0f} MB/s")
    pct_small = 100 * fixed_ms / med
    print(f"[FIF small fixed overhead] ≈ {pct_small:.0f}% of the "
          f"{med:.1f} ms small save")


def bench_fif_extra():
    """Non-preloaded save (the historical ~103ms case?), per-buffer fetch,
    cold-start single save in a fresh interpreter."""
    from mne.io import read_raw_edf

    meta = json.loads((DATA / "meta.json").read_text())

    # (a) non-preloaded source -> save reads through the EDF reader per buffer
    raw_nop = read_raw_edf(DATA / "bench.edf", preload=False)
    out = TMP / "w7n_raw.fif"
    med, ts, _ = median_ms(
        lambda: raw_nop.save(out, fmt="single", overwrite=True), 5
    )
    size = out.stat().st_size
    out.unlink()
    print(f"\n[FIF small NON-preloaded] {mb(size):.1f} MB in {med:.1f} ms "
          f"(median of 5) -> {size / (med / 1e3) / 1e6:.0f} MB/s")
    print("  reps ms:", ", ".join(f"{t:.1f}" for t in ts))
    RESULTS["fif_small_nopreload"] = dict(ms=med,
                                          mbps=size / (med / 1e3) / 1e6)

    with Instrument() as ins:
        t0 = time.perf_counter()
        raw_nop.save(out, fmt="single", overwrite=True)
        tot = (time.perf_counter() - t0) * 1e3
    out.unlink()
    print(f"[FIF small NON-preloaded decomposed] total {tot:.1f} ms")
    ins.dump(tot, mb(size))

    # (b) instrumented preloaded save incl. per-buffer fetch (__getitem__)
    raw_small = read_raw_edf(DATA / "bench.edf", preload=True)
    import mne.io.base as io_base

    orig_gi = io_base.BaseRaw.__getitem__
    acc = Acc("BaseRaw.__getitem__")

    def gi_spy(*a, **kw):
        t0 = time.perf_counter_ns()
        r = orig_gi(*a, **kw)
        acc.ms += (time.perf_counter_ns() - t0) / 1e6
        acc.calls += 1
        return r

    io_base.BaseRaw.__getitem__ = gi_spy
    try:
        with Instrument() as ins:
            t0 = time.perf_counter()
            raw_small.save(out, fmt="single", overwrite=True)
            tot = (time.perf_counter() - t0) * 1e3
    finally:
        io_base.BaseRaw.__getitem__ = orig_gi
    out.unlink()
    print(f"\n[FIF small preloaded +fetch] total {tot:.1f} ms; "
          f"__getitem__: {acc.calls} calls {acc.ms:.2f} ms")
    ins.dump(tot, mb(size))
    RESULTS["fif_small_fetch"] = dict(total_ms=tot, getitem_calls=acc.calls,
                                      getitem_ms=acc.ms)

    # (c) cold start: fresh process, one save, wall time end-to-end
    import subprocess

    code = (
        "import sys, time; sys.path.insert(0, %r);"
        "import mne; mne.set_log_level('ERROR');"
        "from mne.io import read_raw_edf;"
        "t0=time.perf_counter();"
        "raw=read_raw_edf(%r, preload=True); t1=time.perf_counter();"
        "raw.save(%r, fmt='single', overwrite=True); t2=time.perf_counter();"
        "print(f'{(t1-t0)*1e3:.1f} {(t2-t1)*1e3:.1f}')"
    ) % (str(TREE), str(DATA / "bench.edf"), str(TMP / "w7c_raw.fif"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=120)
    load_ms, save_ms = r.stdout.split()
    (TMP / "w7c_raw.fif").unlink(missing_ok=True)
    print(f"\n[FIF cold process] load {load_ms} ms + save {save_ms} ms "
          f"(single-shot, includes imports/page-cache warm-up)")
    RESULTS["fif_cold"] = dict(load_ms=float(load_ms), save_ms=float(save_ms))


def bench_bv():
    import pybv

    meta = json.loads((DATA_BIG / "meta.json").read_text())
    n_ch, sfreq = meta["n_ch"], int(meta["sfreq"])
    ch_names = [f"EEG{i:03d}" for i in range(n_ch)]
    # pybv expects VOLTS; fresh copy per call because the locally-patched
    # installed pybv scales `data` IN PLACE (caller-visible mutation).
    base_uv = np.fromfile(DATA_BIG / "bench_big.bin", dtype=np.float32).reshape(
        n_ch, -1
    )
    payload = base_uv.nbytes
    print(f"\n[BV] payload {mb(payload):.1f} MB f32, pybv "
          f"{getattr(pybv, '__version__', '?')}")

    def make_w():
        volts = base_uv * 1e-6  # prepare once, outside timed region

        def w():
            # fresh copy per call: installed pybv scales `data` IN PLACE
            pybv.write_brainvision(data=volts.copy(), folder_out=str(TMP),
                                   fname_base="w7bv", sfreq=sfreq,
                                   ch_names=ch_names, fmt="binary_float32",
                                   events=None, resolution=1e-7, unit="µV",
                                   overwrite=True)
        return w

    w = make_w()

    # warm once (imports etc.), then time; delete between runs
    w()
    for p in TMP.glob("w7bv.*"):
        p.unlink()

    n_rep = 5
    med, ts, _ = median_ms(w, n_rep)
    eeg = (TMP / "w7bv.eeg").stat().st_size
    for p in TMP.glob("w7bv.*"):
        p.unlink()
    rate = payload / (med / 1e3) / 1e6
    RESULTS["bv_big"] = dict(payload_mb=mb(payload), ms=med, mbps=rate,
                             eeg_mb=mb(eeg))
    print(f"[BV big] wrote .eeg {mb(eeg):.1f} MB in {med:.1f} ms "
          f"(median of {n_rep}) -> {rate:.0f} MB/s (of {mb(payload):.0f} MB payload)")

    # decompose: time _write_bveeg_file alone vs header writing
    import pybv.io as pio

    acc = {"bveeg": 0.0, "calls": 0}
    orig = pio._write_bveeg_file

    def spy(*a, **kw):
        t0 = time.perf_counter_ns()
        r = orig(*a, **kw)
        acc["bveeg"] += (time.perf_counter_ns() - t0) / 1e6
        acc["calls"] += 1
        return r

    pio._write_bveeg_file = spy
    try:
        med2, _, _ = median_ms(w, n_rep)
    finally:
        pio._write_bveeg_file = orig
    for p in TMP.glob("w7bv.*"):
        p.unlink()
    bveeg_per_call = acc["bveeg"] / n_rep
    print(f"[BV decomposed] binary .eeg write: {bveeg_per_call:.1f} ms of "
          f"{med2:.1f} ms total ({100 * bveeg_per_call / med2:.0f}%); "
          f"headers/vmrk outside: {med2 - bveeg_per_call:.1f} ms")
    RESULTS["bv_decomp"] = dict(total_ms=med2, bveeg_ms=bveeg_per_call)

    # cost of the per-call defensive copy we must make (installed pybv mutates)
    volts = (base_uv * 1e-6)
    copy_med, _, _ = median_ms(lambda: volts.copy(), 5)
    print(f"[BV caller-side copy] {copy_med:.1f} ms (included in totals above; "
          f"pybv mutates its input)")
    RESULTS["bv_copy_ms"] = copy_med

    # BV-layout floor: multiplexed (time-major) write incl. transposition
    floor_f, _, _ = median_ms(lambda: volts.ravel(order="F").tofile(
        TMP / "w7floor.eeg"), 3)
    (TMP / "w7floor.eeg").unlink()
    print(f"[BV layout floor] ravel(F)+tofile: {floor_f:.1f} ms -> "
          f"{mb(payload) / (floor_f / 1e3):.0f} MB/s")

    # scaling-pass cost estimate: the two in-place multiplies + range checks
    scaled = base_uv.copy()
    scales = np.ones((n_ch, 1))
    t0 = time.perf_counter()
    scaled *= scales
    t_mult1 = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    scaled *= 1e-6
    t_mult2 = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    ok = np.all(scaled >= np.finfo(np.float32).min) and np.all(
        scaled <= np.finfo(np.float32).max)
    t_range = (time.perf_counter() - t0) * 1e3
    del scaled, ok
    print(f"[BV internal passes] mult1 {t_mult1:.1f} ms + mult2 {t_mult2:.1f}"
          f" ms + range checks {t_range:.1f} ms "
          f"(≈{t_mult1 + t_mult2 + t_range:.0f} ms of pure memory passes)")
    RESULTS["bv_passes"] = dict(mult1=t_mult1, mult2=t_mult2, range=t_range)

    # pristine upstream pybv (3 internal full-array copies) via subprocess
    import subprocess

    up = Path("/tmp/pybv_up/x")
    if up.exists():
        code = (
            "import sys, time, tempfile, shutil, gc, numpy as np;"
            "sys.path.insert(0, %r); import pybv;"
            "print('using', pybv.__file__);"
            "d=np.fromfile(%r, dtype=np.float32).reshape(128, -1);"
            "ch=[f'EEG{i:03d}' for i in range(128)];"
            "tmp=tempfile.mkdtemp();"
            "def w():"
            "    d2=(d*1e-6).astype('f4');"
            "    pybv.write_brainvision(data=d2, folder_out=tmp, fname_base='p',"
            "sfreq=512, ch_names=ch, fmt='binary_float32', resolution=1e-7,"
            "overwrite=True);"
            "w(); ts=[];"
            "for i in range(3):"
            "    shutil.rmtree(tmp, ignore_errors=True); tmp=tempfile.mkdtemp();"
            "    gc.collect(); gc.disable(); t0=time.perf_counter(); w();"
            "    ts.append(time.perf_counter()-t0); gc.enable()"
            "print('pristine ms:', sorted(ts)[1] * 1e3)"
        ) % (str(up), str(DATA_BIG / "bench_big.bin"))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=300)
        out = r.stdout.strip().splitlines()
        for ln in out:
            if "using" in ln or "pristine" in ln:
                print(f"[BV {ln.split(':')[0].strip()}]")
                if "ms" in ln:
                    ms = float(ln.rsplit(":", 1)[1])
                    RESULTS["bv_pristine_ms"] = ms
                    print(f"   -> {mb(payload) / (ms / 1e3):.0f} MB/s")


def bench_edf():
    from edfio import Edf, EdfSignal

    meta = json.loads((DATA_BIG / "meta.json").read_text())
    n_ch, sfreq = meta["n_ch"], meta["sfreq"]
    ch_names = [f"EEG{i:03d}" for i in range(n_ch)]
    f32 = np.fromfile(DATA_BIG / "bench_big.bin", dtype=np.float32).reshape(
        n_ch, -1
    )
    data = f32.astype(np.float64)  # what generate_data feeds edfio
    pmin, pmax = float(data.min()) - 1.0, float(data.max()) + 1.0
    payload_i16 = data.size * 2

    def mk():
        return Edf(signals=[
            EdfSignal(data=data[i].copy(), sampling_frequency=sfreq,
                      physical_range=(pmin, pmax), label=ch_names[i],
                      physical_dimension="uV") for i in range(n_ch)
        ])

    def w(edf):
        edf.write(TMP / "w7.edf")

    edf_obj = mk()
    w(edf_obj)  # warm
    (TMP / "w7.edf").unlink()

    n_rep = 5
    med, ts, _ = median_ms(lambda: w(edf_obj), n_rep)
    size = (TMP / "w7.edf").stat().st_size
    (TMP / "w7.edf").unlink()
    rate = size / (med / 1e3) / 1e6
    RESULTS["edf_big"] = dict(file_mb=mb(size), ms=med, mbps=rate,
                              payload_i16_mb=mb(payload_i16))
    print(f"\n[EDF big] write-only {mb(size):.1f} MB (int16) in {med:.1f} ms "
          f"(median of {n_rep}) -> {rate:.0f} MB/s")
    print("  note: digital int16 conversion happens at EdfSignal CONSTRUCTION,"
          " not in write(); conversion cost measured separately below")

    def full():
        w(mk())

    med_e, _, _ = median_ms(full, n_rep)
    (TMP / "w7.edf").unlink()
    RESULTS["edf_big_e2e"] = dict(ms=med_e, mbps=size / (med_e / 1e3) / 1e6)
    print(f"[EDF big end-to-end (construct+convert+write)] {med_e:.1f} ms -> "
          f"{mb(size) / (med_e / 1e3):.0f} MB/s")

    # digital-conversion pass analog (what edfio must do before disk)
    res = (pmax - pmin) / 32767.0
    t0 = time.perf_counter()
    dig = ((data - pmin) / res).round().astype(np.int16)
    t_conv = (time.perf_counter() - t0) * 1e3
    del dig
    print(f"[EDF reference conversion pass] f64->i16 vectorized: "
          f"{t_conv:.1f} ms (memory-bound)")


def bench_floor():
    small = np.fromfile(DATA / "bench.bin", dtype=np.float32)
    big = np.fromfile(DATA_BIG / "bench_big.bin", dtype=np.float32)
    out = TMP / "floor.bin"

    med_s, _, _ = median_ms(lambda: small.tofile(out), 7)
    med_b, _, _ = median_ms(lambda: big.tofile(out), 5)
    out.unlink()
    print(f"\n[floor tofile] small {mb(small.nbytes):.1f} MB: {med_s:.1f} ms "
          f"-> {small.nbytes / (med_s / 1e3) / 1e6:.0f} MB/s")
    print(f"[floor tofile] big   {mb(big.nbytes):.1f} MB: {med_b:.1f} ms "
          f"-> {big.nbytes / (med_b / 1e3) / 1e6:.0f} MB/s")
    RESULTS["floor"] = dict(
        small_ms=med_s, small_mbps=small.nbytes / (med_s / 1e3) / 1e6,
        big_ms=med_b, big_mbps=big.nbytes / (med_b / 1e3) / 1e6,
    )

    # open/close syscall cost (per-save fixed floor)
    t0 = time.perf_counter()
    for i in range(200):
        with open(TMP / f"f{i}.bin", "wb"):
            pass
    t_open = (time.perf_counter() - t0) / 200 * 1e3
    for i in range(200):
        (TMP / f"f{i}.bin").unlink()
    print(f"[floor] bare open+close: {t_open * 1000:.0f} us")
    RESULTS["floor"]["open_us"] = t_open * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["fif", "fifx", "bv", "edf", "floor"])
    args = ap.parse_args()
    try:
        if args.only in (None, "floor"):
            bench_floor()
        if args.only in (None, "fif"):
            bench_fif()
        if args.only in (None, "fifx"):
            bench_fif_extra()
        if args.only in (None, "bv"):
            bench_bv()
        if args.only in (None, "edf"):
            bench_edf()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
        (HERE / "results.json").write_text(json.dumps(RESULTS, indent=2))
        print(f"\nresults -> {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
