"""Generate synthetic EEG files across many formats and sizes.

Same underlying signal written to every format. Content identical up to each
format's storage precision.

Two presets:
    default ("small"):  64 ch, 256 Hz, 300 s  -> data/
    --preset big:      128 ch, 512 Hz, 1800 s -> data_big/   (~470 MB float32)

Formats: EDF(int16), BDF(int24), BrainVision(f32), FIF(f32), NPZ(f32),
HDF5 f32 with three chunk layouts, Zarr f32 with two chunk layouts,
raw .bin floor.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def make_signal(n_ch: int, n_times: int, sfreq: float, seed: int = 42) -> np.ndarray:
    """Band-limited synthetic EEG-like signal, microvolt scale."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_ch, n_times))
    # smooth channel-by-channel without apply_along_axis (slow for BIG)
    csum = np.cumsum(data, axis=1, dtype=np.float64)
    csum[:, 9:] -= csum[:, :-9]
    data[:, :8] = csum[:, :8] / np.arange(1, 9)
    data[:, 8:] = csum[:, 8:] / 9.0
    del csum
    t = np.arange(n_times) / sfreq
    data += 10 * np.sin(2 * np.pi * 10 * t)
    return (data * 20.0).astype(np.float64)


def _edf_signals(sig_cls, data, ch_names, sfreq):
    pmin, pmax = float(data.min()) - 1.0, float(data.max()) + 1.0
    return [
        sig_cls(
            data=data[i].copy(),
            sampling_frequency=sfreq,
            physical_range=(pmin, pmax),
            label=ch_names[i],
            physical_dimension="uV",
        )
        for i in range(len(ch_names))
    ]


def write_edf(data, ch_names, sfreq, path):
    from edfio import Edf, EdfSignal

    Edf(signals=_edf_signals(EdfSignal, data, ch_names, sfreq)).write(path)


def write_bdf(data, ch_names, sfreq, path):
    from edfio import Bdf, BdfSignal

    Bdf(signals=_edf_signals(BdfSignal, data, ch_names, sfreq)).write(path)


def write_bv(data_f32, ch_names, sfreq, out_dir, base):
    import pybv

    pybv.write_brainvision(
        data=data_f32,
        folder_out=str(out_dir),
        fname_base=base,
        sfreq=int(sfreq),
        ch_names=ch_names,
        fmt="binary_float32",
        overwrite=True,
    )


def write_fif(data, ch_names, sfreq, path):
    import mne

    mne.set_log_level("ERROR")
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    raw.save(path, fmt="single", overwrite=True)


def write_h5(data_f32, path, layouts):
    import h5py

    n_ch, n_times = data_f32.shape
    with h5py.File(path, "w") as f:
        for name, chunks in layouts.items():
            if chunks is None:
                f.create_dataset(name, shape=(n_ch, n_times), dtype="f4")
                continue
            dset = f.create_dataset(name, shape=(n_ch, n_times),
                                    chunks=chunks, dtype="f4")
            dset[:] = data_f32


def write_npz(data_f32, path):
    np.savez(path, data=data_f32)



def write_eeglab(data_f32, ch_names, sfreq, out_dir, base):
    """Write EEGLAB .set (header) + .fdt (float32 multiplexed)."""
    import scipy.io as sio

    n_ch, n_times = data_f32.shape
    fdt_path = out_dir / f"{base}.fdt"
    np.ascontiguousarray(data_f32.T).tofile(fdt_path)  # Fortran order of (ch, t)
    chanlocs = np.empty(1, dtype=[("labels", "O"), ("type", "O"), ("unit", "O")])
    locs = np.zeros(n_ch, dtype=chanlocs.dtype)
    for i, ch in enumerate(ch_names):
        locs[i] = (ch, "EEG", "uV")
    eeg = {
        "nbchan": float(n_ch),
        "pnts": float(n_times),
        "trials": 1.0,
        "srate": float(sfreq),
        "xmin": 0.0,
        "xmax": (n_times - 1) / sfreq,
        "data": str(fdt_path.name),
        "ref": "n/a",
        "chanlocs": locs.reshape(1, -1),
        "chaninfo": {"nodatchans": {}},
        "event": np.empty((0, 0), dtype=object),
        "setname": base,
    }
    sio.savemat(out_dir / f"{base}.set", {"EEG": eeg}, appendmat=False)


def write_nwb(data_f32, ch_names, sfreq, path):
    from datetime import datetime, timezone

    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.ecephys import ElectricalSeries

    nwbfile = NWBFile(
        session_description="benchmark",
        identifier="bench",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device(name="bench_device")
    group = nwbfile.create_electrode_group(
        name="electrodes", description="all", location="unknown", device=device
    )
    for _ in ch_names:
        nwbfile.add_electrode(group=group, location="unknown")
    region = nwbfile.create_electrode_table_region(
        region=list(range(len(ch_names))), description="all channels"
    )
    es = ElectricalSeries(
        name="ElectricalSeries",
        data=data_f32.T,          # time-major, as required by NWB
        electrodes=region,
        starting_time=0.0,
        rate=sfreq,
    )
    nwbfile.add_acquisition(es)
    with NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ch", type=int, default=None)
    ap.add_argument("--sfreq", type=float, default=None)
    ap.add_argument("--dur", type=float, default=None, help="seconds")
    ap.add_argument(
        "--preset",
        choices=["small", "big"],
        default=None,
        help="small=64ch/256Hz/300s -> data/, big=128ch/512Hz/1800s -> data_big/",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--formats",
        type=str,
        default="bin,edf,bdf,bv,fif,npz,h5,zarr,set,nwb",
        help="comma list: bin,edf,bdf,bv,fif,npz,h5,zarr",
    )
    args = ap.parse_args()

    presets = {
        "small": dict(n_ch=64, sfreq=256.0, dur=300.0, out="data"),
        "big": dict(n_ch=128, sfreq=512.0, dur=1800.0, out="data_big"),
    }
    if args.preset:
        p = presets[args.preset]
        args.n_ch = args.n_ch or p["n_ch"]
        args.sfreq = args.sfreq or p["sfreq"]
        args.dur = args.dur or p["dur"]
        args.out_dir = args.out_dir or Path(__file__).parent / p["out"]
    args.n_ch = args.n_ch or 64
    args.sfreq = args.sfreq or 256.0
    args.dur = args.dur or 300.0
    args.out_dir = args.out_dir or Path(__file__).parent / "data"

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_times = int(args.sfreq * args.dur)
    ch_names = [f"EEG{i:03d}" for i in range(args.n_ch)]
    base = out_dir.name.replace("data", "bench")

    t0 = time.perf_counter()
    data = make_signal(args.n_ch, n_times, args.sfreq)
    print(f"signal: {args.n_ch} ch x {n_times} samples "
          f"({data.nbytes / 1e6:.0f} MB f64) in {time.perf_counter() - t0:.1f} s")

    meta = dict(n_ch=args.n_ch, sfreq=args.sfreq, dur=args.dur, n_times=n_times,
                formats=args.formats.split(","), seed=42)
    sizes = {}
    fmts = args.formats.split(",")

    def reg(name, paths):
        sizes[name] = sum(p.stat().st_size for p in paths if Path(p).exists())

    jobs = []
    if "bin" in fmts:
        binp = out_dir / f"{base}.bin"
        jobs.append(("bin", lambda p=binp: np.ascontiguousarray(
            data.astype(np.float32)).tofile(p), [binp]))
    if "edf" in fmts:
        p = out_dir / f"{base}.edf"
        jobs.append(("edf", lambda p=p: write_edf(data, ch_names, args.sfreq, p), [p]))
    if "bdf" in fmts:
        p = out_dir / f"{base}.bdf"
        jobs.append(("bdf", lambda p=p: write_bdf(data, ch_names, args.sfreq, p), [p]))
    if "bv" in fmts:
        jobs.append(("bv", lambda: write_bv(data.astype(np.float32), ch_names,
                    args.sfreq, out_dir, base),
                    list(out_dir.glob(f"{base}.v*")) + [out_dir / f"{base}.eeg"]))
    if "fif" in fmts:
        p = out_dir / f"{base}_raw.fif"
        jobs.append(("fif", lambda p=p: write_fif(data, ch_names, args.sfreq, p), [p]))
    if "npz" in fmts:
        p = out_dir / f"{base}.npz"
        jobs.append(("npz", lambda p=p: write_npz(data.astype(np.float32), p), [p]))
    if "h5" in fmts:
        p = out_dir / f"{base}.h5"
        layouts = {
            "t10s": (1, int(args.sfreq * 10)),   # per-channel time slabs
            "win": (args.n_ch, 512),             # window-shaped chunks
            "full": None,                        # contiguous
        }

        def wh(p=p, layouts=layouts):
            write_h5(data.astype(np.float32), p, layouts)

        jobs.append(("h5", wh, [p]))
    if "set" in fmts:

        def wset():
            write_eeglab(
                data.astype(np.float32), ch_names, args.sfreq, out_dir, base
            )

        jobs.append(("set", wset, list(out_dir.glob(f"{base}.s*"))
                     + [out_dir / f"{base}.fdt"]))
    if "nwb" in fmts:
        pnwb = out_dir / f"{base}.nwb"

        def wnwb(p=pnwb):
            write_nwb(data.astype(np.float32), ch_names, args.sfreq, p)

        jobs.append(("nwb", wnwb, [pnwb]))
    if "zarr" in fmts:
        layouts = {"t10s": (1, int(args.sfreq * 10)), "win": (args.n_ch, 512)}

        def wz(layouts=layouts):
            import zarr

            for name, chunks in layouts.items():
                arr = zarr.open_array(
                    store=str(out_dir / f"{base}_{name}.zarr"), mode="w",
                    shape=data.shape, chunks=chunks, dtype="f4",
                )
                arr[:] = data.astype(np.float32)

        jobs.append(("zarr", wz, list(out_dir.glob(f"{base}_*.zarr"))))

    for name, fn, paths in jobs:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        reg(name, paths)
        print(f"wrote {name:<5} {sizes[name] / 1e6:8.1f} MB in {dt:7.1f} s")

    (out_dir / "meta.json").write_text(json.dumps(meta | {"sizes": sizes}, indent=2))
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
