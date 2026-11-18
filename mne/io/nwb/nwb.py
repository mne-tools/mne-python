"""Read a single NWB voltage series using h5py."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# Adapted from eegdash/io/nwb.py (https://github.com/eegdash/EEGDash), BSD-3-Clause.

from datetime import datetime, timedelta

import numpy as np

from mne._fiff.meas_info import create_info
from mne._fiff.utils import _mult_cal_one
from mne.annotations import Annotations
from mne.io.base import BaseRaw
from mne.utils import (
    _check_fname,
    _check_option,
    _import_h5py,
    _verbose_control,
    fill_doc_static,
    logger,
    warn,
)

_VOLTS = {
    **dict.fromkeys(("v", "volt", "volts"), 1.0),
    **dict.fromkeys(("mv", "millivolt", "millivolts"), 1e-3),
    **dict.fromkeys(("µv", "μv", "uv", "microvolt", "microvolts"), 1e-6),
}


@fill_doc_static("preload", "verbose")
def read_raw_nwb(fname, *, ch_type="seeg", series=None, preload=False, verbose=None):
    """Read an NWB voltage series using h5py.

    Parameters
    ----------
    fname : path-like
        Path to an NWB file.
    ch_type : str
        The channel type to use for all channels, e.g. ``"seeg"`` (default) or
        ``"ecog"``. NWB files do not store channel types.
    series : str | None
        The path of the series to read inside the file, e.g.
        ``"processing/ecephys/LFP/lfp"``. If ``None`` (default), read the only
        voltage series in ``acquisition``, or else the only ``ElectricalSeries``
        in the file.
    preload : bool | str
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, it is the name of a
        freshly created memory-mapped file used to store the data on the hard
        drive (slower, requires less memory). An existing file is overwritten.
        The caller owns the file and is responsible for removing it after the
        Raw object is no longer in use. For supported Raw readers, the exact string
        ``"auto"`` instead reuses decoded data below the directory configured by
        :func:`mne.set_cache_dir`. Entries persist without a size limit and are mapped
        copy-on-write. Use ``Path("auto")`` for a literal filename.

        .. versionchanged:: 1.13
           Support for the ``"auto"`` decoded-data cache was added.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    raw : instance of RawNWB
        The raw recording.

    See Also
    --------
    mne.io.read_raw : Read raw data from supported formats.

    Notes
    -----
    Requires ``h5py``. Without ``series``, reads the single ``ElectricalSeries``
    in the file's ``acquisition`` group, or the single two-dimensional
    ``TimeSeries`` in volts if there is no ``ElectricalSeries``. Channel names are
    the electrode IDs if available, otherwise channel numbers. Timestamps must be
    uniformly sampled. Each table in ``intervals`` (e.g. ``trials``) becomes
    annotations described as ``NWB/<table>``, except ``invalid_times``, which
    becomes ``BAD_nwb_invalid``; rows scoped to other series are skipped. The
    subject ID and sex are stored in ``raw.info["subject_info"]``.

    .. versionadded:: 1.14
    """
    return RawNWB(
        fname, ch_type=ch_type, series=series, preload=preload, verbose=verbose
    )


def _str(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _read_nwb_time(series, n_times):
    """Read the sampling frequency and first-sample time in seconds."""
    if "timestamps" in series:
        times = series["timestamps"][()]
        if times.shape != (n_times,) or n_times < 2:
            raise ValueError("Expected one NWB timestamp per sample")
        sfreq = (n_times - 1) / (times[-1] - times[0])
        error = np.abs(times - times[0] - np.arange(n_times) / sfreq).max()
        tolerance = max(1e-3 / sfreq, 8 * np.spacing(np.abs(times).max()))
        if not error <= tolerance:  # also catches NaN
            raise ValueError("NWB has nonuniform timestamps, which are not supported")
        t_start = times[0]
    else:
        clock = series["starting_time"]
        sfreq, t_start = clock.attrs["rate"], clock[()]
    if not 0 < sfreq < np.inf:
        raise ValueError(f"Expected a finite positive NWB sampling rate, got {sfreq}")
    return float(sfreq), float(t_start)


def _find_series(f, series):
    """Return the NWB voltage series to read."""
    h5py = _import_h5py()
    if series is not None:
        if not isinstance(f.get(series), h5py.Group) or "data" not in f[series]:
            raise ValueError(f"No NWB series with data at {series!r}")
        return f[series]
    by_type = {}
    for group in f.get("acquisition", {}).values():
        if not isinstance(group, h5py.Group) or "data" not in group:
            continue
        kind = group.attrs.get("neurodata_type")
        data = group["data"]
        unit = _str(data.attrs.get("unit", "")).strip().lower()
        if kind == "ElectricalSeries" or (
            kind == "TimeSeries" and data.ndim == 2 and unit in _VOLTS
        ):
            by_type.setdefault(kind, []).append(group)
    candidates = by_type.get("ElectricalSeries", by_type.get("TimeSeries", []))
    if not candidates:  # e.g. only an LFP series in processing/ecephys
        f.visititems(
            lambda _, obj: (
                candidates.append(obj)
                if _str(obj.attrs.get("neurodata_type", "")) == "ElectricalSeries"
                else None
            )
        )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one voltage series, found {len(candidates)}: "
            f"{[c.name for c in candidates]}; choose one with series="
        )
    return candidates[0]


def _read_annotations(f, series, t_start, meas_date):
    """Read each NWB interval table that applies to the series as annotations."""
    onset, duration, description = [], [], []
    for name, table in f.get("intervals", {}).items():
        if "start_time" not in table or "stop_time" not in table:
            continue
        start, stop = table["start_time"][()], table["stop_time"][()]
        keep = stop >= start  # also drops NaN
        if "timeseries_index" in table:  # rows may be scoped to some series
            refs = table["timeseries"]["timeseries"]
            bounds = np.concatenate([[0], table["timeseries_index"][()]])
            # one reference lookup per row, fine for typical interval tables
            keep &= [
                lo == hi or any(f[ref] == series for ref in refs[lo:hi])
                for lo, hi in zip(bounds[:-1], bounds[1:])
            ]
        label = "BAD_nwb_invalid" if name == "invalid_times" else f"NWB/{name}"
        onset.extend(start[keep] - t_start)
        duration.extend(stop[keep] - start[keep])
        description.extend([label] * int(keep.sum()))
    return Annotations(onset, duration, description, orig_time=meas_date)


def _get_info(fname, ch_type, series):
    """Read NWB header information without loading the voltage samples."""
    h5py = _import_h5py()
    with h5py.File(fname, "r") as f:
        series = _find_series(f, series)
        signal = series["data"]
        if signal.ndim not in (1, 2) or signal.dtype.kind not in "fiu":
            raise ValueError("Expected 1-D or 2-D numeric voltage data")
        n_times = signal.shape[0]
        n_channels = 1 if signal.ndim == 1 else signal.shape[1]
        unit = _str(signal.attrs.get("unit", "")).strip().lower()
        _check_option("NWB voltage unit", unit, tuple(_VOLTS))
        unit_scale = _VOLTS[unit]
        scales = np.full(n_channels, signal.attrs.get("conversion", 1.0) * unit_scale)
        if "channel_conversion" in series:
            per_channel = series["channel_conversion"][()]
            if per_channel.shape != (n_channels,):
                raise ValueError("Expected one channel_conversion per channel")
            scales *= per_channel
        offset = float(signal.attrs.get("offset", 0.0)) * unit_scale
        names = [str(ii) for ii in range(n_channels)]
        if "electrodes" in series:
            rows = series["electrodes"][()]
            if rows.shape != (n_channels,):
                raise ValueError("Electrode indices do not match the data channels")
            ids = f[series["electrodes"].attrs["table"]]["id"][()]
            names = [str(ids[row]) for row in rows]
        sfreq, t_start = _read_nwb_time(series, n_times)
        info = create_info(names, sfreq, ch_type)
        ref = f.get("timestamps_reference_time", f.get("session_start_time"))
        date = None if ref is None else datetime.fromisoformat(_str(ref[()]))
        if date is not None and date.tzinfo is None:
            warn("NWB measurement date has no timezone; leaving it unset.")
            date = None
        if date is not None:
            date += timedelta(seconds=t_start)
        info.set_meas_date(date)
        subject = f.get("general/subject", {})
        subject_info = {}
        if "subject_id" in subject:
            subject_info["his_id"] = _str(subject["subject_id"][()])
        if "sex" in subject:  # NWB uses M, F, U or O
            sex = _str(subject["sex"][()]).strip().upper()[:1]
            subject_info["sex"] = {"M": 1, "F": 2}.get(sex, 0)
        if subject_info:
            info["subject_info"] = subject_info
        annotations = _read_annotations(f, series, t_start, info["meas_date"])
        nwb_info = dict(
            filename=signal.file.filename,
            n_samples=n_times,
            path=signal.name,
            scales=scales,
            offset=offset,
            ndim=signal.ndim,
        )
    orig_units = dict.fromkeys(names, {1.0: "V", 1e-3: "mV", 1e-6: "µV"}[unit_scale])
    return info, nwb_info, orig_units, annotations


class RawNWB(BaseRaw):
    """Raw object for a single NWB voltage series."""

    @_verbose_control
    def __init__(
        self, fname, *, ch_type="seeg", series=None, preload=False, verbose=None
    ):
        fname = _check_fname(fname, "read", True, "fname")
        logger.info(f"Extracting NWB parameters from {fname}...")
        info, nwb_info, orig_units, annotations = _get_info(fname, ch_type, series)
        super().__init__(
            info,
            preload=preload,
            last_samps=[nwb_info["n_samples"] - 1],
            filenames=[nwb_info["filename"]],
            raw_extras=[nwb_info],
            orig_units=orig_units,
            orig_format="double",
            verbose=verbose,
        )
        self.set_annotations(annotations, emit_warning=False)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        h5py = _import_h5py()
        extras = self._raw_extras[fi]
        with h5py.File(self.filenames[fi], "r") as f:
            signal = f[extras["path"]]
            if extras["ndim"] == 1:
                block = signal[start:stop][None, :]
            elif isinstance(idx, slice):
                block = signal[start:stop, idx].T
            else:  # h5py needs increasing indices
                columns, inverse = np.unique(idx, return_inverse=True)
                block = signal[start:stop, columns].T[inverse]
        scales = extras["scales"][idx]
        if mult is None:
            _mult_cal_one(data, block, slice(None), cals * scales[:, None], None)
            if extras["offset"]:
                data += extras["offset"] * cals
        else:
            _mult_cal_one(data, block, slice(None), None, mult * scales[None, :])
            if extras["offset"]:
                # A voltage offset must pass through the projector as well.
                data += extras["offset"] * mult.sum(axis=1, keepdims=True)
