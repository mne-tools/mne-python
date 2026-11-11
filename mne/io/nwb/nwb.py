"""Read a single NWB voltage series using h5py."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# Adapted from eegdash/io/nwb.py (https://github.com/eegdash/EEGDash), BSD-3-Clause.

import re
from datetime import datetime, timedelta

import numpy as np

from mne._fiff.meas_info import create_info
from mne._fiff.utils import _mult_cal_one
from mne.annotations import Annotations
from mne.io.base import BaseRaw
from mne.utils import (
    _check_dict_keys,
    _check_fname,
    _check_option,
    _import_h5py,
    _validate_type,
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
def read_raw_nwb(fname, *, overrides=None, preload=False, verbose=None):
    """Read an NWB voltage series using h5py.

    Parameters
    ----------
    fname : path-like
        Path to an NWB file.
    overrides : dict | None
        Corrections applied in memory, without changing the source file.
        Supported keys are ``sfreq`` (finite positive frequency in Hz),
        ``meas_date`` (timezone-aware first-sample datetime, or None),
        ``ch_names`` (unique names matching the channel count), and ``ch_types``
        (one MNE channel type or a list matching the channel count).
        An explicit ``sfreq`` permits a uniform approximation of nonuniform
        timestamps, with a warning reporting the maximum timing discrepancy.
        It does not resample or reorder samples, or retime source intervals.
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
    Requires :mod:`h5py`. Reads the single acquisition ``ElectricalSeries``,
    or the single two-dimensional voltage ``TimeSeries`` if none exists.
    Channel names default to electrode IDs or column numbers. Channel types
    default to sEEG; this is not inferred from anatomical metadata.
    Use ``overrides={"ch_types": "ecog"}`` to select another type at read time,
    or :meth:`~mne.io.Raw.set_channel_types` to change types afterward.

    Nonuniform timestamps require an explicit sampling-frequency override.
    Without an override, displacement from the uniform grid must be within
    0.1 percent of one sample interval, allowing numerical rounding.
    Dates without a timezone stay unset unless overridden.

    Subject, device, experimenter and date metadata use existing Info fields.
    Other descriptive metadata, including electrode columns and software
    provenance, is stored as labeled text in ``info['description']``.
    Invalid intervals become ``BAD_nwb_invalid`` annotations, and trials
    become ``NWB/trial`` annotations with scalar columns in their extras.
    Coordinates are described without guessing their MNE reference frame.
    Original unit labels are available while reading NWB but are not retained
    by a FIF round-trip. The source NWB remains authoritative for other
    streams, references and arbitrary arrays.

    .. versionadded:: 1.14
    """
    return RawNWB(fname, overrides=overrides, preload=preload, verbose=verbose)


def _read_nwb_time(series, n_times, user_sfreq):
    """Read the sample rate and first-sample time in seconds."""
    timestamps = None
    file_sfreq = np.nan
    if "timestamps" in series:
        timestamps = series["timestamps"][()]
        if timestamps.shape != (n_times,) or not np.isfinite(timestamps).all():
            raise ValueError("Expected finite timestamps matching the data samples")
        t_start = float(timestamps[0])
        if user_sfreq is None:
            if n_times < 2 or timestamps[-1] <= t_start:
                raise ValueError(
                    "NWB timestamps cannot define a positive sampling rate; "
                    "supply an sfreq override"
                )
            file_sfreq = (n_times - 1) / (float(timestamps[-1]) - t_start)
    else:
        clock = series["starting_time"]
        t_start = float(clock[()])
        file_sfreq = float(clock.attrs["rate"])
    sfreq = float(file_sfreq if user_sfreq is None else user_sfreq)
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Expected a finite positive NWB sampling rate")
    if not np.isfinite(t_start):
        raise ValueError("Expected a finite NWB starting time")
    if timestamps is None:
        if user_sfreq is not None and not np.isclose(sfreq, file_sfreq):
            warn(f"Overriding NWB sampling frequency {file_sfreq} Hz with {sfreq} Hz.")
        return sfreq, t_start
    relative_times = timestamps - t_start
    max_error = np.max(np.abs(relative_times - np.arange(n_times) / sfreq))
    roundoff = 8 * np.spacing(max(1.0, np.abs(timestamps).max()))
    tolerance = max(1e-3 / sfreq, roundoff)
    if user_sfreq is None and (
        np.any(timestamps[1:] <= timestamps[:-1]) or max_error > tolerance
    ):
        raise ValueError("NWB has nonuniform timestamps; supply an sfreq override")
    if user_sfreq is not None and max_error > tolerance:
        warn(f"Approximating NWB timestamps; maximum discrepancy is {max_error:g} s.")
    return sfreq, t_start


def _read_h5_value(dataset):
    """Read a metadata value using h5py's string and reference handling."""
    h5py = _import_h5py()
    value = (
        dataset.asstr()[()] if h5py.check_string_dtype(dataset.dtype) else dataset[()]
    )
    if isinstance(value, np.generic):
        value = value.item()
    return dataset.file[value].name if isinstance(value, h5py.Reference) else value


def _read_nwb_table(table, rows):
    """Iterate over NWB table rows, resolving references and indexed columns."""
    h5py = _import_h5py()
    columns = {
        name: _read_h5_value(table[name]) for name in ("id", *table.attrs["colnames"])
    }
    references = {name for name in columns if h5py.check_ref_dtype(table[name].dtype)}
    n_rows = len(columns["id"])
    bounds = {}
    for name, column in columns.items():
        message = f"Invalid NWB table column {table.name}/{name}"
        if name + "_index" not in table:
            if np.shape(column)[:1] != (n_rows,):
                raise ValueError(message)
            continue
        stops = table[name + "_index"][()]
        if stops.shape != (n_rows,) or stops.dtype.kind not in "iu":
            raise ValueError(message)
        bounds[name] = np.concatenate((np.zeros(1, dtype=stops.dtype), stops))
        if np.any(bounds[name][1:] < bounds[name][:-1]) or bounds[name][-1] != len(
            column
        ):
            raise ValueError(message)
    for row in rows:
        record = {}
        for name, column in columns.items():
            index = row
            if name in bounds:
                index = slice(bounds[name][row], bounds[name][row + 1])
            value = column[index]
            if isinstance(value, np.generic):
                value = value.item()
            if name in references and not isinstance(value, np.ndarray):
                value = table.file[value].name
            record[name] = value
        yield record


def _read_nwb_metadata(fid, series, electrodes, info):
    """Populate native Info fields and return descriptive metadata."""
    h5py = _import_h5py()
    metadata = {}
    for row in electrodes:
        fields = [f"{name}={value}" for name, value in row.items() if name != "id"]
        metadata[f"electrode {row['id']}"] = ", ".join(fields)
    objects = [fid, series, series["data"]]
    for path in ("/", "general", "general/subject"):
        for dataset in fid.get(path, {}).values():
            if isinstance(dataset, h5py.Dataset):
                metadata[dataset.name] = _read_h5_value(dataset)
    subject = {k: _read_h5_value(v) for k, v in fid.get("general/subject", {}).items()}
    subject_info = dict(
        his_id=subject.get("subject_id"),
        sex={"M": 1, "F": 2, "U": 0, "O": 0}.get(subject.get("sex")),
    )
    if "date_of_birth" in subject:
        try:
            subject_info["birthday"] = datetime.fromisoformat(
                subject["date_of_birth"]
            ).date()
        except ValueError:
            pass
    if "weight" in subject:
        match = re.fullmatch(r"(.+?)\s*(kg|mg|g)", subject["weight"].strip())
        if match is not None:
            try:
                weight = float(match[1]) * {"kg": 1.0, "g": 0.001, "mg": 1e-6}[match[2]]
            except ValueError:
                weight = np.nan
            if np.isfinite(weight) and weight >= 0:
                subject_info["weight"] = weight
    subject_info = {
        key: value for key, value in subject_info.items() if value is not None
    }
    if subject_info:
        info["subject_info"] = subject_info
    if "general/experimenter" in fid:
        info["experimenter"] = "; ".join(
            np.atleast_1d(_read_h5_value(fid["general/experimenter"]))
        )
    for idx, row in enumerate(electrodes):
        if row.get("bad") is True or row.get("status") == "bad":
            info["bads"].append(info["ch_names"][idx])
    groups = {row["group"] for row in electrodes if "group" in row}
    devices = {fid[group]["device"] for group in groups}
    if len(devices) == 1:
        device = next(iter(devices))
        device = device.file[device.ref]
        objects.append(device)
        device_info = {}
        if "model" in device:
            model = fid[device["model"].ref]
            device_info["model"] = model.name.rsplit("/", 1)[-1]
            objects.append(model)
        if "serial_number" in device.attrs:
            device_info["serial"] = device.attrs["serial_number"]
        if device_info:
            info["device_info"] = device_info
    for obj in objects:
        for name, value in obj.attrs.items():
            if name not in ("namespace", "neurodata_type", "object_id"):
                metadata[f"{obj.name.rstrip('/')}/{name}"] = value
    return metadata


def _read_nwb_annotations(fid, series, t_start, meas_date):
    """Read invalid intervals and trials for the selected series."""
    onset, duration, description, extras = [], [], [], []
    metadata = {}
    for name, label in (("invalid_times", "BAD_nwb_invalid"), ("trials", "NWB/trial")):
        table = fid.get("intervals/" + name)
        if table is None:
            continue
        for row in _read_nwb_table(table, range(len(table["id"]))):
            references = row.pop("timeseries", [])
            if len(references) and not any(
                fid[ref["timeseries"]] == series for ref in references
            ):
                continue
            start, stop = row.pop("start_time"), row.pop("stop_time")
            if not np.isfinite([start, stop]).all() or stop < start:
                raise ValueError(f"Invalid interval in NWB {name}")
            onset.append(start - t_start)
            duration.append(stop - start)
            description.append(label)
            extra = {}
            for key, value in row.items():
                if isinstance(value, (str, int, float)) or value is None:
                    extra[f"nwb_{key}"] = value
                else:
                    metadata[f"{name}/{row['id']}/{key}"] = value
            extras.append(extra)
    annotations = Annotations(
        onset, duration, description, orig_time=meas_date, extras=extras
    )
    return annotations, metadata


def _get_info(fname, overrides):
    """Read NWB header information without loading the voltage samples."""
    h5py = _import_h5py()
    _validate_type(overrides, (dict, None), "overrides")
    overrides = {} if overrides is None else overrides.copy()
    _check_dict_keys(
        overrides,
        ("sfreq", "meas_date", "ch_names", "ch_types"),
        "override key(s)",
        "valid override keys",
    )
    if "sfreq" in overrides:
        _validate_type(overrides["sfreq"], "numeric", "sfreq")
    with h5py.File(fname, "r") as f:
        by_type = {}
        for group in f.get("acquisition", {}).values():
            if not isinstance(group, h5py.Group) or "data" not in group:
                continue
            kind = group.attrs.get("neurodata_type")
            unit = group["data"].attrs.get("unit", "")
            unit = unit.decode() if isinstance(unit, bytes) else str(unit)
            if kind == "ElectricalSeries" or (
                kind == "TimeSeries"
                and group["data"].ndim == 2
                and unit.strip().lower() in _VOLTS
            ):
                by_type.setdefault(kind, []).append(group)
        candidates = by_type.get("ElectricalSeries", by_type.get("TimeSeries", []))
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one acquisition voltage series, found {len(candidates)}"
            )
        series = candidates[0]
        signal = series["data"]
        if (
            signal.ndim not in (1, 2)
            or not all(signal.shape)
            or signal.dtype.kind not in "fiu"
        ):
            raise ValueError("Expected nonempty 1-D or 2-D numeric voltage data")
        n_times = signal.shape[0]
        n_channels = 1 if signal.ndim == 1 else signal.shape[1]
        unit = signal.attrs.get("unit", "")
        unit = unit.decode() if isinstance(unit, bytes) else str(unit)
        unit = unit.strip().lower()
        _check_option("NWB voltage unit", unit, tuple(_VOLTS))
        unit_scale = _VOLTS[unit]
        scales = np.full(
            n_channels, float(signal.attrs.get("conversion", 1.0)) * unit_scale
        )
        if "channel_conversion" in series:
            per_channel = series["channel_conversion"][()]
            if per_channel.shape != (n_channels,):
                raise ValueError("Expected one channel_conversion per channel")
            scales *= per_channel
        offset = float(signal.attrs.get("offset", 0.0)) * unit_scale
        if not np.isfinite(scales).all() or not np.isfinite(offset):
            raise ValueError(
                "Expected finite conversion, channel_conversion and offset"
            )
        electrodes = []
        ids = np.arange(n_channels)
        if "electrodes" in series:
            region = series["electrodes"]
            rows = region[()]
            table = f[region.attrs["table"]]
            if (
                rows.shape != (n_channels,)
                or rows.dtype.kind not in "iu"
                or np.any(rows < 0)
                or np.any(rows >= len(table["id"]))
            ):
                raise ValueError("Electrode indices do not match the data channels")
            electrodes = list(_read_nwb_table(table, rows))
            ids = [row["id"] for row in electrodes]
        sfreq, t_start = _read_nwb_time(series, n_times, overrides.get("sfreq"))
        names = overrides.get("ch_names", [str(idx) for idx in ids])
        _validate_type(names, (list, tuple), "ch_names")
        if len(names) != n_channels:
            raise ValueError("ch_names must match the channel count")
        if len(set(names)) != n_channels:
            raise ValueError("Channel names must be unique")
        ch_types = overrides.get("ch_types", "seeg")
        info = create_info(names, sfreq, ch_types)
        if "meas_date" in overrides:
            date = overrides["meas_date"]
            _validate_type(date, (datetime, None), "meas_date")
        else:
            ref = f.get("timestamps_reference_time", f.get("session_start_time"))
            try:
                date = (
                    datetime.fromisoformat(_read_h5_value(ref))
                    if ref is not None
                    else None
                )
            except (TypeError, ValueError):
                date = None
            if date is None or date.tzinfo is None:
                warn("NWB measurement date has no valid timezone; leaving date unset.")
                date = None
            else:
                date += timedelta(seconds=t_start)
        info.set_meas_date(date)
        metadata = _read_nwb_metadata(f, series, electrodes, info)
        annotations, interval_metadata = _read_nwb_annotations(
            f, series, t_start, info["meas_date"]
        )
        metadata.update(interval_metadata)
        info["description"] = "\n".join(
            f"NWB {key}: {value}" for key, value in metadata.items()
        )
        nwb_info = dict(
            filename=signal.file.filename,
            n_samples=n_times,
            path=signal.name,
            scales=scales,
            offset=offset,
            ndim=signal.ndim,
        )
        canonical_unit = {1.0: "V", 1e-3: "mV", 1e-6: "µV"}[unit_scale]
        orig_units = dict.fromkeys(names, canonical_unit)
    return info, nwb_info, orig_units, annotations


class RawNWB(BaseRaw):
    """Raw object for a single NWB voltage series."""

    @_verbose_control
    def __init__(self, fname, *, overrides=None, preload=False, verbose=None):
        fname = _check_fname(fname, "read", True, "fname")
        logger.info(f"Extracting NWB parameters from {fname}...")
        info, nwb_info, orig_units, annotations = _get_info(fname, overrides)
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
        self.set_annotations(annotations)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        h5py = _import_h5py()
        extras = self._raw_extras[fi]
        with h5py.File(self.filenames[fi], "r") as f:
            signal = f[extras["path"]]
            if extras["ndim"] == 1:
                block = signal[start:stop][None, :]
            elif isinstance(idx, slice):
                block = signal[start:stop, idx].T
            else:
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
