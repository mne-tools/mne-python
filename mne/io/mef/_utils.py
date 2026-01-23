# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

_UNIT_SCALE_FACTORS = {
    "v": 1.0,
    "volt": 1.0,
    "volts": 1.0,
    "uv": 1e-6,
    "uvolt": 1e-6,
    "uvolts": 1e-6,
    "microv": 1e-6,
    "microvolt": 1e-6,
    "microvolts": 1e-6,
    "mv": 1e-3,
    "mvolt": 1e-3,
    "millivolt": 1e-3,
    "millivolts": 1e-3,
    "nv": 1e-9,
    "nvolt": 1e-9,
    "nanovolt": 1e-9,
    "nanovolts": 1e-9,
    "kv": 1e3,
    "kilovolt": 1e3,
    "kilovolts": 1e3,
}

_UUTC_NO_ENTRY = 0x800000000000
_GMT_OFFSET_NO_ENTRY = -86401


def _mef_get(md, key, *, kind=None, default=None):
    if md is None:
        return default
    try:
        value = md.get(key) if isinstance(md, dict) else md[key]
    except Exception:
        return default
    if isinstance(value, np.ndarray):
        value = value.ravel()[0]
    if value is None:
        return default
    if kind == "text":
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8", errors="ignore")
        return str(value).strip().strip("\x00")
    if kind == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if kind == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return value


def _get_mef_units_scale(unit_desc, unit_factor):
    if isinstance(unit_desc, np.ndarray):
        unit_desc = unit_desc.ravel()[0]
    if unit_desc is None:
        unit_desc = ""
    if isinstance(unit_desc, (bytes, np.bytes_)):
        unit_desc = unit_desc.decode("utf-8", errors="ignore")
    unit_desc = str(unit_desc).strip().strip("\x00")
    if unit_desc:
        unit_desc_norm = unit_desc.lower().replace(" ", "")
        unit_desc_norm = unit_desc_norm.replace("\N{MICRO SIGN}", "u")
        unit_desc_norm = unit_desc_norm.replace("\N{GREEK SMALL LETTER MU}", "u")
    else:
        unit_desc_norm = ""

    if isinstance(unit_factor, np.ndarray):
        unit_factor = unit_factor.ravel()[0]
    try:
        unit_factor = float(unit_factor)
    except (TypeError, ValueError):
        unit_factor = np.nan
    unit_factor_valid = np.isfinite(unit_factor) and unit_factor > 0
    if not unit_factor_valid:
        unit_factor = 1.0
    unit_scale = _UNIT_SCALE_FACTORS.get(unit_desc_norm)
    scale = unit_factor * (unit_scale if unit_scale is not None else 1.0)
    return (
        scale,
        unit_desc,
        unit_desc_norm,
        unit_factor_valid,
        unit_scale is not None,
    )


def _mef_time_metadata_extras(section_3):
    extras = {}
    recording_time_offset = _mef_get(
        section_3, "recording_time_offset", kind="int"
    )
    if recording_time_offset not in (None, _UUTC_NO_ENTRY):
        extras["recording_time_offset"] = int(recording_time_offset)
    dst_start_time = _mef_get(section_3, "DST_start_time", kind="int")
    if dst_start_time not in (None, _UUTC_NO_ENTRY):
        extras["dst_start_time"] = int(dst_start_time)
    dst_end_time = _mef_get(section_3, "DST_end_time", kind="int")
    if dst_end_time not in (None, _UUTC_NO_ENTRY):
        extras["dst_end_time"] = int(dst_end_time)
    return extras


def _records_to_annotations(session, ts_channels, start_uutc):
    if start_uutc is None:
        return [], [], [], [], []

    def _convert_record(record):
        if isinstance(record, dict) and "type" in record:
            return record
        if isinstance(record, dict) and "record_header" in record:
            try:
                return session._create_dict_record(record)
            except Exception:
                return None
        return None

    def _iter_records():
        session_records = session.session_md.get("records_info", {}).get("records", [])
        for record in session_records:
            yield record, None, None
        if isinstance(ts_channels, dict):
            for ch_name, ch_md in ts_channels.items():
                ch_records = ch_md.get("records_info", {}).get("records", [])
                for record in ch_records:
                    yield record, ch_name, None
                for seg_name, seg_md in ch_md.get("segments", {}).items():
                    seg_records = seg_md.get("records_info", {}).get("records", [])
                    for record in seg_records:
                        yield record, ch_name, seg_name

    onsets = []
    durations = []
    descriptions = []
    ch_names = []
    extras_list = []
    for record, channel, segment in _iter_records():
        record = _convert_record(record)
        if record is None:
            continue
        record_type = _mef_get(record, "type", kind="text", default="")
        detail = ""
        for key in ("text", "annotation", "epoch_type", "name"):
            detail = _mef_get(record, key, kind="text", default="")
            if detail:
                break
        description = record_type or "MEF"
        if detail:
            description = f"{description}: {detail}"

        onset_uutc = _mef_get(record, "time")
        if onset_uutc is None:
            onset_uutc = _mef_get(record, "timestamp")
        if onset_uutc is None and _mef_get(record, "earliest_onset") is not None:
            onset_uutc = _mef_get(record, "earliest_onset")
        if onset_uutc is None or onset_uutc == _UUTC_NO_ENTRY:
            continue

        duration_uutc = _mef_get(record, "duration")
        if duration_uutc is None:
            end_timestamp = _mef_get(record, "end_timestamp")
            timestamp = _mef_get(record, "timestamp")
            latest_offset = _mef_get(record, "latest_offset")
            earliest_onset = _mef_get(record, "earliest_onset")
            if end_timestamp is not None and timestamp is not None:
                duration_uutc = end_timestamp - timestamp
            elif latest_offset is not None and earliest_onset is not None:
                duration_uutc = latest_offset - earliest_onset
            else:
                duration_uutc = 0
        try:
            onset = (float(onset_uutc) - float(start_uutc)) / 1e6
            duration = max(float(duration_uutc), 0.0) / 1e6
        except (TypeError, ValueError):
            continue
        extras = {}
        if detail:
            extras["detail"] = detail
        if channel:
            extras["channel"] = channel
        if segment:
            extras["segment"] = segment
        onsets.append(onset)
        durations.append(duration)
        descriptions.append(description)
        ch_names.append([channel] if channel else [])
        extras_list.append(extras)
    return onsets, durations, descriptions, ch_names, extras_list


def _toc_to_gap_annotations(toc, sfreq):
    if toc is None or toc.size == 0:
        return [], []
    start_samples = toc[2]
    n_samples = toc[1]
    order = np.argsort(start_samples)
    start_samples = start_samples[order]
    n_samples = n_samples[order]
    onsets = []
    durations = []
    prev_start = start_samples[0]
    prev_n = n_samples[0]
    for start, n_samp in zip(start_samples[1:], n_samples[1:]):
        expected = prev_start + prev_n
        gap = start - expected
        if gap > 0:
            onsets.append(expected / sfreq)
            durations.append(gap / sfreq)
        prev_start = start
        prev_n = n_samp
    return onsets, durations
