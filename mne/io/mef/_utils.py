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
    """Safely extract and convert a value from MEF metadata.

    Parameters
    ----------
    md : dict | object
        The metadata dictionary or object.
    key : str
        The key to look up.
    kind : str | None
        The expected type ('text', 'int', 'float').
    default : any
        The default value if the key is missing or conversion fails.

    Returns
    -------
    value : any
        The extracted and converted value.
    """
    if md is None:
        return default
    try:
        value = md.get(key) if isinstance(md, dict) else md[key]
    except (KeyError, TypeError, AttributeError):
        return default
    if isinstance(value, np.ndarray) and value.size > 0:
        value = value.ravel()[0]
    if value is None:
        return default
    if kind == "text":
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8", errors="ignore")
        return str(value).strip().strip("\x00")
    if kind in ("int", "float"):
        try:
            return int(value) if kind == "int" else float(value)
        except (TypeError, ValueError):
            return default
    return value


def _get_mef_units_scale(unit_desc, unit_factor):
    """Normalize unit description and calculate the scaling factor to Volts.

    Parameters
    ----------
    unit_desc : str
        The unit description (e.g., 'uV', 'mV').
    unit_factor : float
        The multiplier for the units.

    Returns
    -------
    scale : float
        The total scaling factor to convert data to Volts.
    unit_desc : str
        The original unit description.
    unit_desc_norm : str
        The normalized unit description.
    unit_factor_valid : bool
        True if the unit factor is valid (finite and > 0).
    unit_known : bool
        True if the unit description is recognized.
    """
    if isinstance(unit_desc, np.ndarray):
        unit_desc = unit_desc.ravel()[0]
    if unit_desc is None:
        unit_desc = ""
    if isinstance(unit_desc, (bytes, np.bytes_)):
        unit_desc = unit_desc.decode("utf-8", errors="ignore")
    unit_desc = str(unit_desc).strip().strip("\x00")
    if unit_desc:
        # Normalize unit string for lookup
        unit_desc_norm = unit_desc.lower().replace(" ", "")
        # Replace common micro symbols with 'u'
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
    """Extract session-level time metadata (offsets, DST info)."""
    extras = {}
    recording_time_offset = _mef_get(section_3, "recording_time_offset", kind="int")
    if recording_time_offset not in (None, _UUTC_NO_ENTRY):
        extras["recording_time_offset"] = int(recording_time_offset)
    dst_start_time = _mef_get(section_3, "DST_start_time", kind="int")
    if dst_start_time not in (None, _UUTC_NO_ENTRY):
        extras["dst_start_time"] = int(dst_start_time)
    dst_end_time = _mef_get(section_3, "DST_end_time", kind="int")
    if dst_end_time not in (None, _UUTC_NO_ENTRY):
        extras["dst_end_time"] = int(dst_end_time)
    return extras


def _convert_record(record, session):
    """Ensure a MEF record is in dictionary format.

    If it's already a dict with a 'type', return it. Otherwise, try to convert it
    using the session's internal conversion method.
    """
    if isinstance(record, dict) and "type" in record:
        return record
    if isinstance(record, dict) and "record_header" in record:
        try:
            return session._create_dict_record(record)
        except Exception:
            # We don't know what this might raise, so we catch all
            return None
    return None


def _iter_records(session, ts_channels):
    """Iterate over all records in a MEF session (session, channel, and segment levels).

    Yields
    ------
    record : dict
        The record metadata.
    ch_name : str | None
        The channel name if the record is at the channel or segment level.
    seg_name : str | None
        The segment name if the record is at the segment level.
    """
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


def _records_to_annotations(session, ts_channels, start_uutc):
    """Convert MEF records to MNE annotation components.

    Parameters
    ----------
    session : MefSession
        The MEF session object.
    ts_channels : dict
        The time series channels metadata.
    start_uutc : int
        The earliest start time of the recording in UUTC microseconds.

    Returns
    -------
    onsets : list of float
        Annotation onsets in seconds.
    durations : list of float
        Annotation durations in seconds.
    descriptions : list of str
        Annotation descriptions.
    ch_names : list of list of str
        Channel names (if any) associated with each annotation.
    extras_list : list of dict
        Additional metadata for each annotation.
    """
    if start_uutc is None:
        return [], [], [], [], []

    onsets = []
    durations = []
    descriptions = []
    ch_names = []
    extras_list = []
    for record, channel, segment in _iter_records(session, ts_channels):
        record = _convert_record(record, session)
        if record is None:
            continue

        # Extract record type and detailed description
        record_type = _mef_get(record, "type", kind="text", default="")
        detail = ""
        for key in ("text", "annotation", "epoch_type", "name"):
            detail = _mef_get(record, key, kind="text", default="")
            if detail:
                break
        description = record_type or "MEF"
        if detail:
            description = f"{description}: {detail}"

        # Resolve onset timestamp
        onset_uutc = (
            _mef_get(record, "time")
            or _mef_get(record, "timestamp")
            or _mef_get(record, "earliest_onset")
        )
        if onset_uutc is None or onset_uutc == _UUTC_NO_ENTRY:
            continue

        # Resolve duration
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

        # Convert to seconds relative to session start
        try:
            onset = (float(onset_uutc) - float(start_uutc)) / 1e6
            duration = max(float(duration_uutc), 0.0) / 1e6
        except (TypeError, ValueError):
            continue

        onsets.append(onset)
        durations.append(duration)
        descriptions.append(description)
        ch_names.append([channel] if channel else [])
        extras_list.append(
            {
                k: v
                for k, v in [
                    ("detail", detail),
                    ("channel", channel),
                    ("segment", segment),
                ]
                if v
            }
        )
    return onsets, durations, descriptions, ch_names, extras_list


def _toc_to_gap_annotations(toc, sfreq):
    """Identify data gaps from the Table of Contents (TOC).

    The TOC stores the start sample and number of samples for each continuous block.
    Gaps between these blocks are converted to 'BAD_ACQ_SKIP' annotations.

    Parameters
    ----------
    toc : np.ndarray
        The TOC array from pymef.
    sfreq : float
        The sampling frequency.

    Returns
    -------
    onsets : list of float
        Onset of each gap in seconds.
    durations : list of float
        Duration of each gap in seconds.
    """
    if toc is None or toc.size == 0:
        return [], []

    # The Table of Contents (TOC) array contains multiple columns.
    # We extract the start samples and the number of samples for each continuous block.
    start_samples = toc[2]  # Column 2: Start sample index of the block
    n_samples = toc[1]  # Column 1: Number of samples in the block

    # Ensure blocks are processed in chronological order based on start samples
    order = np.argsort(start_samples)
    start_samples = start_samples[order]
    n_samples = n_samples[order]

    onsets = []
    durations = []

    # Initialize trackers with the first block
    prev_start = start_samples[0]
    prev_n = n_samples[0]

    # Compare each block with the previous one to find discontinuities (gaps)
    for start, n_samp in zip(start_samples[1:], n_samples[1:]):
        # The next sample we expect to see if there were no gap
        expected = prev_start + prev_n

        # Calculate the gap size in samples
        gap = start - expected
        if gap > 0:
            # If a gap exists, store the onset (where the expected sample was)
            # and the duration, both converted to seconds using the sampling frequency.
            onsets.append(expected / sfreq)
            durations.append(gap / sfreq)

        # Update trackers for the next iteration
        prev_start = start
        prev_n = n_samp
    return onsets, durations
