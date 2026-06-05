#
# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from os.path import basename, splitext

import numpy as np

from ...utils import _soft_import, _validate_type, logger, warn


def _read_events(input_fname, info):
    """Read events for the record.

    Parameters
    ----------
    input_fname : path-like
        The file path.
    info : dict
        Header info array.
    """
    n_samples = info["last_samps"][-1]
    mff_events, event_codes = _read_mff_events(
        input_fname, info["sfreq"], info["meas_dt_local"]
    )
    info["n_events"] = len(event_codes)
    info["event_codes"] = event_codes
    events = np.zeros([info["n_events"], info["n_segments"] * n_samples])
    for n, event in enumerate(event_codes):
        for i in mff_events[event]:
            if (i < 0) or (i >= events.shape[1]):
                continue
            events[n][i] = n + 1
    return events, info, mff_events


def _read_mff_events(filename, sfreq, start_time):
    """Extract the events with mffpy."""
    # 1. Trigger MNE's helpful error message if the package is missing
    _soft_import("mffpy", "reading EGI MFF data")

    # 2. Now it is safe to do the actual imports locally
    import mffpy
    from mffpy.xml_files import XML

    reader = mffpy.Reader(filename)
    try:
        files_list = sorted(reader.directory.listdir())
    except Exception:
        files_list = []

    tracks = []
    for xml_name in files_list:
        stem = splitext(basename(xml_name))[0]
        # Only parse actual Event XML files to save I/O
        if not stem.startswith("Events"):
            continue

        with reader.directory.filepointer(stem) as fp:
            # Let mffpy 0.11 handle all the heavy lifting!
            track = XML.from_file(fp)
            tracks.append(track)

    markers = []
    code = []

    for track in tracks:
        for event in track.events:
            code_str = event.get("code")
            if code_str is None:
                continue

            if code_str not in code:
                code.append(code_str)

            # mffpy 0.11 properly parses beginTime into a datetime object
            event_start = event.get("beginTime")
            if event_start is None:
                continue

            if event_start.tzinfo is None and start_time.tzinfo is not None:
                event_start = event_start.replace(tzinfo=start_time.tzinfo)

            start_sec = (event_start - start_time).total_seconds()

            duration = event.get("duration")
            if duration is not None:
                try:
                    duration = int(duration) / 1e9
                except Exception:
                    duration = None

            markers.append(
                {
                    "name": code_str,
                    "start": start_sec,
                    "start_sample": int(np.trunc(start_sec * sfreq)),
                    "end": start_sec + (duration if duration is not None else 0.0),
                    "chan": None,
                }
            )

    events_tims = {
        ev: [marker["start_sample"] for marker in markers if marker["name"] == ev]
        for ev in code
    }
    return events_tims, code


def _combine_triggers(data, remapping=None):
    """Combine binary triggers."""
    new_trigger = np.zeros(data.shape[1])
    if data.astype(bool).sum(axis=0).max() > 1:  # ensure no overlaps
        logger.info(
            "    Found multiple events at the same time "
            "sample. Cannot create trigger channel."
        )
        return
    if remapping is None:
        remapping = np.arange(data) + 1
    for d, event_id in zip(data, remapping):
        idx = d.nonzero()
        if np.any(idx):
            new_trigger[idx] += event_id
    return new_trigger


def _triage_include_exclude(include, exclude, egi_events, egi_info):
    """Triage include and exclude."""
    _validate_type(exclude, (list, None), "exclude")
    _validate_type(include, (list, None), "include")
    event_codes = list(egi_info["event_codes"])
    for name, lst in dict(exclude=exclude, include=include).items():
        for ii, item in enumerate(lst or []):
            what = f"{name}[{ii}]"
            _validate_type(item, str, what)
            if item not in event_codes:
                raise ValueError(
                    f"Could not find event channel named {what}={repr(item)}"
                )
    if include is None:
        if exclude is None:
            default_exclude = ["sync", "TREV"]
            exclude = [code for code in default_exclude if code in event_codes]
            for code, event in zip(event_codes, egi_events):
                if event.sum() < 1 and code:
                    exclude.append(code)
            if (
                len(exclude) == len(event_codes)
                and egi_info["n_events"]
                and set(exclude) - set(default_exclude)
            ):
                warn(
                    "Did not find any event code with at least one event.",
                    RuntimeWarning,
                )
        include = [k for k in event_codes if k not in exclude]
    del exclude
    excl_events = ", ".join(k for k in event_codes if k not in include)
    logger.info(f"    Excluding events {{{excl_events}}} ...")
    return include
