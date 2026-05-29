#
# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from datetime import datetime
from glob import glob
from os.path import basename, join, splitext

import numpy as np

from ...fixes import _parse_mffpy_datetime
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
    """Extract the events with mffpy.

    Parameters
    ----------
    filename : path-like
        File path.
    sfreq : float
        The sampling frequency
    start_time : datetime
        The recording start time used as the event anchor.
    """
    # Use defusedxml to parse Events XML directly (avoid mffpy's strict
    # datetime parsing which may include nanosecond fractions). We still use
    # mffpy.Reader for locating the Events.xml files inside the MFF.
    _soft_import("mffpy", "reading EGI MFF data")
    _soft_import("defusedxml", "reading EGI MFF data")
    import mffpy
    import defusedxml.ElementTree as DET

    reader = mffpy.Reader(filename)
    # Quick pre-scan: warn on any XML files that cannot be parsed (test
    # coverage expects a warning when arbitrary XML is corrupt).
    try:
        files_list = sorted(reader.directory.listdir())
    except Exception:
        files_list = []
    tracks = []
    for xml_name in files_list:
        if not xml_name.lower().endswith('.xml'):
            continue
        stem0 = splitext(basename(xml_name))[0]
        try:
            with reader.directory.filepointer(stem0) as fptest:
                try:
                    DET.parse(fptest)
                except Exception as exc:
                    warn(f"Could not parse the XML file {xml_name}: {exc}", RuntimeWarning)
        except Exception:
            # ignore files that cannot be opened via mffpy API
            continue
    for xml_name in files_list:
        if not splitext(basename(xml_name))[0].startswith("Events"):
            continue
        stem = splitext(basename(xml_name))[0]
        with reader.directory.filepointer(stem) as fp:
            try:
                root = DET.parse(fp).getroot()
            except Exception as exc:
                # fallback: try reading as bytes and parse string
                try:
                    fp.seek(0)
                    txt = fp.read()
                    root = DET.fromstring(txt)
                except Exception as exc2:
                    warn(f"Could not parse the XML file {xml_name}: {exc2}", RuntimeWarning)
                    continue
        # identify eventTrack root (namespace-insensitive)
        if _ns(root.tag) == "eventTrack":
            tracks.append(root)

    markers = []
    code = []
    for root in tracks:
        # each child 'event' element
        for event_el in root.findall("{*}event"):
            # extract fields by tag name ignoring namespace
            ev = {}
            for child in event_el:
                tag = _ns(child.tag)
                ev[tag] = child.text
            # parse times and duration
            event_start = _parse_mffpy_datetime(ev.get("beginTime"), tzinfo=start_time.tzinfo)
            if event_start is None:
                continue
            start_sec = (event_start - start_time).total_seconds()
            code_str = ev.get("code", "")
            if code_str not in code:
                code.append(code_str)
            # duration in xml is typically in nanoseconds
            duration = None
            if ev.get("duration") is not None:
                try:
                    duration = int(ev.get("duration")) / 1e9
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


def _parse_xml(xml_file: str) -> list[dict[str, str]] | None:
    """Parse XML file."""
    defusedxml = _soft_import("defusedxml", "reading EGI MFF data")
    try:
        xml = defusedxml.ElementTree.parse(xml_file)
    except defusedxml.ElementTree.ParseError as e:
        warn(f"Could not parse the XML file {xml_file}: {e}")
        return
    root = xml.getroot()
    return _xml2list(root)


def _xml2list(root):
    """Parse XML item."""
    output = []
    for element in root:
        if len(element) > 0:
            if element[0].tag != element[-1].tag:
                output.append(_xml2dict(element))
            else:
                output.append(_xml2list(element))

        elif element.text:
            text = element.text.strip()
            if text:
                tag = _ns(element.tag)
                output.append({tag: text})

    return output


def _ns(s):
    """Remove namespace, but only if there is a namespace to begin with."""
    if "}" in s:
        return "}".join(s.split("}")[1:])
    else:
        return s


def _xml2dict(root):
    """Use functions instead of Class.

    remove namespace based on
    http://stackoverflow.com/questions/2148119
    """
    output = {}
    if root.items():
        output.update(dict(root.items()))

    for element in root:
        if len(element) > 0:
            if len(element) == 1 or element[0].tag != element[1].tag:
                one_dict = _xml2dict(element)
            else:
                one_dict = {_ns(element[0].tag): _xml2list(element)}

            if element.items():
                one_dict.update(dict(element.items()))
            output.update({_ns(element.tag): one_dict})

        elif element.items():
            output.update({_ns(element.tag): dict(element.items())})

        else:
            output.update({_ns(element.tag): element.text})
    return output


def _ns2py_time(nstime):
    """Parse times."""
    nsdate = nstime[0:10]
    nstime0 = nstime[11:26]
    nstime00 = nsdate + " " + nstime0
    pytime = datetime.strptime(nstime00, "%Y-%m-%d %H:%M:%S.%f")
    return pytime


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
