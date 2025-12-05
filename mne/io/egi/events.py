#
# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from datetime import datetime
from glob import glob
from os.path import basename, join, splitext

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
    mff_events, event_codes = _read_mff_events(input_fname, info["sfreq"])
    info["n_events"] = len(event_codes)
    info["event_codes"] = event_codes
    events = np.zeros([info["n_events"], info["n_segments"] * n_samples])
    for n, event in enumerate(event_codes):
        for i in mff_events[event]:
            if (i < 0) or (i >= events.shape[1]):
                continue
            events[n][i] = n + 1
    return events, info, mff_events


def _read_mff_events(filename, sfreq):
    """Extract the events.

    Parameters
    ----------
    filename : path-like
        File path.
    sfreq : float
        The sampling frequency
    """
    orig = {}
    for xml_file in glob(join(filename, "*.xml")):
        xml_type = splitext(basename(xml_file))[0]
        et = _parse_xml(xml_file)
        if et is not None:
            orig[xml_type] = et
    xml_files = orig.keys()
    xml_events = [x for x in xml_files if x[:7] == "Events_"]
    for item in orig["info"]:
        if "recordTime" in item:
            start_time = _ns2py_time(item["recordTime"])
            break
    markers = []
    code = []
    for xml in xml_events:
        for event in orig[xml][2:]:
            event_start = _ns2py_time(event["beginTime"])
            start = (event_start - start_time).total_seconds()
            if event["code"] not in code:
                code.append(event["code"])
            marker = {
                "name": event["code"],
                "start": start,
                "start_sample": int(np.fix(start * sfreq)),
                "end": start + float(event["duration"]) / 1e9,
                "chan": None,
            }
            markers.append(marker)
    events_tims = dict()
    for ev in code:
        trig_samp = list(
            c["start_sample"] for n, c in enumerate(markers) if c["name"] == ev
        )
        events_tims.update({ev: trig_samp})
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
