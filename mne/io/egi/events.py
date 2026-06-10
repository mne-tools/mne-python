#
# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ...utils import _validate_type, logger, warn


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
