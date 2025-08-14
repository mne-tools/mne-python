# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
from pathlib import Path

import numpy as np

from mne.channels.channels import _unit2human
from mne.io.constants import FIFF
from mne.utils import _check_pybv_installed, warn

_check_pybv_installed()
from pybv import write_brainvision  # noqa: E402


def _export_mne_raw(*, raw, fname, events=None, overwrite=False):
    """Export raw data from MNE-Python.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data to export.
    fname : str | pathlib.Path
        The name of the file where raw data will be exported to. Must end with
        ``".vhdr"``, and accompanying *.vmrk* and *.eeg* files will be written inside
        the same directory.
    events : np.ndarray | None
        Events to be written to the marker file (*.vmrk*). If array, must be in
        `MNE-Python format <https://mne.tools/stable/glossary.html#term-events>`_. If
        ``None`` (default), events will be written based on ``raw.annotations``.
    overwrite : bool
        Whether or not to overwrite existing data. Defaults to ``False``.

    """
    # prepare file location
    if not str(fname).endswith(".vhdr"):
        raise ValueError("`fname` must have the '.vhdr' extension for BrainVision.")
    fname = Path(fname)
    folder_out = fname.parents[0]
    fname_base = fname.stem

    # prepare data from raw
    data = raw.get_data()  # gets data starting from raw.first_samp
    sfreq = raw.info["sfreq"]  # in Hz
    meas_date = raw.info["meas_date"]  # datetime.datetime
    ch_names = raw.ch_names

    # write voltage units as micro-volts and all other units without scaling
    # write units that we don't know as n/a
    unit = []
    for ch in raw.info["chs"]:
        if ch["unit"] == FIFF.FIFF_UNIT_V:
            unit.append("µV")
        elif ch["unit"] == FIFF.FIFF_UNIT_CEL:
            unit.append("°C")
        else:
            unit.append(_unit2human.get(ch["unit"], "n/a"))
    unit = [u if u != "NA" else "n/a" for u in unit]

    # enforce conversion to float32 format
    # XXX: Could add a feature that checks data and optimizes `unit`, `resolution`, and
    #      `format` so that raw.orig_format could be retained if reasonable.
    if raw.orig_format != "single":
        warn(
            f"Encountered data in '{raw.orig_format}' format. Converting to float32.",
            RuntimeWarning,
        )

    fmt = "binary_float32"
    resolution = 0.1

    # handle events
    # if we got an ndarray, this is in MNE-Python format
    msg = "`events` must be None or array in MNE-Python format."
    if events is not None:
        # subtract raw.first_samp because brainvision marks events starting from the
        # first available data point and ignores the raw.first_samp
        assert isinstance(events, np.ndarray), msg
        assert events.ndim == 2, msg
        assert events.shape[-1] == 3, msg
        events[:, 0] -= raw.first_samp
        events = events[:, [0, 2]]  # reorder for pybv required order
    else:  # else, prepare pybv style events from raw.annotations
        events = _mne_annots2pybv_events(raw)

    # no information about reference channels in mne currently
    ref_ch_names = None

    # write to BrainVision
    write_brainvision(
        data=data,
        sfreq=sfreq,
        ch_names=ch_names,
        ref_ch_names=ref_ch_names,
        fname_base=fname_base,
        folder_out=folder_out,
        overwrite=overwrite,
        events=events,
        resolution=resolution,
        unit=unit,
        fmt=fmt,
        meas_date=meas_date,
    )


def _mne_annots2pybv_events(raw):
    """Convert mne Annotations to pybv events."""
    # check that raw.annotations.orig_time is the same as raw.info["meas_date"]
    # so that onsets are relative to the first sample
    # (after further correction for first_time)
    if raw.annotations and raw.info["meas_date"] != raw.annotations.orig_time:
        raise ValueError(
            "Annotations must have the same orig_time as raw.info['meas_date']"
        )
    events = []
    for annot in raw.annotations:
        # handle onset and duration: seconds to sample, relative to
        # raw.first_samp / raw.first_time
        onset = annot["onset"] - raw.first_time
        onset = raw.time_as_index(onset).astype(int)[0]
        duration = int(annot["duration"] * raw.info["sfreq"])

        # triage type and description
        # defaults to type="Comment" and the full description
        etype = "Comment"
        description = annot["description"]
        for start in ["Stimulus/S", "Response/R", "Comment/"]:
            if description.startswith(start):
                etype = start.split("/")[0]
                description = description.replace(start, "")
                break

        if etype in ["Stimulus", "Response"] and description.strip().isdigit():
            description = int(description.strip())
        else:
            # if cannot convert to int, we must use this as "Comment"
            etype = "Comment"

        event_dict = dict(
            onset=onset,  # in samples
            duration=duration,  # in samples
            description=description,
            type=etype,
        )

        if "ch_names" in annot:
            # handle channels
            channels = list(annot["ch_names"])
            event_dict["channels"] = channels

        # add a "pybv" event
        events += [event_dict]

    return events


def _export_raw(fname, raw, overwrite):
    """Export Raw object to BrainVision via pybv."""
    fname = str(fname)
    ext = os.path.splitext(fname)[-1]
    if ext != ".vhdr":
        fname = fname.replace(ext, ".vhdr")
    _export_mne_raw(raw=raw, fname=fname, overwrite=overwrite)
