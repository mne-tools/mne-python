"""Read .eeg files."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from os import listdir
from os.path import join

import numpy as np

from ..._fiff.constants import FIFF
from ...transforms import apply_trans
from ...utils import logger, warn
from .res4 import _make_ctf_name

_cardinal_dict = dict(
    nasion=FIFF.FIFFV_POINT_NASION,
    lpa=FIFF.FIFFV_POINT_LPA,
    left=FIFF.FIFFV_POINT_LPA,
    rpa=FIFF.FIFFV_POINT_RPA,
    right=FIFF.FIFFV_POINT_RPA,
)


def _read_eeg(directory):
    """Read the .eeg file."""
    # Missing file is ok
    fname, found = _make_ctf_name(directory, "eeg", raise_error=False)
    if not found:
        logger.info("    Separate EEG position data file not present.")
        return
    eeg = dict(
        labels=list(),
        kinds=list(),
        ids=list(),
        rr=list(),
        np=0,
        assign_to_chs=True,
        coord_frame=FIFF.FIFFV_MNE_COORD_CTF_HEAD,
    )
    with open(fname, "rb") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0:
                parts = line.decode("utf-8").split()
                if len(parts) != 5:
                    raise RuntimeError(f"Illegal data in EEG position file: {line}")
                r = np.array([float(p) for p in parts[2:]]) / 100.0
                if (r * r).sum() > 1e-4:
                    label = parts[1]
                    eeg["labels"].append(label)
                    eeg["rr"].append(r)
                    id_ = _cardinal_dict.get(label.lower(), int(parts[0]))
                    if label.lower() in _cardinal_dict:
                        kind = FIFF.FIFFV_POINT_CARDINAL
                    else:
                        kind = FIFF.FIFFV_POINT_EXTRA
                    eeg["ids"].append(id_)
                    eeg["kinds"].append(kind)
                    eeg["np"] += 1
    logger.info("    Separate EEG position data file read.")
    return eeg


def _read_pos(directory, transformations):
    """Read the .pos file and return eeg positions as dig extra points."""
    fname = [join(directory, f) for f in listdir(directory) if f.endswith(".pos")]
    if len(fname) < 1:
        return list()
    elif len(fname) > 1:
        warn("    Found multiple pos files. Extra digitizer points not added.")
        return list()
    logger.info(f"    Reading digitizer points from {fname}...")
    if transformations["t_ctf_head_head"] is None:
        warn("    No transformation found. Extra digitizer points not added.")
        return list()
    fname = fname[0]
    digs = list()
    i = 2000
    with open(fname) as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0:
                parts = line.split()
                # The lines can have 4 or 5 parts. First part is for the id,
                # which can be an int or a string. The last three are for xyz
                # coordinates. The extra part is for additional info
                # (e.g. 'Pz', 'Cz') which is ignored.
                if len(parts) not in [4, 5]:
                    continue
                try:
                    ident = int(parts[0]) + 1000
                except ValueError:  # if id is not an int
                    ident = i
                    i += 1
                dig = dict(
                    kind=FIFF.FIFFV_POINT_EXTRA,
                    ident=ident,
                    r=list(),
                    coord_frame=FIFF.FIFFV_COORD_HEAD,
                )
                r = np.array([float(p) for p in parts[-3:]]) / 100.0  # cm to m
                if (r * r).sum() > 1e-4:
                    r = apply_trans(transformations["t_ctf_head_head"], r)
                    dig["r"] = r
                    digs.append(dig)
    return digs
