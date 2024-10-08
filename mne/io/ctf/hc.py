"""Read .hc files."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ..._fiff.constants import FIFF
from ...utils import logger
from .constants import CTF
from .res4 import _make_ctf_name

_kind_dict = {
    "nasion": CTF.CTFV_COIL_NAS,
    "left ear": CTF.CTFV_COIL_LPA,
    "right ear": CTF.CTFV_COIL_RPA,
    "spare": CTF.CTFV_COIL_SPARE,
}

_coord_dict = {
    "relative to dewar": FIFF.FIFFV_MNE_COORD_CTF_DEVICE,
    "relative to head": FIFF.FIFFV_MNE_COORD_CTF_HEAD,
}


def _read_one_coil_point(fid):
    """Read coil coordinate information from the hc file."""
    # Descriptor
    one = "#"
    while len(one) > 0 and one[0] == "#":
        one = fid.readline()
    if len(one) == 0:
        return None
    one = one.strip().decode("utf-8")
    if "Unable" in one:
        raise RuntimeError("HPI information not available")

    # Hopefully this is an unambiguous interpretation
    p = dict()
    p["valid"] = "measured" in one
    for key, val in _coord_dict.items():
        if key in one:
            p["coord_frame"] = val
            break
    else:
        p["coord_frame"] = -1

    for key, val in _kind_dict.items():
        if key in one:
            p["kind"] = val
            break
    else:
        p["kind"] = -1

    # Three coordinates
    p["r"] = np.empty(3)
    for ii, coord in enumerate("xyz"):
        sp = fid.readline().decode("utf-8").strip()
        if len(sp) == 0:  # blank line
            continue
        sp = sp.split(" ")
        if len(sp) != 3 or sp[0] != coord or sp[1] != "=":
            raise RuntimeError(f"Bad line: {one}")
        # We do not deal with centimeters
        p["r"][ii] = float(sp[2]) / 100.0
    return p


def _read_hc(directory):
    """Read the hc file to get the HPI info and to prepare for coord trans."""
    fname, found = _make_ctf_name(directory, "hc", raise_error=False)
    if not found:
        logger.info("    hc data not present")
        return None
    s = list()
    with open(fname, "rb") as fid:
        while True:
            p = _read_one_coil_point(fid)
            if p is None:
                # First point bad indicates that the file is empty
                if len(s) == 0:
                    logger.info("hc file empty, no data present")
                    return None
                # Returns None if at EOF
                logger.info("    hc data read.")
                return s
            if p["valid"]:
                s.append(p)
