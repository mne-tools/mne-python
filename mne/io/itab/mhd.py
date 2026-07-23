"""Read .mhd file."""

# Author: Vittorio Pizzella <vittorio.pizzella@unich.it>
#
# License: BSD (3-clause)

import numpy as np

from ...utils import logger
from .constants import ITAB


def _read_double(fid, n=1):
    """Read a double."""
    return np.fromfile(fid, "<f8", n)


def _read_string(fid, n_bytes, decode=True):
    """Read string."""
    s0 = fid.read(n_bytes)
    s = s0.split(b"\x00")[0]
    return s.decode("utf-8") if decode else s


def _read_char(fid, n=1):
    """Read character."""
    return np.fromfile(fid, ">B", n)


# TODO: Is it similar to _read_char
def _read_ustring(fid, n_bytes):
    """Read unsigned character string."""
    return np.fromfile(fid, ">B", n_bytes)


def _read_int2(fid):
    """Read int from short."""
    return np.fromfile(fid, "<i2", 1)[0]


# TODO: Maybe we could use only one of _read_int*
def _read_int(fid):
    """Read int."""
    return np.fromfile(fid, "<i4", 1)[0]


def _read_float(fid):
    """Read float."""
    return np.fromfile(fid, "<f4", 1)[0]


def _read_position(fid):
    """Read position and orientation."""
    pos = dict()
    pos["posx"] = _read_float(fid)
    pos["posy"] = _read_float(fid)
    pos["posz"] = _read_float(fid)
    pos["orix"] = _read_float(fid)
    pos["oriy"] = _read_float(fid)
    pos["oriz"] = _read_float(fid)
    return pos


def _read_sample(fid):
    """Read sample info."""
    smpl = dict()
    smpl["start"] = _read_int(fid)
    smpl["ntptot"] = _read_int(fid)
    smpl["ntppre"] = _read_int(fid)
    smpl["type"] = _read_int(fid)
    smpl["quality"] = _read_int(fid)
    return smpl


def _read_ch_info(fid):
    """Read channel information."""
    ch = dict()
    ch["type"] = _read_int(fid)
    ch["number"] = _read_int(fid)
    ch["label"] = _read_string(fid, 16)
    ch["flag"] = _read_int(fid)
    ch["amvbit"] = _read_float(fid)
    ch["calib"] = np.double(_read_float(fid))
    ch["unit"] = _read_string(fid, 8)
    ch["ncoils"] = _read_int(fid)
    ch["wgt"] = list()

    for k in range(10):
        w = _read_float(fid)
        if k < ch["ncoils"]:
            ch["wgt"].append(w)

    ch["pos"] = list()
    for k in range(10):
        p = _read_position(fid)
        if k < ch["ncoils"]:
            ch["pos"].append(p)
    return ch


def _read_mhd(name):
    """Read the mhd file."""
    mhd = dict()
    with open(name, "rb") as fid:
        # Read the different fields
        mhd["stname"] = _read_string(fid, 10)
        mhd["stver"] = _read_string(fid, 8)
        mhd["stendian"] = _read_string(fid, 4)

        mhd["first_name"] = _read_string(fid, 32)
        mhd["last_name"] = _read_string(fid, 32)
        mhd["id"] = _read_string(fid, 32)
        mhd["notes"] = _read_string(fid, 256)
        mhd["subinfo"] = {
            "sex": _read_char(fid),
            "notes": _read_string(fid, 256),
            "height": _read_float(fid),
            "weight": _read_float(fid),
            "birthday": _read_int(fid),
            "birthmonth": _read_int(fid),
            "birthyear": _read_int(fid),
        }
        fid.seek(ITAB.MHD_TIME, 0)
        mhd["time"] = _read_string(fid, 12)
        mhd["date"] = _read_string(fid, 16)

        mhd["nchan"] = _read_int(fid)
        mhd["nelech"] = _read_int(fid)
        mhd["nelerefch"] = _read_int(fid)
        mhd["nmagch"] = _read_int(fid)
        mhd["nmagchref"] = _read_int(fid)
        mhd["nauxch"] = _read_int(fid)
        mhd["nparamch"] = _read_int(fid)
        mhd["ndigitch"] = _read_int(fid)
        mhd["nflagch"] = _read_int(fid)

        mhd["data_type"] = _read_int(fid)

        mhd["smpfq"] = _read_float(fid)
        mhd["hw_low_fr"] = _read_float(fid)
        mhd["hw_hig_fr"] = _read_float(fid)
        mhd["hw_comb"] = _read_int(fid)
        mhd["sw_hig_tc"] = _read_float(fid)
        mhd["compensation"] = _read_int(fid)
        mhd["ntpdata"] = _read_int(fid)

        mhd["no_segments"] = _read_int(fid)

        mhd["segment"] = list()
        n_segments = mhd["no_segments"]
        if n_segments == 0:
            logger.warning("No segments found in the mhd file.")
        else:
            logger.debug(f"Found {n_segments} segments in the mhd file.")
            for k in range(n_segments):
                segment = _read_sample(fid)
                mhd["segment"].append(segment)

        mhd["nsmpl"] = _read_int(fid)
        mhd["sample"] = list()
        for k in range(mhd["nsmpl"]):
            sample = _read_sample(fid)
            mhd["sample"].append(sample)

        fid.seek(ITAB.MHD_NREFCH, 0)
        mhd["nrefchan"] = _read_int(fid)
        mhd["ref_ch"] = list()
        for k in range(mhd["nrefchan"]):
            mhd["ref_ch"].append(_read_int(fid))

        fid.seek(ITAB.MHD_RAWHDTYPE, 0)
        mhd["raw_header_type"] = _read_int(fid)
        mhd["header_type"] = _read_int(fid)

        mhd["conf_file"] = _read_string(fid, 64)

        mhd["header_size"] = _read_int(fid)
        mhd["start_reference"] = _read_int(fid)
        mhd["start_data"] = _read_int(fid)

        mhd["rawfile"] = _read_int(fid)
        mhd["multiplexed_data"] = _read_int(fid)
        mhd["isns"] = _read_int(fid)

        fid.seek(ITAB.MHD_CHINFO, 0)
        mhd["ch"] = list()
        for k in range(mhd["nchan"]):
            mhd["ch"].append(_read_ch_info(fid))

        fid.seek(ITAB.MHD_SENSPOS, 0)
        mhd["r_center"] = _read_position(fid)
        mhd["u_center"] = _read_position(fid)
        mhd["th_center"] = _read_float(fid)
        mhd["fi_center"] = _read_float(fid)
        mhd["rotation-angle"] = _read_float(fid)
        mhd["cosdir"] = list()
        for k in range(3):
            mhd["cosdir"].append(_read_position(fid))
        mhd["irefsys"] = _read_int(fid)

        fid.seek(ITAB.MHD_NMARKER, 0)
        mhd["num_markers"] = _read_int(fid)

        mhd["i_coil"] = list()
        for k in range(mhd["num_markers"]):
            mhd["i_coil"].append(_read_int(fid))

        fid.seek(ITAB.MHD_MARKERS, 0)
        mhd["marker"] = list()
        for k in range(mhd["num_markers"]):
            mhd["marker"].append(_read_position(fid))

    logger.info(f"Header {name} correctly read.")

    return mhd
