"""Read .res4 files."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os.path as op

import numpy as np

from ...utils import logger
from .constants import CTF


def _make_ctf_name(directory, extra, raise_error=True):
    """Make a CTF name."""
    fname = op.join(directory, op.basename(directory)[:-3] + "." + extra)
    found = True
    if not op.isfile(fname):
        if raise_error:
            raise OSError(f"Standard file {fname} not found")
        found = False
    return fname, found


def _read_double(fid, n=1):
    """Read a double."""
    return np.fromfile(fid, ">f8", n)


def _read_string(fid, n_bytes, decode=True):
    """Read string."""
    s0 = fid.read(n_bytes)
    s = s0.split(b"\x00")[0]
    return s.decode("utf-8") if decode else s


def _read_ustring(fid, n_bytes):
    """Read unsigned character string."""
    return np.fromfile(fid, ">B", n_bytes)


def _read_int2(fid):
    """Read int from short."""
    return _auto_cast(np.fromfile(fid, ">i2", 1)[0])


def _read_int(fid):
    """Read a 32-bit integer."""
    return np.fromfile(fid, ">i4", 1)[0]


def _move_to_next(fid, byte=8):
    """Move to next byte boundary."""
    now = fid.tell()
    if now % byte != 0:
        now = now - (now % byte) + byte
        fid.seek(now, 0)


def _read_filter(fid):
    """Read filter information."""
    f = dict()
    f["freq"] = _read_double(fid)[0]
    f["class"] = _read_int(fid)
    f["type"] = _read_int(fid)
    f["npar"] = _read_int2(fid)
    f["pars"] = _read_double(fid, f["npar"])
    return f


def _read_comp_coeff(fid, d):
    """Read compensation coefficients."""
    # Read the coefficients and initialize
    d["ncomp"] = _read_int2(fid)
    d["comp"] = list()
    # Read each record
    dt = np.dtype(
        [
            ("sensor_name", "S32"),
            ("coeff_type", ">i4"),
            ("d0", ">i4"),
            ("ncoeff", ">i2"),
            ("sensors", f"S{CTF.CTFV_SENSOR_LABEL}", CTF.CTFV_MAX_BALANCING),
            ("coeffs", ">f8", CTF.CTFV_MAX_BALANCING),
        ]
    )
    comps = np.fromfile(fid, dt, d["ncomp"])
    for k in range(d["ncomp"]):
        comp = dict()
        d["comp"].append(comp)
        comp["sensor_name"] = comps["sensor_name"][k].split(b"\x00")[0].decode("utf-8")
        comp["coeff_type"] = comps["coeff_type"][k].item()
        comp["ncoeff"] = comps["ncoeff"][k].item()
        comp["sensors"] = [
            s.split(b"\x00")[0].decode("utf-8")
            for s in comps["sensors"][k][: comp["ncoeff"]]
        ]
        comp["coeffs"] = comps["coeffs"][k][: comp["ncoeff"]]
        comp["scanno"] = d["ch_names"].index(comp["sensor_name"])


def _read_res4(dsdir):
    """Read the magical res4 file."""
    # adapted from read_res4.c
    name, _ = _make_ctf_name(dsdir, "res4")
    res = dict()
    with open(name, "rb") as fid:
        # Read the fields
        res["head"] = _read_string(fid, 8)
        res["appname"] = _read_string(fid, 256)
        res["origin"] = _read_string(fid, 256)
        res["desc"] = _read_string(fid, 256)
        res["nave"] = _read_int2(fid)
        res["data_time"] = _read_string(fid, 255)
        res["data_date"] = _read_string(fid, 255)
        # Seems that date and time can be swapped
        # (are they entered manually?!)
        if "/" in res["data_time"] and ":" in res["data_date"]:
            data_date = res["data_date"]
            res["data_date"] = res["data_time"]
            res["data_time"] = data_date
        res["nsamp"] = _read_int(fid)
        res["nchan"] = _read_int2(fid)
        _move_to_next(fid, 8)
        res["sfreq"] = _read_double(fid)[0]
        res["epoch_time"] = _read_double(fid)[0]
        res["no_trials"] = _read_int2(fid)
        _move_to_next(fid, 4)
        res["pre_trig_pts"] = _read_int(fid)
        res["no_trials_done"] = _read_int2(fid)
        res["no_trials_bst_message_windowlay"] = _read_int2(fid)
        _move_to_next(fid, 4)
        res["save_trials"] = _read_int(fid)
        res["primary_trigger"] = fid.read(1)
        res["secondary_trigger"] = [
            fid.read(1) for k in range(CTF.CTFV_MAX_AVERAGE_BINS)
        ]
        res["trigger_polarity_mask"] = fid.read(1)
        res["trigger_mode"] = _read_int2(fid)
        _move_to_next(fid, 4)
        res["accept_reject"] = _read_int(fid)
        res["run_time_bst_message_windowlay"] = _read_int2(fid)
        _move_to_next(fid, 4)
        res["zero_head"] = _read_int(fid)
        _move_to_next(fid, 4)
        res["artifact_mode"] = _read_int(fid)
        _read_int(fid)  # padding
        res["nf_run_name"] = _read_string(fid, 32)
        res["nf_run_title"] = _read_string(fid, 256)
        res["nf_instruments"] = _read_string(fid, 32)
        res["nf_collect_descriptor"] = _read_string(fid, 32)
        res["nf_subject_id"] = _read_string(fid, 32)
        res["nf_operator"] = _read_string(fid, 32)
        if len(res["nf_operator"]) == 0:
            res["nf_operator"] = None
        res["nf_sensor_file_name"] = _read_ustring(fid, 60)
        _move_to_next(fid, 4)
        res["rdlen"] = _read_int(fid)
        fid.seek(CTF.FUNNY_POS, 0)

        if res["rdlen"] > 0:
            res["run_desc"] = _read_string(fid, res["rdlen"])

        # Filters
        res["nfilt"] = _read_int2(fid)
        res["filters"] = list()
        for k in range(res["nfilt"]):
            res["filters"].append(_read_filter(fid))

        # Channel information (names, then data)
        res["ch_names"] = list()
        for k in range(res["nchan"]):
            ch_name = _read_string(fid, 32)
            res["ch_names"].append(ch_name)
        _coil_dt = np.dtype(
            [
                ("pos", ">f8", 3),
                ("d0", ">f8"),
                ("norm", ">f8", 3),
                ("d1", ">f8"),
                ("turns", ">i2"),
                ("d2", ">i4"),
                ("d3", ">i2"),
                ("area", ">f8"),
            ]
        )
        _ch_dt = np.dtype(
            [
                ("sensor_type_index", ">i2"),
                ("original_run_no", ">i2"),
                ("coil_type", ">i4"),
                ("proper_gain", ">f8"),
                ("qgain", ">f8"),
                ("io_gain", ">f8"),
                ("io_offset", ">f8"),
                ("num_coils", ">i2"),
                ("grad_order_no", ">i2"),
                ("d0", ">i4"),
                ("coil", _coil_dt, CTF.CTFV_MAX_COILS),
                ("head_coil", _coil_dt, CTF.CTFV_MAX_COILS),
            ]
        )
        chs = np.fromfile(fid, _ch_dt, res["nchan"])
        for coil in (chs["coil"], chs["head_coil"]):
            coil["pos"] /= 100.0
            coil["area"] *= 1e-4
        # convert to dict
        chs = [dict(zip(chs.dtype.names, x)) for x in chs]
        for ch in chs:
            for key, val in ch.items():
                ch[key] = _auto_cast(val)
        res["chs"] = chs
        for k in range(res["nchan"]):
            res["chs"][k]["ch_name"] = res["ch_names"][k]

        # The compensation coefficients
        _read_comp_coeff(fid, res)
    logger.info("    res4 data read.")
    return res


def _auto_cast(x):
    # Upcast scalars
    if isinstance(x, np.ScalarType):
        if x.dtype.kind == "i":
            if x.dtype != np.int64:
                x = x.astype(np.int64)
        elif x.dtype.kind == "f":
            if x.dtype != np.float64:
                x = x.astype(np.float64)
    return x
