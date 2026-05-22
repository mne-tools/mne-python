# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from functools import partial

import numpy as np

from ..fixes import _csc_array_cast
from ..utils import _check_fname, logger, warn
from .constants import FIFF
from .open import fiff_open, read_tag
from .tag import _float_item, _int_item, find_tag
from .tree import dir_tree_find
from .write import (
    _safe_name_list,
    end_block,
    start_block,
    write_float,
    write_float_matrix,
    write_float_sparse,
    write_id,
    write_int,
    write_int_matrix,
    write_name_list_sanitized,
    write_string,
)

_proc_map = dict(  # ID, caster, writer
    parent_file_id=(
        FIFF.FIFF_PARENT_FILE_ID,
        dict,
        write_id,
    ),
    block_id=(
        FIFF.FIFF_BLOCK_ID,
        dict,
        write_id,
    ),
    parent_block_id=(
        FIFF.FIFF_PARENT_BLOCK_ID,
        dict,
        write_id,
    ),
    date=(
        FIFF.FIFF_MEAS_DATE,
        lambda d: tuple(int(dd) for dd in d),
        write_int,
    ),
    experimenter=(
        FIFF.FIFF_EXPERIMENTER,
        str,
        write_string,
    ),
    creator=(
        FIFF.FIFF_CREATOR,
        str,
        write_string,
    ),
)


def _read_proc_history(fid, tree):
    """Read processing history from fiff file.

    This function reads the SSS info, the CTC correction and the
    calibaraions from the SSS processing logs inside af a raw file
    (C.f. Maxfilter v2.2 manual (October 2010), page 21)::

        104 = {                 900 = proc. history
          104 = {               901 = proc. record
            103 = block ID
            204 = date
            212 = scientist
            113 = creator program
            104 = {             502 = SSS info
              264 = SSS task
              263 = SSS coord frame
              265 = SSS origin
              266 = SSS ins.order
              267 = SSS outs.order
              268 = SSS nr chnls
              269 = SSS components
              278 = SSS nfree
              243 = HPI g limit    0.98
              244 = HPI dist limit 0.005
            105 = }             502 = SSS info
            104 = {             504 = MaxST info
              264 = SSS task
              272 = SSST subspace correlation
              279 = SSST buffer length
            105 = }
            104 = {             501 = CTC correction
              103 = block ID
              204 = date
              113 = creator program
              800 = CTC matrix
              3417 = proj item chs
            105 = }             501 = CTC correction
            104 = {             503 = SSS finecalib.
              270 = SSS cal chnls
              271 = SSS cal coeff
            105 = }             503 = SSS finecalib.
          105 = }               901 = proc. record
        105 = }                 900 = proc. history
    """
    proc_history = dir_tree_find(tree, FIFF.FIFFB_PROCESSING_HISTORY)
    out = list()
    if len(proc_history) > 0:
        proc_history = proc_history[0]
        proc_records = dir_tree_find(proc_history, FIFF.FIFFB_PROCESSING_RECORD)
        for proc_record in proc_records:
            record = dict()
            for i_ent in range(proc_record["nent"]):
                kind = proc_record["directory"][i_ent].kind
                pos = proc_record["directory"][i_ent].pos
                for key, (id_, cast, _) in _proc_map.items():
                    if kind == id_:
                        tag = read_tag(fid, pos)
                        record[key] = cast(tag.data)
                        break
                else:
                    warn(f"Unknown processing history item {kind}")
            record["max_info"] = _read_maxfilter_record(fid, proc_record)
            iass = dir_tree_find(proc_record, FIFF.FIFFB_IAS)
            if len(iass) > 0:
                # XXX should eventually populate this
                ss = [dict() for _ in range(len(iass))]
                record["ias"] = ss
            if len(record["max_info"]) > 0:
                out.append(record)
    return out


def _write_proc_history(fid, info):
    """Write processing history to file."""
    if len(info["proc_history"]) > 0:
        start_block(fid, FIFF.FIFFB_PROCESSING_HISTORY)
        for record in info["proc_history"]:
            start_block(fid, FIFF.FIFFB_PROCESSING_RECORD)
            for key, (id_, _, writer) in _proc_map.items():
                if key in record:
                    writer(fid, id_, record[key])
            _write_maxfilter_record(fid, record["max_info"])
            if "ias" in record:
                for _ in record["ias"]:
                    start_block(fid, FIFF.FIFFB_IAS)
                    # XXX should eventually populate this
                    end_block(fid, FIFF.FIFFB_IAS)
            end_block(fid, FIFF.FIFFB_PROCESSING_RECORD)
        end_block(fid, FIFF.FIFFB_PROCESSING_HISTORY)


_sss_info_keys = (
    "job",
    "frame",
    "origin",
    "in_order",
    "out_order",
    "nchan",
    "components",
    "nfree",
    "hpi_g_limit",
    "hpi_dist_limit",
)
_sss_info_ids = (
    FIFF.FIFF_SSS_JOB,
    FIFF.FIFF_SSS_FRAME,
    FIFF.FIFF_SSS_ORIGIN,
    FIFF.FIFF_SSS_ORD_IN,
    FIFF.FIFF_SSS_ORD_OUT,
    FIFF.FIFF_SSS_NMAG,
    FIFF.FIFF_SSS_COMPONENTS,
    FIFF.FIFF_SSS_NFREE,
    FIFF.FIFF_HPI_FIT_GOOD_LIMIT,
    FIFF.FIFF_HPI_FIT_DIST_LIMIT,
)
_sss_info_writers = (
    write_int,
    write_int,
    write_float,
    write_int,
    write_int,
    write_int,
    write_int,
    write_int,
    write_float,
    write_float,
)
_sss_info_casters = (
    _int_item,
    _int_item,
    np.array,
    _int_item,
    _int_item,
    _int_item,
    np.array,
    _int_item,
    _float_item,
    _float_item,
)

_max_st_keys = ("job", "subspcorr", "buflen")
_max_st_ids = (FIFF.FIFF_SSS_JOB, FIFF.FIFF_SSS_ST_CORR, FIFF.FIFF_SSS_ST_LENGTH)
_max_st_writers = (write_int, write_float, write_float)
_max_st_casters = (_int_item, _float_item, _float_item)


def _sss_ctc_ch_name_clean(tag_data):
    """Clean channel names from CTC files."""
    chs = _safe_name_list(tag_data, "read", "ch_names")
    # CTC files can have null chars in the last entry, e.g.:
    # [..., 'MEG2642', 'MEG2643', 'MEG2641\x00 ... \x00']
    if len(chs) > 0:
        chs[-1] = chs[-1].split("\x00")[0]
    return chs


_sss_ctc_keys = (
    "block_id",
    "parent_block_id",
    "date",
    "creator",
    "decoupler",
    "ch_names",
)
_sss_ctc_ids = (
    FIFF.FIFF_BLOCK_ID,
    FIFF.FIFF_PARENT_BLOCK_ID,
    FIFF.FIFF_MEAS_DATE,
    FIFF.FIFF_CREATOR,
    FIFF.FIFF_DECOUPLER_MATRIX,
    FIFF.FIFF_PROJ_ITEM_CH_NAME_LIST,
)
_sss_ctc_writers = (
    write_id,
    write_id,
    write_int,
    write_string,
    write_float_sparse,
    partial(write_name_list_sanitized, name="ch_names"),
)
_sss_ctc_casters = (
    dict,
    dict,
    np.array,
    str,
    _csc_array_cast,
    _sss_ctc_ch_name_clean,
)

_sss_cal_keys = ("cal_chans", "cal_corrs")
_sss_cal_ids = (FIFF.FIFF_SSS_CAL_CHANS, FIFF.FIFF_SSS_CAL_CORRS)
_sss_cal_writers = (write_int_matrix, write_float_matrix)
_sss_cal_casters = (np.array, np.array)

_io_map = dict(
    sss_info=dict(
        keys=_sss_info_keys,
        ids=_sss_info_ids,
        casters=_sss_info_casters,
        writers=_sss_info_writers,
        block=FIFF.FIFFB_SSS_INFO,
    ),
    max_st=dict(
        keys=_max_st_keys,
        ids=_max_st_ids,
        casters=_max_st_casters,
        writers=_max_st_writers,
        block=FIFF.FIFFB_SSS_ST_INFO,
    ),
    sss_ctc=dict(
        keys=_sss_ctc_keys,
        ids=_sss_ctc_ids,
        casters=_sss_ctc_casters,
        writers=_sss_ctc_writers,
        block=FIFF.FIFFB_CHANNEL_DECOUPLER,
    ),
    sss_cal=dict(
        keys=_sss_cal_keys,
        ids=_sss_cal_ids,
        casters=_sss_cal_casters,
        writers=_sss_cal_writers,
        block=FIFF.FIFFB_SSS_CAL,
    ),
)


def _write_mf_data(fid, info, *, kind, key=None):
    key = kind if key is None else key
    this_data = info.get(key, dict())
    if not this_data:  # empty or None
        return
    del info, key
    logger.debug("Writing %s info with keys: %s", kind, list(this_data))
    keys, ids, _, writers, block = _io_map[kind].values()
    start_block(fid, block)
    for key, id_, writer in zip(keys, ids, writers):
        if key in this_data:
            writer(fid, id_, this_data[key])
    end_block(fid, block)


def _read_ctc(fname):
    """Read cross-talk correction matrix."""
    fname = _check_fname(fname, overwrite="read", must_exist=True)
    f, tree, _ = fiff_open(fname)
    with f as fid:
        sss_ctc = _read_mf_data(fid, tree, kind="sss_ctc")
        bad_str = f"Invalid cross-talk FIF: {fname}"
        if len(sss_ctc) == 0:
            raise ValueError(bad_str)
        node = dir_tree_find(tree, FIFF.FIFFB_DATA_CORRECTION)[0]
        comment = find_tag(fid, node, FIFF.FIFF_COMMENT).data
        if comment != "cross-talk compensation matrix":
            raise ValueError(bad_str)
        sss_ctc["creator"] = find_tag(fid, node, FIFF.FIFF_CREATOR).data
        sss_ctc["date"] = find_tag(fid, node, FIFF.FIFF_MEAS_DATE).data
    return sss_ctc


def _read_maxfilter_record(fid, tree):
    """Read maxfilter processing record from file."""
    max_info = dict()
    for key in _io_map:
        this_data = _read_mf_data(fid, tree, kind=key)
        max_info[key] = this_data or dict()  # always add, even if empty
    return max_info


def _read_mf_data(fid, tree, *, kind):
    keys, ids, casters, _, block = _io_map[kind].values()
    sss_kind_block = dir_tree_find(tree, block)
    sss_out = dict()
    if len(sss_kind_block) > 0:
        logger.debug("Reading %s info with keys: %s", kind, list(keys))
        sss_kind_block = sss_kind_block[0]
        for i_ent in range(sss_kind_block["nent"]):
            kind = sss_kind_block["directory"][i_ent].kind
            pos = sss_kind_block["directory"][i_ent].pos
            for key, id_, cast in zip(keys, ids, casters):
                if kind == id_:
                    tag = read_tag(fid, pos)
                    sss_out[key] = cast(tag.data)
                    break
    return sss_out


def _write_maxfilter_record(fid, record):
    """Write maxfilter processing record to file."""
    for key in _io_map:
        _write_mf_data(fid, record, kind=key)
