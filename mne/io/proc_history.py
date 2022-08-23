# -*- coding: utf-8 -*-
# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
# License: Simplified BSD

import numpy as np

from .open import read_tag, fiff_open
from .tree import dir_tree_find
from .write import (start_block, end_block, write_int, write_float,
                    write_string, write_float_matrix, write_int_matrix,
                    write_float_sparse, write_id)
from .tag import find_tag
from .constants import FIFF
from ..fixes import _csc_matrix_cast
from ..utils import warn, _check_fname

_proc_keys = ['parent_file_id', 'block_id', 'parent_block_id',
              'date', 'experimenter', 'creator']
_proc_ids = [FIFF.FIFF_PARENT_FILE_ID,
             FIFF.FIFF_BLOCK_ID,
             FIFF.FIFF_PARENT_BLOCK_ID,
             FIFF.FIFF_MEAS_DATE,
             FIFF.FIFF_EXPERIMENTER,
             FIFF.FIFF_CREATOR]
_proc_writers = [write_id, write_id, write_id,
                 write_int, write_string, write_string]
_proc_casters = [dict, dict, dict, np.array, str, str]


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
        proc_records = dir_tree_find(proc_history,
                                     FIFF.FIFFB_PROCESSING_RECORD)
        for proc_record in proc_records:
            record = dict()
            for i_ent in range(proc_record['nent']):
                kind = proc_record['directory'][i_ent].kind
                pos = proc_record['directory'][i_ent].pos
                for key, id_, cast in zip(_proc_keys, _proc_ids,
                                          _proc_casters):
                    if kind == id_:
                        tag = read_tag(fid, pos)
                        record[key] = cast(tag.data)
                        break
                else:
                    warn('Unknown processing history item %s' % kind)
            record['max_info'] = _read_maxfilter_record(fid, proc_record)
            iass = dir_tree_find(proc_record, FIFF.FIFFB_IAS)
            if len(iass) > 0:
                # XXX should eventually populate this
                ss = [dict() for _ in range(len(iass))]
                record['ias'] = ss
            if len(record['max_info']) > 0:
                out.append(record)
    return out


def _write_proc_history(fid, info):
    """Write processing history to file."""
    if len(info['proc_history']) > 0:
        start_block(fid, FIFF.FIFFB_PROCESSING_HISTORY)
        for record in info['proc_history']:
            start_block(fid, FIFF.FIFFB_PROCESSING_RECORD)
            for key, id_, writer in zip(_proc_keys, _proc_ids, _proc_writers):
                if key in record:
                    writer(fid, id_, record[key])
            _write_maxfilter_record(fid, record['max_info'])
            if 'ias' in record:
                for _ in record['ias']:
                    start_block(fid, FIFF.FIFFB_IAS)
                    # XXX should eventually populate this
                    end_block(fid, FIFF.FIFFB_IAS)
            end_block(fid, FIFF.FIFFB_PROCESSING_RECORD)
        end_block(fid, FIFF.FIFFB_PROCESSING_HISTORY)


_sss_info_keys = ('job', 'frame', 'origin', 'in_order',
                  'out_order', 'nchan', 'components', 'nfree',
                  'hpi_g_limit', 'hpi_dist_limit')
_sss_info_ids = (FIFF.FIFF_SSS_JOB,
                 FIFF.FIFF_SSS_FRAME,
                 FIFF.FIFF_SSS_ORIGIN,
                 FIFF.FIFF_SSS_ORD_IN,
                 FIFF.FIFF_SSS_ORD_OUT,
                 FIFF.FIFF_SSS_NMAG,
                 FIFF.FIFF_SSS_COMPONENTS,
                 FIFF.FIFF_SSS_NFREE,
                 FIFF.FIFF_HPI_FIT_GOOD_LIMIT,
                 FIFF.FIFF_HPI_FIT_DIST_LIMIT)
_sss_info_writers = (write_int, write_int, write_float, write_int,
                     write_int, write_int, write_int, write_int,
                     write_float, write_float)
_sss_info_casters = (int, int, np.array, int,
                     int, int, np.array, int,
                     float, float)

_max_st_keys = ('job', 'subspcorr', 'buflen')
_max_st_ids = (FIFF.FIFF_SSS_JOB, FIFF.FIFF_SSS_ST_CORR,
               FIFF.FIFF_SSS_ST_LENGTH)
_max_st_writers = (write_int, write_float, write_float)
_max_st_casters = (int, float, float)

_sss_ctc_keys = ('block_id', 'date', 'creator', 'decoupler')
_sss_ctc_ids = (FIFF.FIFF_BLOCK_ID,
                FIFF.FIFF_MEAS_DATE,
                FIFF.FIFF_CREATOR,
                FIFF.FIFF_DECOUPLER_MATRIX)
_sss_ctc_writers = (write_id, write_int, write_string, write_float_sparse)
_sss_ctc_casters = (dict, np.array, str, _csc_matrix_cast)

_sss_cal_keys = ('cal_chans', 'cal_corrs')
_sss_cal_ids = (FIFF.FIFF_SSS_CAL_CHANS, FIFF.FIFF_SSS_CAL_CORRS)
_sss_cal_writers = (write_int_matrix, write_float_matrix)
_sss_cal_casters = (np.array, np.array)


def _read_ctc(fname):
    """Read cross-talk correction matrix."""
    fname = _check_fname(fname, overwrite='read', must_exist=True)
    f, tree, _ = fiff_open(fname)
    with f as fid:
        sss_ctc = _read_maxfilter_record(fid, tree)['sss_ctc']
        bad_str = 'Invalid cross-talk FIF: %s' % fname
        if len(sss_ctc) == 0:
            raise ValueError(bad_str)
        node = dir_tree_find(tree, FIFF.FIFFB_DATA_CORRECTION)[0]
        comment = find_tag(fid, node, FIFF.FIFF_COMMENT).data
        if comment != 'cross-talk compensation matrix':
            raise ValueError(bad_str)
        sss_ctc['creator'] = find_tag(fid, node, FIFF.FIFF_CREATOR).data
        sss_ctc['date'] = find_tag(fid, node, FIFF.FIFF_MEAS_DATE).data
    return sss_ctc


def _read_maxfilter_record(fid, tree):
    """Read maxfilter processing record from file."""
    sss_info_block = dir_tree_find(tree, FIFF.FIFFB_SSS_INFO)  # 502
    sss_info = dict()
    if len(sss_info_block) > 0:
        sss_info_block = sss_info_block[0]
        for i_ent in range(sss_info_block['nent']):
            kind = sss_info_block['directory'][i_ent].kind
            pos = sss_info_block['directory'][i_ent].pos
            for key, id_, cast in zip(_sss_info_keys, _sss_info_ids,
                                      _sss_info_casters):
                if kind == id_:
                    tag = read_tag(fid, pos)
                    sss_info[key] = cast(tag.data)
                    break

    max_st_block = dir_tree_find(tree, FIFF.FIFFB_SSS_ST_INFO)  # 504
    max_st = dict()
    if len(max_st_block) > 0:
        max_st_block = max_st_block[0]
        for i_ent in range(max_st_block['nent']):
            kind = max_st_block['directory'][i_ent].kind
            pos = max_st_block['directory'][i_ent].pos
            for key, id_, cast in zip(_max_st_keys, _max_st_ids,
                                      _max_st_casters):
                if kind == id_:
                    tag = read_tag(fid, pos)
                    max_st[key] = cast(tag.data)
                    break

    sss_ctc_block = dir_tree_find(tree, FIFF.FIFFB_CHANNEL_DECOUPLER)  # 501
    sss_ctc = dict()
    if len(sss_ctc_block) > 0:
        sss_ctc_block = sss_ctc_block[0]
        for i_ent in range(sss_ctc_block['nent']):
            kind = sss_ctc_block['directory'][i_ent].kind
            pos = sss_ctc_block['directory'][i_ent].pos
            for key, id_, cast in zip(_sss_ctc_keys, _sss_ctc_ids,
                                      _sss_ctc_casters):
                if kind == id_:
                    tag = read_tag(fid, pos)
                    sss_ctc[key] = cast(tag.data)
                    break
            else:
                if kind == FIFF.FIFF_PROJ_ITEM_CH_NAME_LIST:
                    tag = read_tag(fid, pos)
                    chs = tag.data.split(':')
                    # This list can null chars in the last entry, e.g.:
                    # [..., u'MEG2642', u'MEG2643', u'MEG2641\x00 ... \x00']
                    chs[-1] = chs[-1].split('\x00')[0]
                    sss_ctc['proj_items_chs'] = chs

    sss_cal_block = dir_tree_find(tree, FIFF.FIFFB_SSS_CAL)  # 503
    sss_cal = dict()
    if len(sss_cal_block) > 0:
        sss_cal_block = sss_cal_block[0]
        for i_ent in range(sss_cal_block['nent']):
            kind = sss_cal_block['directory'][i_ent].kind
            pos = sss_cal_block['directory'][i_ent].pos
            for key, id_, cast in zip(_sss_cal_keys, _sss_cal_ids,
                                      _sss_cal_casters):
                if kind == id_:
                    tag = read_tag(fid, pos)
                    sss_cal[key] = cast(tag.data)
                    break

    max_info = dict(sss_info=sss_info, sss_ctc=sss_ctc,
                    sss_cal=sss_cal, max_st=max_st)
    return max_info


def _write_maxfilter_record(fid, record):
    """Write maxfilter processing record to file."""
    sss_info = record['sss_info']
    if len(sss_info) > 0:
        start_block(fid, FIFF.FIFFB_SSS_INFO)
        for key, id_, writer in zip(_sss_info_keys, _sss_info_ids,
                                    _sss_info_writers):
            if key in sss_info:
                writer(fid, id_, sss_info[key])
        end_block(fid, FIFF.FIFFB_SSS_INFO)

    max_st = record['max_st']
    if len(max_st) > 0:
        start_block(fid, FIFF.FIFFB_SSS_ST_INFO)
        for key, id_, writer in zip(_max_st_keys, _max_st_ids,
                                    _max_st_writers):
            if key in max_st:
                writer(fid, id_, max_st[key])
        end_block(fid, FIFF.FIFFB_SSS_ST_INFO)

    sss_ctc = record['sss_ctc']
    if len(sss_ctc) > 0:  # dict has entries
        start_block(fid, FIFF.FIFFB_CHANNEL_DECOUPLER)
        for key, id_, writer in zip(_sss_ctc_keys, _sss_ctc_ids,
                                    _sss_ctc_writers):
            if key in sss_ctc:
                writer(fid, id_, sss_ctc[key])
        if 'proj_items_chs' in sss_ctc:
            write_string(fid, FIFF.FIFF_PROJ_ITEM_CH_NAME_LIST,
                         ':'.join(sss_ctc['proj_items_chs']))
        end_block(fid, FIFF.FIFFB_CHANNEL_DECOUPLER)

    sss_cal = record['sss_cal']
    if len(sss_cal) > 0:
        start_block(fid, FIFF.FIFFB_SSS_CAL)
        for key, id_, writer in zip(_sss_cal_keys, _sss_cal_ids,
                                    _sss_cal_writers):
            if key in sss_cal:
                writer(fid, id_, sss_cal[key])
        end_block(fid, FIFF.FIFFB_SSS_CAL)
