from .open import fiff_open
from .open import read_tag
from .tree import dir_tree_find
from .constants import FIFF

import numpy as np


def _read_maxfilter_info(fname):
    """Read maxfilter processing information from fiff file

    This function reads the SSS info, the CTC correction and the
    calibaraions from the SSS processing logs inside af a raw file
    (C.f. Maxfilter v2.0 manual (5th revision), page 19):

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
        105 = }             502 = SSS info
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

    Paramters
    ---------
    fname : str | fid
        The Neuromag recordings. Filenames should end
        with raw.fif, raw.fif.gz , raw_sss.fif, raw_sss.fif.gz,
        raw_tsss.fif or raw_tsss.fif.gz.

    Returns
    -------
    sss_info : dict
        The sss processing info. Will be empty if block is not present.
    sss_ctc : dict
        The sss CTC info. Will be empty if block is not present.
    sss_cal : dict
        The sss calibration info. Will be empty if block is not present.
    """
    fid, tree, directory = fiff_open(fname)

    sss_ctc_block = dir_tree_find(tree, FIFF.FIFFB_SSS_CTC)  # 501
    sss_ctc = {}
    if len(sss_ctc_block) > 0:
        sss_ctc_block = sss_ctc_block[0]
        for i_ent in range(sss_ctc_block['nent']):
            kind = sss_ctc_block['directory'][i_ent].kind
            pos = sss_ctc_block['directory'][i_ent].pos
            if kind == FIFF.FIFF_SSS_CTC_BLOCK_ID:
                tag = read_tag(fid, pos)
                sss_ctc['cal_chans'] = tag.data
            elif kind == FIFF.FIFF_SSS_CTC_DATE:
                tag = read_tag(fid, pos)
                sss_ctc['date'] = tag.data
            elif kind == FIFF.FIFF_SSS_CTC_CREATOR_PROGRAM:
                tag = read_tag(fid, pos)
                sss_ctc['creator'] = tag.data
            elif kind == FIFF.FIFF_SSS_CTC_CTC_MAT:
                tag = read_tag(fid, pos)
                sss_ctc['ctc'] = tag.data
            elif kind == FIFF.FIFF_SSS_CTC_PROJ_ITEM_CHS:
                tag = read_tag(fid, pos)
                sss_ctc['proj_items_chs'] = tag.data.split(':')

    sss_info_block = dir_tree_find(tree, FIFF.FIFFB_SSS_INFO)  # 502
    sss_info = dict()
    if len(sss_info_block) > 0:
        sss_info_block = sss_info_block[0]
        for i_ent in range(sss_info_block['nent']):
            kind = sss_info_block['directory'][i_ent].kind
            pos = sss_info_block['directory'][i_ent].pos
            if kind == FIFF.FIFF_SSS_INFO_TASK:
                tag = read_tag(fid, pos)
                sss_info['task'] = int(tag.data)
            elif kind == FIFF.FIFF_SSS_INFO_COORD_FRAME:
                tag = read_tag(fid, pos)
                sss_info['frame'] = int(tag.data)
            elif kind == FIFF.FIFF_SSS_INFO_ORIGIN:
                tag = read_tag(fid, pos)
                sss_info['origin'] = np.array(tag.data)
            elif kind == FIFF.FIFF_SSS_INFO_IN_ORDER:
                tag = read_tag(fid, pos)
                sss_info['in_order'] = int(tag.data)
            elif kind == FIFF.FIFF_SSS_INFO_OUT_ORDER:
                tag = read_tag(fid, pos)
                sss_info['out_order'] = int(tag.data)
            elif kind == FIFF.FIFF_SSS_INFO_NCHAN:
                tag = read_tag(fid, pos)
                sss_info['nchan'] = int(tag.data)
            elif kind == FIFF.FIFF_SSS_INFO_COMPONENTS:
                tag = read_tag(fid, pos)
                sss_info['components'] = np.array(tag.data)

    sss_cal_block = dir_tree_find(tree, FIFF.FIFFB_SSS_CAL_ADJUST)  # 503
    sss_cal = {}
    if len(sss_info_block) > 0:
        sss_cal_block = sss_cal_block[0]
        for i_ent in range(sss_cal_block['nent']):
            kind = sss_cal_block['directory'][i_ent].kind
            pos = sss_cal_block['directory'][i_ent].pos
            if kind == FIFF.FIFF_SSS_CAL_CHNLS:
                tag = read_tag(fid, pos)
                sss_cal['cal_chans'] = np.array(tag.data)
            elif kind == FIFF.FIFF_SSS_CAL_COEFF:
                tag = read_tag(fid, pos)
                sss_cal['cal_coef'] = np.array(tag.data)

    # return ordered of relevance
    return sss_info, sss_ctc, sss_cal
