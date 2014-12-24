from mne.io.maxfilter import read_maxfilter_info

import numpy as np
import os.path as op
from mne import io
from mne.io.constants import FIFF
from nose.tools import assert_true, assert_equal

base_dir = op.join(op.dirname(__file__), 'data')
raw_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')


def test_maxfilter_io():
    """test maxfilter io"""
    raw = io.Raw(raw_fname)
    sss_info, sss_ctc, sss_cal = read_maxfilter_info(raw_fname)

    assert_true(sss_info['frame'], FIFF.FIFFV_COORD_HEAD)
    assert_true(sss_info['in_order'] >= 5 and sss_info['in_order'] <= 11)
    assert_true(sss_info['out_order'] <= 5)
    assert_true(sss_info['nchan'] > len(sss_info['components']))

    assert_equal(raw.ch_names[:sss_info['nchan']], sss_ctc['proj_items_chs'])
    assert_equal(sss_ctc['ctc'].shape, (sss_info['nchan'], sss_info['nchan']))
    assert_equal(np.unique(np.diag(sss_ctc['ctc'].toarray())),
                 np.array([1.], dtype=np.float32))

    assert_equal(sss_cal['cal_coef'].shape, (306, 14))
    assert_equal(sss_cal['cal_chans'].shape, (306, 14))
    vv_coils = [v for k, v in FIFF.items() if 'FIFFV_COIL_VV' in k]
    assert_true(all(k in vv_coils for k in set(sss_cal['cal_chans'][:, 1])))
