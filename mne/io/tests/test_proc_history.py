# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
# License: Simplified BSD

import numpy as np
import os.path as op
from mne import io
from mne.io.constants import FIFF
from mne.io.proc_history import _get_sss_rank
from nose.tools import assert_true, assert_equal

base_dir = op.join(op.dirname(__file__), 'data')
raw_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')


def test_maxfilter_io():
    """test maxfilter io"""
    raw = io.read_raw_fif(raw_fname)
    mf = raw.info['proc_history'][1]['max_info']

    assert_true(mf['sss_info']['frame'], FIFF.FIFFV_COORD_HEAD)
    # based on manual 2.0, rev. 5.0 page 23
    assert_true(5 <= mf['sss_info']['in_order'] <= 11)
    assert_true(mf['sss_info']['out_order'] <= 5)
    assert_true(mf['sss_info']['nchan'] > len(mf['sss_info']['components']))

    assert_equal(raw.ch_names[:mf['sss_info']['nchan']],
                 mf['sss_ctc']['proj_items_chs'])
    assert_equal(mf['sss_ctc']['decoupler'].shape,
                 (mf['sss_info']['nchan'], mf['sss_info']['nchan']))
    assert_equal(np.unique(np.diag(mf['sss_ctc']['decoupler'].toarray())),
                 np.array([1.], dtype=np.float32))

    assert_equal(mf['sss_cal']['cal_corrs'].shape, (306, 14))
    assert_equal(mf['sss_cal']['cal_chans'].shape, (306, 2))
    vv_coils = [v for k, v in FIFF.items() if 'FIFFV_COIL_VV' in k]
    assert_true(all(k in vv_coils
                    for k in set(mf['sss_cal']['cal_chans'][:, 1])))


def test_maxfilter_get_rank():
    """test maxfilter rank lookup"""
    raw = io.read_raw_fif(raw_fname)
    mf = raw.info['proc_history'][0]['max_info']
    rank1 = mf['sss_info']['nfree']
    rank2 = _get_sss_rank(mf)
    assert_equal(rank1, rank2)
