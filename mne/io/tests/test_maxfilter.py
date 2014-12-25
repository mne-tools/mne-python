import numpy as np
import os.path as op
from mne import io
from mne.io.constants import FIFF
from mne.io.maxfilter import _get_sss_rank
from nose.tools import assert_true, assert_equal

base_dir = op.join(op.dirname(__file__), 'data')
raw_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')


def test_maxfilter_io():
    """test maxfilter io"""
    raw = io.Raw(raw_fname)
    sss = raw.info['sss']

    assert_true(sss['info']['frame'], FIFF.FIFFV_COORD_HEAD)
    # based on manual 2.0, rev. 5.0 page 23
    assert_true(5 <= sss['info']['in_order'] <= 11)
    assert_true(sss['info']['out_order'] <= 5)
    assert_true(sss['info']['nchan'] > len(sss['info']['components']))

    assert_equal(raw.ch_names[:sss['info']['nchan']],
                 sss['ctc']['proj_items_chs'])
    assert_equal(sss['ctc']['ctc'].shape,
                 (sss['info']['nchan'], sss['info']['nchan']))
    assert_equal(np.unique(np.diag(sss['ctc']['ctc'].toarray())),
                 np.array([1.], dtype=np.float32))

    assert_equal(sss['cal']['cal_coef'].shape, (306, 14))
    assert_equal(sss['cal']['cal_chans'].shape, (306, 2))
    vv_coils = [v for k, v in FIFF.items() if 'FIFFV_COIL_VV' in k]
    assert_true(all(k in vv_coils for k in set(sss['cal']['cal_chans'][:, 1])))


def test_maxfilter_get_rank():
    """test maxfilter rank lookup"""
    raw = io.Raw(raw_fname)
    sss = raw.info['sss']
    rank1 = sss['info']['nfree']
    rank2 = _get_sss_rank(sss)
    assert_equal(rank1, rank2)
