"""Data and Channel Location Equivalence Tests"""
from __future__ import print_function

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import os.path as op
import inspect
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import scipy.io
from mne.utils import _TempDir
from mne import pick_types
from mne.io import Raw
from mne.io import read_raw_kit
from mne.io.kit.coreg import read_sns

FILE = inspect.getfile(inspect.currentframe())
parent_dir = op.dirname(op.abspath(FILE))
data_dir = op.join(parent_dir, 'data')
sqd_path = op.join(data_dir, 'test.sqd')
mrk_path = op.join(data_dir, 'test_mrk.sqd')
mrk2_path = op.join(data_dir, 'test_mrk_pre.sqd')
mrk3_path = op.join(data_dir, 'test_mrk_post.sqd')
elp_path = op.join(data_dir, 'test_elp.txt')
hsp_path = op.join(data_dir, 'test_hsp.txt')

tempdir = _TempDir()


def test_data():
    """Test reading raw kit files
    """
    raw_py = read_raw_kit(sqd_path, mrk_path, elp_path, hsp_path,
                          stim=list(range(167, 159, -1)), slope='+',
                          stimthresh=1, preload=True)
    print(repr(raw_py))

    # Binary file only stores the sensor channels
    py_picks = pick_types(raw_py.info, exclude='bads')
    raw_bin = op.join(data_dir, 'test_bin_raw.fif')
    raw_bin = Raw(raw_bin, preload=True)
    bin_picks = pick_types(raw_bin.info, stim=True, exclude='bads')
    data_bin, _ = raw_bin[bin_picks]
    data_py, _ = raw_py[py_picks]

    # this .mat was generated using the Yokogawa MEG Reader
    data_Ykgw = op.join(data_dir, 'test_Ykgw.mat')
    data_Ykgw = scipy.io.loadmat(data_Ykgw)['data']
    data_Ykgw = data_Ykgw[py_picks]

    assert_array_almost_equal(data_py, data_Ykgw)

    py_picks = pick_types(raw_py.info, stim=True, ref_meg=False,
                          exclude='bads')
    data_py, _ = raw_py[py_picks]
    assert_array_almost_equal(data_py, data_bin)


def test_read_segment():
    """Test writing raw kit files when preload is False
    """
    raw1 = read_raw_kit(sqd_path, mrk_path, elp_path, hsp_path, stim='<',
                        preload=False)
    raw1_file = op.join(tempdir, 'test1-raw.fif')
    raw1.save(raw1_file, buffer_size_sec=.1, overwrite=True)
    raw2 = read_raw_kit(sqd_path, mrk_path, elp_path, hsp_path, stim='<',
                        preload=True)
    raw2_file = op.join(tempdir, 'test2-raw.fif')
    raw2.save(raw2_file, buffer_size_sec=.1, overwrite=True)
    raw1 = Raw(raw1_file, preload=True)
    raw2 = Raw(raw2_file, preload=True)
    assert_array_equal(raw1._data, raw2._data)
    raw3 = read_raw_kit(sqd_path, mrk_path, elp_path, hsp_path, stim='<',
                        preload=True)
    assert_array_almost_equal(raw1._data, raw3._data)


def test_ch_loc():
    """Test raw kit loc
    """
    raw_py = read_raw_kit(sqd_path, mrk_path, elp_path, hsp_path, stim='<')
    raw_bin = Raw(op.join(data_dir, 'test_bin_raw.fif'))

    ch_py = raw_py._sqd_params['sensor_locs'][:, :5]
    # ch locs stored as m, not mm
    ch_py[:, :3] *= 1e3
    ch_sns = read_sns(op.join(data_dir, 'sns.txt'))
    assert_array_almost_equal(ch_py, ch_sns, 2)

    assert_array_almost_equal(raw_py.info['dev_head_t']['trans'],
                              raw_bin.info['dev_head_t']['trans'], 4)
    for py_ch, bin_ch in zip(raw_py.info['chs'], raw_bin.info['chs']):
        if bin_ch['ch_name'].startswith('MEG'):
            # the stored ch locs have more precision than the sns.txt
            assert_array_almost_equal(py_ch['loc'], bin_ch['loc'], decimal=2)
            assert_array_almost_equal(py_ch['coil_trans'],
                                      bin_ch['coil_trans'],
                                      decimal=2)

    # test when more than one marker file provided
    mrks = [mrk_path, mrk2_path, mrk3_path]
    _ = read_raw_kit(sqd_path, mrks, elp_path, hsp_path, preload=False)


def test_stim_ch():
    """Test raw kit stim ch
    """
    raw = read_raw_kit(sqd_path, mrk_path, elp_path, hsp_path, stim='<',
                       slope='+', preload=True)
    stim_pick = pick_types(raw.info, meg=False, ref_meg=False,
                           stim=True, exclude='bads')
    stim1, _ = raw[stim_pick]
    stim2 = np.array(raw.read_stim_ch(), ndmin=2)
    assert_array_equal(stim1, stim2)
