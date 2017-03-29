"""Data and Channel Location Equivalence Tests"""
from __future__ import print_function

# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import inspect
import os.path as op
import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import (assert_equal, assert_almost_equal, assert_raises,
                        assert_true)
from scipy import linalg
import scipy.io

from mne import pick_types, Epochs, find_events, read_events
from mne.transforms import apply_trans
from mne.tests.common import assert_dig_allclose
from mne.utils import run_tests_if_main, _TempDir
from mne.io import read_raw_fif, read_raw_kit, read_epochs_kit
from mne.io.kit.coreg import read_sns
from mne.io.kit.constants import KIT, KIT_CONSTANTS, KIT_NY, KIT_UMD_2014
from mne.io.tests.test_raw import _test_raw_reader
from mne.surface import _get_ico_surface

FILE = inspect.getfile(inspect.currentframe())
parent_dir = op.dirname(op.abspath(FILE))
data_dir = op.join(parent_dir, 'data')
sqd_path = op.join(data_dir, 'test.sqd')
sqd_umd_path = op.join(data_dir, 'test_umd-raw.sqd')
epochs_path = op.join(data_dir, 'test-epoch.raw')
events_path = op.join(data_dir, 'test-eve.txt')
mrk_path = op.join(data_dir, 'test_mrk.sqd')
mrk2_path = op.join(data_dir, 'test_mrk_pre.sqd')
mrk3_path = op.join(data_dir, 'test_mrk_post.sqd')
elp_txt_path = op.join(data_dir, 'test_elp.txt')
hsp_txt_path = op.join(data_dir, 'test_hsp.txt')
elp_path = op.join(data_dir, 'test.elp')
hsp_path = op.join(data_dir, 'test.hsp')


def test_data():
    """Test reading raw kit files."""
    assert_raises(TypeError, read_raw_kit, epochs_path)
    assert_raises(TypeError, read_epochs_kit, sqd_path)
    assert_raises(ValueError, read_raw_kit, sqd_path, mrk_path, elp_txt_path)
    assert_raises(ValueError, read_raw_kit, sqd_path, None, None, None,
                  list(range(200, 190, -1)))
    assert_raises(ValueError, read_raw_kit, sqd_path, None, None, None,
                  list(range(167, 159, -1)), '*', 1, True)
    # check functionality
    raw_mrk = read_raw_kit(sqd_path, [mrk2_path, mrk3_path], elp_txt_path,
                           hsp_txt_path)
    raw_py = _test_raw_reader(read_raw_kit, input_fname=sqd_path, mrk=mrk_path,
                              elp=elp_txt_path, hsp=hsp_txt_path,
                              stim=list(range(167, 159, -1)), slope='+',
                              stimthresh=1)
    assert_true('RawKIT' in repr(raw_py))
    assert_equal(raw_mrk.info['kit_system_id'], KIT.SYSTEM_NYU_2010)
    assert_true(KIT_CONSTANTS[raw_mrk.info['kit_system_id']] is KIT_NY)

    # Test stim channel
    raw_stim = read_raw_kit(sqd_path, mrk_path, elp_txt_path, hsp_txt_path,
                            stim='<', preload=False)
    for raw in [raw_py, raw_stim, raw_mrk]:
        stim_pick = pick_types(raw.info, meg=False, ref_meg=False,
                               stim=True, exclude='bads')
        stim1, _ = raw[stim_pick]
        stim2 = np.array(raw.read_stim_ch(), ndmin=2)
        assert_array_equal(stim1, stim2)

    # Binary file only stores the sensor channels
    py_picks = pick_types(raw_py.info, exclude='bads')
    raw_bin = op.join(data_dir, 'test_bin_raw.fif')
    raw_bin = read_raw_fif(raw_bin, preload=True)
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

    # KIT-UMD data
    _test_raw_reader(read_raw_kit, input_fname=sqd_umd_path)
    raw = read_raw_kit(sqd_umd_path)
    assert_equal(raw.info['kit_system_id'], KIT.SYSTEM_UMD_2014_12)
    assert_true(KIT_CONSTANTS[raw.info['kit_system_id']] is KIT_UMD_2014)


def test_epochs():
    """Test reading epoched SQD file."""
    raw = read_raw_kit(sqd_path, stim=None)
    events = read_events(events_path)
    raw_epochs = Epochs(raw, events, None, tmin=0, tmax=.099, baseline=None)
    data1 = raw_epochs.get_data()
    epochs = read_epochs_kit(epochs_path, events_path)
    data11 = epochs.get_data()
    assert_array_equal(data1, data11)


def test_raw_events():
    """Test creating stim channel from raw SQD file."""
    def evts(a, b, c, d, e, f=None):
        out = [[269, a, b], [281, b, c], [1552, c, d], [1564, d, e]]
        if f is not None:
            out.append([2000, e, f])
        return out

    raw = read_raw_kit(sqd_path)
    assert_array_equal(find_events(raw, output='step', consecutive=True),
                       evts(255, 254, 255, 254, 255, 0))

    raw = read_raw_kit(sqd_path, slope='+')
    assert_array_equal(find_events(raw, output='step', consecutive=True),
                       evts(0, 1, 0, 1, 0))

    raw = read_raw_kit(sqd_path, stim='<', slope='+')
    assert_array_equal(find_events(raw, output='step', consecutive=True),
                       evts(0, 128, 0, 128, 0))

    raw = read_raw_kit(sqd_path, stim='<', slope='+', stim_code='channel')
    assert_array_equal(find_events(raw, output='step', consecutive=True),
                       evts(0, 160, 0, 160, 0))

    raw = read_raw_kit(sqd_path, stim=range(160, 162), slope='+',
                       stim_code='channel')
    assert_array_equal(find_events(raw, output='step', consecutive=True),
                       evts(0, 160, 0, 160, 0))


def test_ch_loc():
    """Test raw kit loc."""
    raw_py = read_raw_kit(sqd_path, mrk_path, elp_txt_path, hsp_txt_path,
                          stim='<')
    raw_bin = read_raw_fif(op.join(data_dir, 'test_bin_raw.fif'))

    ch_py = raw_py._raw_extras[0]['sensor_locs'][:, :5]
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

    # test when more than one marker file provided
    mrks = [mrk_path, mrk2_path, mrk3_path]
    read_raw_kit(sqd_path, mrks, elp_txt_path, hsp_txt_path, preload=False)
    # this dataset does not have the equivalent set of points :(
    raw_bin.info['dig'] = raw_bin.info['dig'][:8]
    raw_py.info['dig'] = raw_py.info['dig'][:8]
    assert_dig_allclose(raw_py.info, raw_bin.info)


def test_hsp_elp():
    """Test KIT usage of *.elp and *.hsp files against *.txt files."""
    raw_txt = read_raw_kit(sqd_path, mrk_path, elp_txt_path, hsp_txt_path)
    raw_elp = read_raw_kit(sqd_path, mrk_path, elp_path, hsp_path)

    # head points
    pts_txt = np.array([dig_point['r'] for dig_point in raw_txt.info['dig']])
    pts_elp = np.array([dig_point['r'] for dig_point in raw_elp.info['dig']])
    assert_array_almost_equal(pts_elp, pts_txt, decimal=5)

    # transforms
    trans_txt = raw_txt.info['dev_head_t']['trans']
    trans_elp = raw_elp.info['dev_head_t']['trans']
    assert_array_almost_equal(trans_elp, trans_txt, decimal=5)

    # head points in device space
    pts_txt_in_dev = apply_trans(linalg.inv(trans_txt), pts_txt)
    pts_elp_in_dev = apply_trans(linalg.inv(trans_elp), pts_elp)
    assert_array_almost_equal(pts_elp_in_dev, pts_txt_in_dev, decimal=5)


def test_decimate():
    """Test decimation of digitizer headshapes with too many points."""
    # load headshape and convert to meters
    hsp_mm = _get_ico_surface(5)['rr'] * 100
    hsp_m = hsp_mm / 1000.

    # save headshape to a file in mm in temporary directory
    tempdir = _TempDir()
    sphere_hsp_path = op.join(tempdir, 'test_sphere.txt')
    np.savetxt(sphere_hsp_path, hsp_mm)

    # read in raw data using spherical hsp, and extract new hsp
    with warnings.catch_warnings(record=True) as w:
        raw = read_raw_kit(sqd_path, mrk_path, elp_txt_path, sphere_hsp_path)
    assert_true(any('more than' in str(ww.message) for ww in w))
    # collect headshape from raw (should now be in m)
    hsp_dec = np.array([dig['r'] for dig in raw.info['dig']])[8:]

    # with 10242 points and _decimate_points set to resolution of 5 mm, hsp_dec
    # should be a bit over 5000 points. If not, something is wrong or
    # decimation resolution has been purposefully changed
    assert_true(len(hsp_dec) > 5000)

    # should have similar size, distance from center
    dist = np.sqrt(np.sum((hsp_m - np.mean(hsp_m, axis=0))**2, axis=1))
    dist_dec = np.sqrt(np.sum((hsp_dec - np.mean(hsp_dec, axis=0))**2, axis=1))
    hsp_rad = np.mean(dist)
    hsp_dec_rad = np.mean(dist_dec)
    assert_almost_equal(hsp_rad, hsp_dec_rad, places=3)

run_tests_if_main()
