# Authors: Mark Wronkiewicz <wronk@uw.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
from numpy.testing import assert_allclose
import pytest

from mne import pick_types
from mne.io import read_raw_fif
from mne.datasets import testing
from mne.io.tag import _loc_to_coil_trans
from mne.preprocessing import (read_fine_calibration, write_fine_calibration,
                               compute_fine_calibration, maxwell_filter)
from mne.preprocessing.tests.test_maxwell import _assert_shielding
from mne.transforms import rot_to_quat, _angle_between_quats
from mne.utils import object_diff

# Define fine calibration filepaths
data_path = testing.data_path(download=False)
fine_cal_fname = op.join(data_path, 'SSS', 'sss_cal_3053.dat')
fine_cal_fname_3d = op.join(data_path, 'SSS', 'sss_cal_3053_3d.dat')
erm_fname = op.join(data_path, 'SSS', '141027_cropped_90Hz_raw.fif')
ctc = op.join(data_path, 'SSS', 'ct_sparse.fif')
cal_mf_fname = op.join(data_path, 'SSS', '141027.dat')


@pytest.mark.parametrize('fname', (cal_mf_fname, fine_cal_fname,
                                   fine_cal_fname_3d))
@testing.requires_testing_data
def test_fine_cal_io(tmpdir, fname):
    """Test round trip reading/writing of fine calibration .dat file."""
    temp_fname = op.join(str(tmpdir), 'fine_cal_temp.dat')
    # Load fine calibration file
    fine_cal_dict = read_fine_calibration(fname)

    # Save temp version of fine calibration file
    write_fine_calibration(temp_fname, fine_cal_dict)
    fine_cal_dict_reload = read_fine_calibration(temp_fname)

    # Load temp version of fine calibration file and compare hashes
    assert object_diff(fine_cal_dict, fine_cal_dict_reload) == ''


@pytest.mark.slowtest
@testing.requires_testing_data
def test_compute_fine_cal():
    """Test computing fine calibration coefficients."""
    raw = read_raw_fif(erm_fname)
    want_cal = read_fine_calibration(cal_mf_fname)
    got_cal, counts = compute_fine_calibration(
        raw, cross_talk=ctc, n_imbalance=1, verbose='debug')
    assert counts == 1
    assert set(got_cal.keys()) == set(want_cal.keys())
    assert got_cal['ch_names'] == want_cal['ch_names']
    # in practice these should never be exactly 1.
    assert sum([(ic == 1.).any() for ic in want_cal['imb_cals']]) == 0
    assert sum([(ic == 1.).any() for ic in got_cal['imb_cals']]) == 0

    got_imb = np.array(got_cal['imb_cals'], float)
    want_imb = np.array(want_cal['imb_cals'], float)
    assert got_imb.shape == want_imb.shape == (306, 1)
    got_imb, want_imb = got_imb[:, 0], want_imb[:, 0]

    orig_locs = np.array([ch['loc'] for ch in raw.info['chs'][:306]])
    want_locs = want_cal['locs']
    got_locs = got_cal['locs']
    assert want_locs.shape == got_locs.shape

    orig_trans = _loc_to_coil_trans(orig_locs)
    want_trans = _loc_to_coil_trans(want_locs)
    got_trans = _loc_to_coil_trans(got_locs)
    dist = np.linalg.norm(got_trans[:, :3, 3] - want_trans[:, :3, 3], axis=1)
    assert_allclose(dist, 0., atol=1e-6)
    dist = np.linalg.norm(got_trans[:, :3, 3] - orig_trans[:, :3, 3], axis=1)
    assert_allclose(dist, 0., atol=1e-6)
    orig_quat = rot_to_quat(orig_trans[:, :3, :3])
    want_quat = rot_to_quat(want_trans[:, :3, :3])
    got_quat = rot_to_quat(got_trans[:, :3, :3])
    want_orig_angles = np.rad2deg(_angle_between_quats(want_quat, orig_quat))
    got_want_angles = np.rad2deg(_angle_between_quats(got_quat, want_quat))
    got_orig_angles = np.rad2deg(_angle_between_quats(got_quat, orig_quat))
    for key in ('mag', 'grad'):
        # imb_cals value
        p = pick_types(raw.info, meg=key, exclude=())
        r2 = np.dot(got_imb[p], want_imb[p]) / (
            np.linalg.norm(want_imb[p]) * np.linalg.norm(got_imb[p]))
        assert 0.99 < r2 <= 1.00001, f'{key}: {r2:0.3f}'
        # rotation angles
        want_orig_max_angle = want_orig_angles[p].max()
        got_orig_max_angle = got_orig_angles[p].max()
        got_want_max_angle = got_want_angles[p].max()
        if key == 'mag':
            assert 8 < want_orig_max_angle < 11, want_orig_max_angle
            assert 1 < got_orig_max_angle < 2, got_orig_max_angle
            assert 9 < got_want_max_angle < 11, got_want_max_angle
        else:
            # Some of these angles are large, but mostly this has to do with
            # processing a very short (one 10-sec segment), downsampled (90 Hz)
            # file
            assert 66 < want_orig_max_angle < 68, want_orig_max_angle
            assert 67 < got_orig_max_angle < 107, got_orig_max_angle
            assert 53 < got_want_max_angle < 60, got_want_max_angle

    kwargs = dict(bad_condition='warning', cross_talk=ctc, coord_frame='meg')
    raw_sss = maxwell_filter(raw, **kwargs)
    raw_sss_mf = maxwell_filter(raw, calibration=cal_mf_fname, **kwargs)
    raw_sss_py = maxwell_filter(raw, calibration=got_cal, **kwargs)
    _assert_shielding(raw_sss, raw, 26, 27)
    _assert_shielding(raw_sss_mf, raw, 61, 63)
    _assert_shielding(raw_sss_py, raw, 61, 63)

    # redoing with given mag data should yield same result
    got_cal_redo, _ = compute_fine_calibration(
        raw, cross_talk=ctc, n_imbalance=1, calibration=got_cal,
        verbose='debug')
    assert got_cal['ch_names'] == got_cal_redo['ch_names']
    assert_allclose(got_cal['imb_cals'], got_cal_redo['imb_cals'], atol=5e-5)
    assert_allclose(got_cal['locs'], got_cal_redo['locs'], atol=1e-6)
    assert sum([(ic == 1.).any() for ic in got_cal['imb_cals']]) == 0

    # redoing with 3 imlabance parameters should improve the shielding factor
    grad_picks = pick_types(raw.info, meg='grad')
    assert len(grad_picks) == 204 and grad_picks[0] == 0
    got_grad_imbs = np.array(
        [got_cal['imb_cals'][pick] for pick in grad_picks])
    assert got_grad_imbs.shape == (204, 1)
    got_cal_3, _ = compute_fine_calibration(
        raw, cross_talk=ctc, n_imbalance=3, calibration=got_cal,
        verbose='debug')
    got_grad_3_imbs = np.array([
        got_cal_3['imb_cals'][pick] for pick in grad_picks])
    assert got_grad_3_imbs.shape == (204, 3)
    corr = np.corrcoef(got_grad_3_imbs[:, 0], got_grad_imbs[:, 0])[0, 1]
    assert 0.6 < corr < 0.7
    raw_sss_py = maxwell_filter(raw, calibration=got_cal_3, **kwargs)
    _assert_shielding(raw_sss_py, raw, 68, 70)
