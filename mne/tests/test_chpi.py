# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import assert_raises, assert_equal, assert_true
import warnings

from mne.io import read_info, Raw
from mne.io.constants import FIFF
from mne.chpi import (_rot_to_quat, _quat_to_rot, get_chpi_positions,
                      _calculate_chpi_positions, _angle_between_quats)
from mne.utils import (run_tests_if_main, _TempDir, slow_test, catch_logging,
                       requires_version)
from mne.datasets import testing

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
test_fif_fname = op.join(base_dir, 'test_raw.fif')
ctf_fname = op.join(base_dir, 'test_ctf_raw.fif')
hp_fif_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')
hp_fname = op.join(base_dir, 'test_chpi_raw_hp.txt')

data_path = testing.data_path(download=False)
raw_fif_fname = op.join(data_path, 'SSS', 'test_move_anon_raw.fif')
pos_fname = op.join(data_path, 'SSS', 'test_move_anon_raw.pos')
sss_fif_fname = op.join(data_path, 'SSS', 'test_move_anon_raw_sss.fif')

warnings.simplefilter('always')


def test_quaternions():
    """Test quaternion calculations
    """
    rots = [np.eye(3)]
    for fname in [test_fif_fname, ctf_fname, hp_fif_fname]:
        rots += [read_info(fname)['dev_head_t']['trans'][:3, :3]]
    # nasty numerical cases
    rots += [np.array([
        [-0.99978541, -0.01873462, -0.00898756],
        [-0.01873462, 0.62565561, 0.77987608],
        [-0.00898756, 0.77987608, -0.62587152],
    ])]
    rots += [np.array([
        [0.62565561, -0.01873462, 0.77987608],
        [-0.01873462, -0.99978541, -0.00898756],
        [0.77987608, -0.00898756, -0.62587152],
    ])]
    rots += [np.array([
        [-0.99978541, -0.00898756, -0.01873462],
        [-0.00898756, -0.62587152, 0.77987608],
        [-0.01873462, 0.77987608, 0.62565561],
    ])]
    for rot in rots:
        assert_allclose(rot, _quat_to_rot(_rot_to_quat(rot)),
                        rtol=1e-5, atol=1e-5)
        rot = rot[np.newaxis, np.newaxis, :, :]
        assert_allclose(rot, _quat_to_rot(_rot_to_quat(rot)),
                        rtol=1e-5, atol=1e-5)

    # let's make sure our angle function works in some reasonable way
    for ii in range(3):
        for jj in range(3):
            a = np.zeros(3)
            b = np.zeros(3)
            a[ii] = 1.
            b[jj] = 1.
            expected = np.pi if ii != jj else 0.
            assert_allclose(_angle_between_quats(a, b), expected, atol=1e-5)


def test_get_chpi():
    """Test CHPI position computation
    """
    trans0, rot0, _, quat0 = get_chpi_positions(hp_fname, return_quat=True)
    assert_allclose(rot0[0], _quat_to_rot(quat0[0]))
    trans0, rot0 = trans0[:-1], rot0[:-1]
    raw = Raw(hp_fif_fname)
    out = get_chpi_positions(raw)
    trans1, rot1, t1 = out
    trans1, rot1 = trans1[2:], rot1[2:]
    # these will not be exact because they don't use equiv. time points
    assert_allclose(trans0, trans1, atol=1e-5, rtol=1e-1)
    assert_allclose(rot0, rot1, atol=1e-6, rtol=1e-1)
    # run through input checking
    assert_raises(TypeError, get_chpi_positions, 1)
    assert_raises(ValueError, get_chpi_positions, hp_fname, [1])
    raw_no_chpi = Raw(test_fif_fname)
    assert_raises(RuntimeError, get_chpi_positions, raw_no_chpi)
    assert_raises(ValueError, get_chpi_positions, raw, t_step='foo')
    assert_raises(IOError, get_chpi_positions, 'foo')


@testing.requires_testing_data
def test_hpi_info():
    """Test getting HPI info
    """
    tempdir = _TempDir()
    temp_name = op.join(tempdir, 'temp_raw.fif')
    for fname in (raw_fif_fname, sss_fif_fname):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            raw = Raw(fname, allow_maxshield=True)
        assert_true(len(raw.info['hpi_subsystem']) > 0)
        raw.save(temp_name, overwrite=True)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            raw_2 = Raw(temp_name, allow_maxshield=True)
        assert_equal(len(raw_2.info['hpi_subsystem']),
                     len(raw.info['hpi_subsystem']))


def _compare_positions(a, b, max_dist=0.003, max_angle=5.):
    """Compare estimated cHPI positions"""
    from scipy.interpolate import interp1d
    trans, rot, t = a
    trans_est, rot_est, t_est = b
    quats_est = _rot_to_quat(rot_est)

    # maxfilter produces some times that are implausibly large (weird)
    use_mask = (t >= t_est[0]) & (t <= t_est[-1])
    t = t[use_mask]
    trans = trans[use_mask]
    quats = _rot_to_quat(rot)
    quats = quats[use_mask]

    # double-check our angle function
    for q in (quats, quats_est):
        angles = _angle_between_quats(q, q)
        assert_allclose(angles, 0., atol=1e-5)

    # < 3 mm translation difference between MF and our estimation
    trans_est_interp = interp1d(t_est, trans_est, axis=0)(t)
    worst = np.sqrt(np.sum((trans - trans_est_interp) ** 2, axis=1)).max()
    assert_true(worst <= max_dist, '%0.1f > %0.1f mm'
                % (1000 * worst, 1000 * max_dist))

    # < 5 degrees rotation difference between MF and our estimation
    # (note that the interpolation will make this slightly worse)
    quats_est_interp = interp1d(t_est, quats_est, axis=0)(t)
    worst = 180 * _angle_between_quats(quats_est_interp, quats).max() / np.pi
    assert_true(worst <= max_angle, '%0.1f > %0.1f deg' % (worst, max_angle,))


@slow_test
@testing.requires_testing_data
@requires_version('scipy', '0.11')
@requires_version('numpy', '1.7')
def test_calculate_chpi_positions():
    """Test calculation of cHPI positions
    """
    trans, rot, t = get_chpi_positions(pos_fname)
    with warnings.catch_warnings(record=True):
        raw = Raw(raw_fif_fname, allow_maxshield=True, preload=True)
    t -= raw.first_samp / raw.info['sfreq']
    trans_est, rot_est, t_est = _calculate_chpi_positions(raw, verbose='debug')
    _compare_positions((trans, rot, t), (trans_est, rot_est, t_est))

    # degenerate conditions
    raw_no_chpi = Raw(test_fif_fname)
    assert_raises(RuntimeError, _calculate_chpi_positions, raw_no_chpi)
    raw_bad = raw.copy()
    for d in raw_bad.info['dig']:
        if d['kind'] == FIFF.FIFFV_POINT_HPI:
            d['coord_frame'] = 999
            break
    assert_raises(RuntimeError, _calculate_chpi_positions, raw_bad)
    raw_bad = raw.copy()
    for d in raw_bad.info['dig']:
        if d['kind'] == FIFF.FIFFV_POINT_HPI:
            d['r'] = np.ones(3)
    raw_bad.crop(0, 1., copy=False)
    with catch_logging() as log_file:
        _calculate_chpi_positions(raw_bad)
    for line in log_file.getvalue().split('\n')[:-1]:
        assert_true('0/5 acceptable' in line)

run_tests_if_main()
