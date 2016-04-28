# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import assert_raises, assert_equal, assert_true
import warnings

from mne.io import Raw
from mne.io.constants import FIFF
from mne.chpi import (get_chpi_positions, _calculate_chpi_positions,
                      head_pos_to_trans_rot_t, read_head_pos,
                      write_head_pos, filter_chpi)
from mne.fixes import assert_raises_regex
from mne.transforms import rot_to_quat, quat_to_rot, _angle_between_quats
from mne.utils import (run_tests_if_main, _TempDir, slow_test, catch_logging,
                       requires_version)
from mne.datasets import testing
from mne.tests.common import assert_meg_snr

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
test_fif_fname = op.join(base_dir, 'test_raw.fif')
ctf_fname = op.join(base_dir, 'test_ctf_raw.fif')
hp_fif_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')
hp_fname = op.join(base_dir, 'test_chpi_raw_hp.txt')

data_path = testing.data_path(download=False)
chpi_fif_fname = op.join(data_path, 'SSS', 'test_move_anon_raw.fif')
pos_fname = op.join(data_path, 'SSS', 'test_move_anon_raw.pos')
sss_fif_fname = op.join(data_path, 'SSS', 'test_move_anon_raw_sss.fif')
sss_hpisubt_fname = op.join(data_path, 'SSS', 'test_move_anon_hpisubt_raw.fif')

warnings.simplefilter('always')


@testing.requires_testing_data
def test_read_write_head_pos():
    """Test reading and writing head position quaternion parameters"""
    tempdir = _TempDir()
    temp_name = op.join(tempdir, 'temp.pos')
    # This isn't a 100% valid quat matrix but it should be okay for tests
    head_pos_rand = np.random.RandomState(0).randn(20, 10)
    # This one is valid
    head_pos_read = read_head_pos(pos_fname)
    for head_pos_orig in (head_pos_rand, head_pos_read):
        write_head_pos(temp_name, head_pos_orig)
        head_pos = read_head_pos(temp_name)
        assert_allclose(head_pos_orig, head_pos, atol=1e-3)
    # Degenerate cases
    assert_raises(TypeError, write_head_pos, 0, head_pos_read)  # not filename
    assert_raises(ValueError, write_head_pos, temp_name, 'foo')  # not array
    assert_raises(ValueError, write_head_pos, temp_name, head_pos_read[:, :9])
    assert_raises(TypeError, read_head_pos, 0)
    assert_raises(IOError, read_head_pos, temp_name + 'foo')


def test_get_chpi():
    """Test CHPI position computation
    """
    with warnings.catch_warnings(record=True):  # deprecation
        trans0, rot0, _, quat0 = get_chpi_positions(hp_fname, return_quat=True)
    assert_allclose(rot0[0], quat_to_rot(quat0[0]))
    trans0, rot0 = trans0[:-1], rot0[:-1]
    raw = Raw(hp_fif_fname)
    with warnings.catch_warnings(record=True):  # deprecation
        out = get_chpi_positions(raw)
    trans1, rot1, t1 = out
    trans1, rot1 = trans1[2:], rot1[2:]
    # these will not be exact because they don't use equiv. time points
    assert_allclose(trans0, trans1, atol=1e-5, rtol=1e-1)
    assert_allclose(rot0, rot1, atol=1e-6, rtol=1e-1)
    # run through input checking
    raw_no_chpi = Raw(test_fif_fname)
    with warnings.catch_warnings(record=True):  # deprecation
        assert_raises(TypeError, get_chpi_positions, 1)
        assert_raises(ValueError, get_chpi_positions, hp_fname, [1])
        assert_raises(RuntimeError, get_chpi_positions, raw_no_chpi)
        assert_raises(ValueError, get_chpi_positions, raw, t_step='foo')
        assert_raises(IOError, get_chpi_positions, 'foo')


@testing.requires_testing_data
def test_hpi_info():
    """Test getting HPI info
    """
    tempdir = _TempDir()
    temp_name = op.join(tempdir, 'temp_raw.fif')
    for fname in (chpi_fif_fname, sss_fif_fname):
        raw = Raw(fname, allow_maxshield='yes')
        assert_true(len(raw.info['hpi_subsystem']) > 0)
        raw.save(temp_name, overwrite=True)
        raw_2 = Raw(temp_name, allow_maxshield='yes')
        assert_equal(len(raw_2.info['hpi_subsystem']),
                     len(raw.info['hpi_subsystem']))


def _compare_positions(a, b, max_dist=0.003, max_angle=5.):
    """Compare estimated cHPI positions"""
    from scipy.interpolate import interp1d
    trans, rot, t = a
    trans_est, rot_est, t_est = b
    quats_est = rot_to_quat(rot_est)

    # maxfilter produces some times that are implausibly large (weird)
    use_mask = (t >= t_est[0]) & (t <= t_est[-1])
    t = t[use_mask]
    trans = trans[use_mask]
    quats = rot_to_quat(rot)
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
    trans, rot, t = head_pos_to_trans_rot_t(read_head_pos(pos_fname))
    raw = Raw(chpi_fif_fname, allow_maxshield='yes', preload=True)
    t -= raw.first_samp / raw.info['sfreq']
    quats = _calculate_chpi_positions(raw, verbose='debug')
    trans_est, rot_est, t_est = head_pos_to_trans_rot_t(quats)
    _compare_positions((trans, rot, t), (trans_est, rot_est, t_est), 0.003)

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
    with warnings.catch_warnings(record=True):  # bad pos
        with catch_logging() as log_file:
            _calculate_chpi_positions(raw_bad, verbose=True)
    # ignore HPI info header and [done] footer
    for line in log_file.getvalue().strip().split('\n')[4:-1]:
        assert_true('0/5 good' in line)

    # half the rate cuts off cHPI coils
    with warnings.catch_warnings(record=True):  # uint cast suggestion
        raw.resample(300., npad='auto')
    assert_raises_regex(RuntimeError, 'above the',
                        _calculate_chpi_positions, raw)


@testing.requires_testing_data
def test_chpi_subtraction():
    """Test subtraction of cHPI signals"""
    raw = Raw(chpi_fif_fname, allow_maxshield='yes', preload=True)
    with catch_logging() as log:
        filter_chpi(raw, include_line=False, verbose=True)
    assert_true('5 cHPI' in log.getvalue())
    # MaxFilter doesn't do quite as well as our algorithm with the last bit
    raw.crop(0, 16, copy=False)
    # remove cHPI status chans
    raw_c = Raw(sss_hpisubt_fname).crop(0, 16, copy=False).load_data()
    raw_c.pick_types(
        meg=True, eeg=True, eog=True, ecg=True, stim=True, misc=True)
    assert_meg_snr(raw, raw_c, 143, 624)

    # Degenerate cases
    raw_nohpi = Raw(test_fif_fname, preload=True)
    assert_raises(RuntimeError, filter_chpi, raw_nohpi)

    # When MaxFliter downsamples, like::
    #     $ maxfilter -nosss -ds 2 -f test_move_anon_raw.fif \
    #           -o test_move_anon_ds2_raw.fif
    # it can strip out some values of info, which we emulate here:
    raw = Raw(chpi_fif_fname, allow_maxshield='yes')
    with warnings.catch_warnings(record=True):  # uint cast suggestion
        raw = raw.crop(0, 1).load_data().resample(600., npad='auto')
    raw.info['buffer_size_sec'] = np.float64(2.)
    raw.info['lowpass'] = 200.
    del raw.info['maxshield']
    del raw.info['hpi_results'][0]['moments']
    del raw.info['hpi_subsystem']['event_channel']
    with catch_logging() as log:
        filter_chpi(raw, verbose=True)
    assert_true('2 cHPI' in log.getvalue())

run_tests_if_main()
