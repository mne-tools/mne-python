# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import assert_raises, assert_equal, assert_true
import warnings

from mne.io import read_raw_fif
from mne.io.constants import FIFF
from mne.chpi import (_calculate_chpi_positions,
                      head_pos_to_trans_rot_t, read_head_pos,
                      write_head_pos, filter_chpi, _get_hpi_info)
from mne.fixes import assert_raises_regex
from mne.transforms import rot_to_quat, _angle_between_quats
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
def test_chpi_adjust():
    """Test cHPI logging and adjustment."""
    raw = read_raw_fif(chpi_fif_fname, allow_maxshield='yes',
                       add_eeg_ref=False)
    with catch_logging() as log:
        _get_hpi_info(raw.info, adjust=True, verbose='debug')

    # Ran MaxFilter (with -list, -v, -movecomp, etc.), and got:
    msg = ['HPIFIT: 5 coils digitized in order 5 1 4 3 2',
           'HPIFIT: 3 coils accepted: 1 2 4',
           'Hpi coil moments (3 5):',
           '2.08542e-15 -1.52486e-15 -1.53484e-15',
           '2.14516e-15 2.09608e-15 7.30303e-16',
           '-3.2318e-16 -4.25666e-16 2.69997e-15',
           '5.21717e-16 1.28406e-15 1.95335e-15',
           '1.21199e-15 -1.25801e-19 1.18321e-15',
           'HPIFIT errors:  0.3, 0.3, 5.3, 0.4, 3.2 mm.',
           'HPI consistency of isotrak and hpifit is OK.',
           'HP fitting limits: err = 5.0 mm, gval = 0.980.',
           'Using 5 HPI coils: 83 143 203 263 323 Hz',  # actually came earlier
           ]

    log = log.getvalue().splitlines()
    assert_true(set(log) == set(msg), '\n' + '\n'.join(set(msg) - set(log)))

    # Then took the raw file, did this:
    raw.info['dig'][5]['r'][2] += 1.
    # And checked the result in MaxFilter, which changed the logging as:
    msg = msg[:8] + [
        'HPIFIT errors:  0.3, 0.3, 5.3, 999.7, 3.2 mm.',
        'Note: HPI coil 3 isotrak is adjusted by 5.3 mm!',
        'Note: HPI coil 5 isotrak is adjusted by 3.2 mm!'] + msg[-2:]
    with catch_logging() as log:
        _get_hpi_info(raw.info, adjust=True, verbose='debug')
    log = log.getvalue().splitlines()
    assert_true(set(log) == set(msg), '\n' + '\n'.join(set(msg) - set(log)))


@testing.requires_testing_data
def test_read_write_head_pos():
    """Test reading and writing head position quaternion parameters."""
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


@testing.requires_testing_data
def test_hpi_info():
    """Test getting HPI info."""
    tempdir = _TempDir()
    temp_name = op.join(tempdir, 'temp_raw.fif')
    for fname in (chpi_fif_fname, sss_fif_fname):
        raw = read_raw_fif(fname, allow_maxshield='yes', add_eeg_ref=False)
        assert_true(len(raw.info['hpi_subsystem']) > 0)
        raw.save(temp_name, overwrite=True)
        raw_2 = read_raw_fif(temp_name, allow_maxshield='yes',
                             add_eeg_ref=False)
        assert_equal(len(raw_2.info['hpi_subsystem']),
                     len(raw.info['hpi_subsystem']))


def _compare_positions(a, b, max_dist=0.003, max_angle=5.):
    """Compare estimated cHPI positions."""
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
    """Test calculation of cHPI positions."""
    trans, rot, t = head_pos_to_trans_rot_t(read_head_pos(pos_fname))
    raw = read_raw_fif(chpi_fif_fname, allow_maxshield='yes', preload=True,
                       add_eeg_ref=False)
    t -= raw.first_samp / raw.info['sfreq']
    quats = _calculate_chpi_positions(raw, verbose='debug')
    trans_est, rot_est, t_est = head_pos_to_trans_rot_t(quats)
    _compare_positions((trans, rot, t), (trans_est, rot_est, t_est), 0.003)

    # degenerate conditions
    raw_no_chpi = read_raw_fif(test_fif_fname, add_eeg_ref=False)
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
    assert_true('0/5 good' in log_file.getvalue().strip().split('\n')[-2])

    # half the rate cuts off cHPI coils
    with warnings.catch_warnings(record=True):  # uint cast suggestion
        raw.resample(300., npad='auto')
    assert_raises_regex(RuntimeError, 'above the',
                        _calculate_chpi_positions, raw)


@testing.requires_testing_data
def test_chpi_subtraction():
    """Test subtraction of cHPI signals."""
    raw = read_raw_fif(chpi_fif_fname, allow_maxshield='yes', preload=True,
                       add_eeg_ref=False)
    raw.info['bads'] = ['MEG0111']
    with catch_logging() as log:
        filter_chpi(raw, include_line=False, verbose=True)
    assert_true('5 cHPI' in log.getvalue())
    # MaxFilter doesn't do quite as well as our algorithm with the last bit
    raw.crop(0, 16, copy=False)
    # remove cHPI status chans
    raw_c = read_raw_fif(sss_hpisubt_fname,
                         add_eeg_ref=False).crop(0, 16, copy=False).load_data()
    raw_c.pick_types(
        meg=True, eeg=True, eog=True, ecg=True, stim=True, misc=True)
    assert_meg_snr(raw, raw_c, 143, 624)

    # Degenerate cases
    raw_nohpi = read_raw_fif(test_fif_fname, preload=True, add_eeg_ref=False)
    assert_raises(RuntimeError, filter_chpi, raw_nohpi)

    # When MaxFliter downsamples, like::
    #     $ maxfilter -nosss -ds 2 -f test_move_anon_raw.fif \
    #           -o test_move_anon_ds2_raw.fif
    # it can strip out some values of info, which we emulate here:
    raw = read_raw_fif(chpi_fif_fname, allow_maxshield='yes',
                       add_eeg_ref=False)
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
