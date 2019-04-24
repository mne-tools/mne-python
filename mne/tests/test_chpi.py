# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d
import pytest

from mne import pick_types, pick_info
from mne.io import (read_raw_fif, read_raw_artemis123, read_raw_ctf, read_info,
                    RawArray)
from mne.io.constants import FIFF
from mne.chpi import (_calculate_chpi_positions, _calculate_chpi_coil_locs,
                      _calculate_head_pos_ctf, head_pos_to_trans_rot_t,
                      read_head_pos, write_head_pos, filter_chpi,
                      _get_hpi_info, _get_hpi_initial_fit)
from mne.transforms import rot_to_quat, _angle_between_quats
from mne.simulation import add_chpi
from mne.utils import run_tests_if_main, catch_logging, assert_meg_snr
from mne.datasets import testing

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
test_fif_fname = op.join(base_dir, 'test_raw.fif')
ctf_fname = op.join(base_dir, 'test_ctf_raw.fif')
hp_fif_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')
hp_fname = op.join(base_dir, 'test_chpi_raw_hp.txt')
raw_fname = op.join(base_dir, 'test_raw.fif')

data_path = testing.data_path(download=False)
chpi_fif_fname = op.join(data_path, 'SSS', 'test_move_anon_raw.fif')
pos_fname = op.join(data_path, 'SSS', 'test_move_anon_raw.pos')
sss_fif_fname = op.join(data_path, 'SSS', 'test_move_anon_raw_sss.fif')
sss_hpisubt_fname = op.join(data_path, 'SSS', 'test_move_anon_hpisubt_raw.fif')
chpi5_fif_fname = op.join(data_path, 'SSS', 'chpi5_raw.fif')
chpi5_pos_fname = op.join(data_path, 'SSS', 'chpi5_raw_mc.pos')
ctf_chpi_fname = op.join(data_path, 'CTF', 'testdata_ctf_mc.ds')
ctf_chpi_pos_fname = op.join(data_path, 'CTF', 'testdata_ctf_mc.pos')

art_fname = op.join(data_path, 'ARTEMIS123', 'Artemis_Data_2017-04-04' +
                    '-15h-44m-22s_Motion_Translation-z.bin')
art_mc_fname = op.join(data_path, 'ARTEMIS123', 'Artemis_Data_2017-04-04' +
                       '-15h-44m-22s_Motion_Translation-z_mc.pos')


@testing.requires_testing_data
def test_chpi_adjust():
    """Test cHPI logging and adjustment."""
    raw = read_raw_fif(chpi_fif_fname, allow_maxshield='yes')
    with catch_logging() as log:
        _get_hpi_initial_fit(raw.info, adjust=True, verbose='debug')
        _get_hpi_info(raw.info, verbose='debug')
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
    assert set(log) == set(msg), '\n' + '\n'.join(set(msg) - set(log))

    # Then took the raw file, did this:
    raw.info['dig'][5]['r'][2] += 1.
    # And checked the result in MaxFilter, which changed the logging as:
    msg = msg[:8] + [
        'HPIFIT errors:  0.3, 0.3, 5.3, 999.7, 3.2 mm.',
        'Note: HPI coil 3 isotrak is adjusted by 5.3 mm!',
        'Note: HPI coil 5 isotrak is adjusted by 3.2 mm!'] + msg[-2:]
    with catch_logging() as log:
        _get_hpi_initial_fit(raw.info, adjust=True, verbose='debug')
        _get_hpi_info(raw.info, verbose='debug')
    log = log.getvalue().splitlines()
    assert set(log) == set(msg), '\n' + '\n'.join(set(msg) - set(log))


@testing.requires_testing_data
def test_read_write_head_pos(tmpdir):
    """Test reading and writing head position quaternion parameters."""
    temp_name = op.join(str(tmpdir), 'temp.pos')
    # This isn't a 100% valid quat matrix but it should be okay for tests
    head_pos_rand = np.random.RandomState(0).randn(20, 10)
    # This one is valid
    head_pos_read = read_head_pos(pos_fname)
    for head_pos_orig in (head_pos_rand, head_pos_read):
        write_head_pos(temp_name, head_pos_orig)
        head_pos = read_head_pos(temp_name)
        assert_allclose(head_pos_orig, head_pos, atol=1e-3)
    # Degenerate cases
    pytest.raises(TypeError, write_head_pos, 0, head_pos_read)  # not filename
    pytest.raises(ValueError, write_head_pos, temp_name, 'foo')  # not array
    pytest.raises(ValueError, write_head_pos, temp_name, head_pos_read[:, :9])
    pytest.raises(TypeError, read_head_pos, 0)
    pytest.raises(IOError, read_head_pos, temp_name + 'foo')


@testing.requires_testing_data
def test_hpi_info(tmpdir):
    """Test getting HPI info."""
    temp_name = op.join(str(tmpdir), 'temp_raw.fif')
    for fname in (chpi_fif_fname, sss_fif_fname):
        raw = read_raw_fif(fname, allow_maxshield='yes').crop(0, 0.1)
        assert len(raw.info['hpi_subsystem']) > 0
        raw.save(temp_name, overwrite=True)
        info = read_info(temp_name)
        assert len(info['hpi_subsystem']) == len(raw.info['hpi_subsystem'])


def _assert_quats(actual, desired, dist_tol=0.003, angle_tol=5.):
    """Compare estimated cHPI positions."""
    trans_est, rot_est, t_est = head_pos_to_trans_rot_t(actual)
    trans, rot, t = head_pos_to_trans_rot_t(desired)
    quats_est = rot_to_quat(rot_est)

    # maxfilter produces some times that are implausibly large (weird)
    if not np.isclose(t[0], t_est[0], atol=1e-1):  # within 100 ms
        raise AssertionError('Start times not within 100 ms: %0.3f != %0.3f'
                             % (t[0], t_est[0]))
    use_mask = (t >= t_est[0]) & (t <= t_est[-1])
    t = t[use_mask]
    trans = trans[use_mask]
    quats = rot_to_quat(rot)
    quats = quats[use_mask]

    # double-check our angle function
    for q in (quats, quats_est):
        angles = _angle_between_quats(q, q)
        assert_allclose(angles, 0., atol=1e-5)

    # limit translation difference between MF and our estimation
    trans_est_interp = interp1d(t_est, trans_est, axis=0)(t)
    distances = np.sqrt(np.sum((trans - trans_est_interp) ** 2, axis=1))
    arg_worst = np.argmax(distances)
    assert distances[arg_worst] <= dist_tol, (
        '@ %0.3f seconds: %0.3f > %0.3f mm'
        % (t[arg_worst], 1000 * distances[arg_worst], 1000 * dist_tol))

    # limit rotation difference between MF and our estimation
    # (note that the interpolation will make this slightly worse)
    quats_est_interp = interp1d(t_est, quats_est, axis=0)(t)
    angles = 180 * _angle_between_quats(quats_est_interp, quats) / np.pi
    arg_worst = np.argmax(angles)
    assert angles[arg_worst] <= angle_tol, (
        '@ %0.3f seconds: %0.3f > %0.3f deg'
        % (t[arg_worst], angles[arg_worst], angle_tol))


def _decimate_chpi(raw, decim=4):
    """Decimate raw data (with aliasing) in cHPI-fitting compatible way."""
    raw_dec = RawArray(
        raw._data[:, ::decim], raw.info, first_samp=raw.first_samp // decim)
    raw_dec.info['sfreq'] /= decim
    for coil in raw_dec.info['hpi_meas'][0]['hpi_coils']:
        if coil['coil_freq'] > raw_dec.info['sfreq']:
            coil['coil_freq'] = np.mod(coil['coil_freq'],
                                       raw_dec.info['sfreq'])
            if coil['coil_freq'] > raw_dec.info['sfreq'] / 2.:
                coil['coil_freq'] = raw_dec.info['sfreq'] - coil['coil_freq']
    return raw_dec


@pytest.mark.slowtest
@testing.requires_testing_data
def test_calculate_chpi_positions():
    """Test calculation of cHPI positions."""
    # Check to make sure our fits match MF decently
    mf_quats = read_head_pos(pos_fname)
    raw = read_raw_fif(chpi_fif_fname, allow_maxshield='yes', preload=True)
    # This is a little hack (aliasing while decimating) to make it much faster
    # for testing purposes only. We can relax this later if we find it breaks
    # something.
    raw_dec = _decimate_chpi(raw, 15)
    with catch_logging() as log:
        py_quats = _calculate_chpi_positions(raw_dec, t_step_max=1.,
                                             verbose='debug')
    assert log.getvalue().startswith('HPIFIT')
    _assert_quats(py_quats, mf_quats, dist_tol=0.004, angle_tol=2.5)

    # degenerate conditions
    raw_no_chpi = read_raw_fif(test_fif_fname)
    pytest.raises(RuntimeError, _calculate_chpi_positions, raw_no_chpi)
    raw_bad = raw.copy()
    del raw_bad.info['hpi_meas'][0]['hpi_coils'][0]['coil_freq']
    pytest.raises(RuntimeError, _calculate_chpi_positions, raw_bad)
    raw_bad = raw.copy()
    for d in raw_bad.info['dig']:
        if d['kind'] == FIFF.FIFFV_POINT_HPI:
            d['coord_frame'] = FIFF.FIFFV_COORD_UNKNOWN
            break
    pytest.raises(RuntimeError, _calculate_chpi_positions, raw_bad)
    for d in raw_bad.info['dig']:
        if d['kind'] == FIFF.FIFFV_POINT_HPI:
            d['coord_frame'] = FIFF.FIFFV_COORD_HEAD
            d['r'] = np.ones(3)
    raw_bad.crop(0, 1.)
    picks = np.concatenate([np.arange(306, len(raw_bad.ch_names)),
                            pick_types(raw_bad.info, meg=True)[::16]])
    raw_bad.pick_channels([raw_bad.ch_names[pick] for pick in picks])
    with pytest.warns(RuntimeWarning, match='Discrepancy'):
        with catch_logging() as log_file:
            _calculate_chpi_positions(raw_bad, t_step_min=1., verbose=True)
    # ignore HPI info header and [done] footer
    assert '0/5 good' in log_file.getvalue().strip().split('\n')[-2]

    # half the rate cuts off cHPI coils
    raw.info['lowpass'] /= 2.
    with pytest.raises(RuntimeError, match='above the'):
        _calculate_chpi_positions(raw)

    # test on 5k artemis data
    raw = read_raw_artemis123(art_fname, preload=True)
    mf_quats = read_head_pos(art_mc_fname)
    with catch_logging() as log:
        py_quats = _calculate_chpi_positions(raw, t_step_min=2.,
                                             verbose='debug')
    _assert_quats(py_quats, mf_quats, dist_tol=0.004, angle_tol=2.5)


@testing.requires_testing_data
def test_calculate_chpi_positions_on_chpi5_in_one_second_steps():
    """Comparing estimated cHPI positions with MF results (one second)."""
    # Check to make sure our fits match MF decently
    mf_quats = read_head_pos(chpi5_pos_fname)
    raw = read_raw_fif(chpi5_fif_fname, allow_maxshield='yes')
    # the last two seconds contain a maxfilter problem!
    # fiff file timing: 26. to 43. seconds
    # maxfilter estimates a wrong head position for interval 16: 41.-42. sec
    raw = _decimate_chpi(raw.crop(0., 15.).load_data(), decim=8)
    # needs no interpolation, because maxfilter pos files comes with 1 s steps
    py_quats = _calculate_chpi_positions(raw, t_step_min=1.0, t_step_max=1.0,
                                         t_window=1.0, verbose='debug')
    _assert_quats(py_quats, mf_quats, dist_tol=0.0008, angle_tol=.5)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_calculate_chpi_positions_on_chpi5_in_shorter_steps():
    """Comparing estimated cHPI positions with MF results (smaller steps)."""
    # Check to make sure our fits match MF decently
    mf_quats = read_head_pos(chpi5_pos_fname)
    raw = read_raw_fif(chpi5_fif_fname, allow_maxshield='yes')
    raw = _decimate_chpi(raw.crop(0., 15.).load_data(), decim=8)
    py_quats = _calculate_chpi_positions(raw, t_step_min=0.1, t_step_max=0.1,
                                         t_window=0.1, verbose='debug')
    # needs interpolation, tolerance must be increased
    _assert_quats(py_quats, mf_quats, dist_tol=0.001, angle_tol=0.6)


def test_simulate_calculate_chpi_positions():
    """Test calculation of cHPI positions with simulated data."""
    # Read info dict from raw FIF file
    info = read_info(raw_fname)
    # Tune the info structure
    chpi_channel = u'STI201'
    ncoil = len(info['hpi_results'][0]['order'])
    coil_freq = 10 + np.arange(ncoil) * 5
    hpi_subsystem = {'event_channel': chpi_channel,
                     'hpi_coils': [{'event_bits': np.array([256, 0, 256, 256],
                                                           dtype=np.int32)},
                                   {'event_bits': np.array([512, 0, 512, 512],
                                                           dtype=np.int32)},
                                   {'event_bits':
                                       np.array([1024, 0, 1024, 1024],
                                                dtype=np.int32)},
                                   {'event_bits':
                                       np.array([2048, 0, 2048, 2048],
                                                dtype=np.int32)}],
                     'ncoil': ncoil}

    info['hpi_subsystem'] = hpi_subsystem
    for l, freq in enumerate(coil_freq):
        info['hpi_meas'][0]['hpi_coils'][l]['coil_freq'] = freq
    picks = pick_types(info, meg=True, stim=True, eeg=False, exclude=[])
    info['sfreq'] = 100.  # this will speed it up a lot
    info = pick_info(info, picks)
    info['chs'][info['ch_names'].index('STI 001')]['ch_name'] = 'STI201'
    info._update_redundant()
    info['projs'] = []

    info_trans = info['dev_head_t']['trans'].copy()

    dev_head_pos_ini = np.concatenate([rot_to_quat(info_trans[:3, :3]),
                                       info_trans[:3, 3]])
    ez = np.array([0, 0, 1])  # Unit vector in z-direction of head coordinates

    # Define some constants
    duration = 10  # Time / s

    # Quotient of head position sampling frequency
    # and raw sampling frequency
    head_pos_sfreq_quotient = 0.01

    # Round number of head positions to the next integer
    S = int(duration * info['sfreq'] * head_pos_sfreq_quotient)
    assert S == 10
    dz = 0.001  # Shift in z-direction is 0.1mm for each step

    dev_head_pos = np.zeros((S, 10))
    dev_head_pos[:, 0] = np.arange(S) * info['sfreq'] * head_pos_sfreq_quotient
    dev_head_pos[:, 1:4] = dev_head_pos_ini[:3]
    dev_head_pos[:, 4:7] = dev_head_pos_ini[3:] + \
        np.outer(np.arange(S) * dz, ez)
    dev_head_pos[:, 7] = 1.0

    # cm/s
    dev_head_pos[:, 9] = 100 * dz / (info['sfreq'] * head_pos_sfreq_quotient)

    # Round number of samples to the next integer
    raw_data = np.zeros((len(picks), int(duration * info['sfreq'] + 0.5)))
    raw = RawArray(raw_data, info)
    add_chpi(raw, dev_head_pos)
    quats = _calculate_chpi_positions(
        raw, t_step_min=raw.info['sfreq'] * head_pos_sfreq_quotient,
        t_step_max=raw.info['sfreq'] * head_pos_sfreq_quotient, t_window=1.0)
    _assert_quats(quats, dev_head_pos, dist_tol=0.001, angle_tol=1.)


@testing.requires_testing_data
def test_calculate_chpi_coil_locs():
    """Test computing just cHPI locations."""
    raw = read_raw_fif(chpi_fif_fname, allow_maxshield='yes', preload=True)
    # This is a little hack (aliasing while decimating) to make it much faster
    # for testing purposes only. We can relax this later if we find it breaks
    # something.
    raw_dec = _decimate_chpi(raw, 15)
    times, cHPI_digs = _calculate_chpi_coil_locs(raw_dec, verbose='debug')

    # spot check
    assert_allclose(times[9], 9.9, atol=1e-3)
    assert_allclose(cHPI_digs[9][2]['r'],
                    [-0.01937833, 0.00346804, 0.06331209], atol=1e-3)
    assert_allclose(cHPI_digs[9][2]['gof'], 0.9957976, atol=1e-3)

    assert_allclose(cHPI_digs[9][4]['r'],
                    [0.05442122, 0.00997692, 0.03721696], atol=1e-3)
    assert_allclose(cHPI_digs[9][4]['gof'], 0.075700080794629199, atol=1e-3)

    # test on 5k artemis data
    raw = read_raw_artemis123(art_fname, preload=True)
    times, cHPI_digs = _calculate_chpi_coil_locs(raw, verbose='debug')

    assert_allclose(times[2], 2.9, atol=1e-3)
    assert_allclose(cHPI_digs[2][0]['gof'], 0.9980471794552791, atol=1e-3)
    assert_allclose(cHPI_digs[2][0]['r'],
                    [-0.0157762, 0.06655744, 0.00545172], atol=1e-3)
    with pytest.raises(ValueError, match='too_close'):
        _calculate_chpi_coil_locs(raw, too_close='foo')


@testing.requires_testing_data
def test_chpi_subtraction():
    """Test subtraction of cHPI signals."""
    raw = read_raw_fif(chpi_fif_fname, allow_maxshield='yes', preload=True)
    raw.info['bads'] = ['MEG0111']
    raw.del_proj()
    with catch_logging() as log:
        filter_chpi(raw, include_line=False, verbose=True)
    assert 'No average EEG' not in log.getvalue()
    assert '5 cHPI' in log.getvalue()
    # MaxFilter doesn't do quite as well as our algorithm with the last bit
    raw.crop(0, 16)
    # remove cHPI status chans
    raw_c = read_raw_fif(sss_hpisubt_fname).crop(0, 16).load_data()
    raw_c.pick_types(
        meg=True, eeg=True, eog=True, ecg=True, stim=True, misc=True)
    assert_meg_snr(raw, raw_c, 143, 624)

    # Degenerate cases
    raw_nohpi = read_raw_fif(test_fif_fname, preload=True)
    pytest.raises(RuntimeError, filter_chpi, raw_nohpi)

    # When MaxFliter downsamples, like::
    #     $ maxfilter -nosss -ds 2 -f test_move_anon_raw.fif \
    #           -o test_move_anon_ds2_raw.fif
    # it can strip out some values of info, which we emulate here:
    raw = read_raw_fif(chpi_fif_fname, allow_maxshield='yes')
    raw = raw.crop(0, 1).load_data().resample(600., npad='auto')
    raw.info['lowpass'] = 200.
    del raw.info['maxshield']
    del raw.info['hpi_results'][0]['moments']
    del raw.info['hpi_subsystem']['event_channel']
    with catch_logging() as log:
        filter_chpi(raw, verbose=True)
    pytest.raises(ValueError, filter_chpi, raw, t_window=-1)
    assert '2 cHPI' in log.getvalue()


@testing.requires_testing_data
def test_calculate_head_pos_ctf():
    """Test extracting of cHPI positions from ctf data."""
    raw = read_raw_ctf(ctf_chpi_fname)
    quats = _calculate_head_pos_ctf(raw)
    mc_quats = read_head_pos(ctf_chpi_pos_fname)
    _assert_quats(quats, mc_quats, dist_tol=0.004, angle_tol=2.5)

    raw = read_raw_fif(ctf_fname)
    pytest.raises(RuntimeError, _calculate_head_pos_ctf, raw)


run_tests_if_main()
