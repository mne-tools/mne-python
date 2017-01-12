from __future__ import print_function
import warnings
import os.path as op
import copy as cp

from nose.tools import assert_true, assert_raises, assert_equal
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import mne
from mne.datasets import testing
from mne.beamformer import dics, dics_epochs, dics_source_power, tf_dics
from mne.time_frequency import csd_epochs
from mne.externals.six import advance_iterator
from mne.utils import run_tests_if_main

# Note that this is the first test file, this will apply to all subsequent
# tests in a full nosetest:
warnings.simplefilter("always")  # ensure we can verify expected warnings

data_path = testing.data_path(download=False)
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_fwd_vol = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc-meg-vol-7-fwd.fif')
fname_event = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc_raw-eve.fif')
label = 'Aud-lh'
fname_label = op.join(data_path, 'MEG', 'sample', 'labels', '%s.label' % label)


def read_forward_solution_meg(*args, **kwargs):
    fwd = mne.read_forward_solution(*args, **kwargs)
    return mne.pick_types_forward(fwd, meg=True, eeg=False)


def _get_data(tmin=-0.11, tmax=0.15, read_all_forward=True, compute_csds=True):
    """Read in data used in tests
    """
    label = mne.read_label(fname_label)
    events = mne.read_events(fname_event)[:10]
    raw = mne.io.read_raw_fif(fname_raw, preload=False)
    raw.add_proj([], remove_existing=True)  # we'll subselect so remove proj
    forward = mne.read_forward_solution(fname_fwd)
    if read_all_forward:
        forward_surf_ori = read_forward_solution_meg(fname_fwd, surf_ori=True)
        forward_fixed = read_forward_solution_meg(fname_fwd, force_fixed=True,
                                                  surf_ori=True)
        forward_vol = mne.read_forward_solution(fname_fwd_vol, surf_ori=True)
    else:
        forward_surf_ori = None
        forward_fixed = None
        forward_vol = None

    event_id, tmin, tmax = 1, tmin, tmax

    # Setup for reading the raw data
    raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

    # Set up pick list: MEG - bad channels
    left_temporal_channels = mne.read_selection('Left-temporal')
    picks = mne.pick_types(raw.info, meg=True, eeg=False,
                           stim=True, eog=True, exclude='bads',
                           selection=left_temporal_channels)

    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks, baseline=(None, 0), preload=True,
                        reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
    epochs.resample(200, npad=0, n_jobs=2)
    evoked = epochs.average()

    # Computing the data and noise cross-spectral density matrices
    if compute_csds:
        data_csd = csd_epochs(epochs, mode='multitaper', tmin=0.045,
                              tmax=None, fmin=8, fmax=12,
                              mt_bandwidth=72.72)
        noise_csd = csd_epochs(epochs, mode='multitaper', tmin=None,
                               tmax=0.0, fmin=8, fmax=12,
                               mt_bandwidth=72.72)
    else:
        data_csd, noise_csd = None, None

    return raw, epochs, evoked, data_csd, noise_csd, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol


@testing.requires_testing_data
def test_dics():
    """Test DICS with evoked data and single trials
    """
    raw, epochs, evoked, data_csd, noise_csd, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol = _get_data()

    stc = dics(evoked, forward, noise_csd=noise_csd, data_csd=data_csd,
               label=label)

    stc.crop(0, None)
    stc_pow = np.sum(stc.data, axis=1)
    idx = np.argmax(stc_pow)
    max_stc = stc.data[idx]
    tmax = stc.times[np.argmax(max_stc)]

    # Incorrect due to limited number of epochs
    assert_true(0.04 < tmax < 0.05)
    assert_true(10 < np.max(max_stc) < 13)

    # Test picking normal orientation
    stc_normal = dics(evoked, forward_surf_ori, noise_csd, data_csd,
                      pick_ori="normal", label=label)
    stc_normal.crop(0, None)

    # The amplitude of normal orientation results should always be smaller than
    # free orientation results
    assert_true((np.abs(stc_normal.data) <= stc.data).all())

    # Test if fixed forward operator is detected when picking normal
    # orientation
    assert_raises(ValueError, dics_epochs, epochs, forward_fixed, noise_csd,
                  data_csd, pick_ori="normal")

    # Test if non-surface oriented forward operator is detected when picking
    # normal orientation
    assert_raises(ValueError, dics_epochs, epochs, forward, noise_csd,
                  data_csd, pick_ori="normal")

    # Test if volume forward operator is detected when picking normal
    # orientation
    assert_raises(ValueError, dics_epochs, epochs, forward_vol, noise_csd,
                  data_csd, pick_ori="normal")

    # Now test single trial using fixed orientation forward solution
    # so we can compare it to the evoked solution
    stcs = dics_epochs(epochs, forward_fixed, noise_csd, data_csd, reg=0.01,
                       label=label)

    # Testing returning of generator
    stcs_ = dics_epochs(epochs, forward_fixed, noise_csd, data_csd, reg=0.01,
                        return_generator=True, label=label)
    assert_array_equal(stcs[0].data, advance_iterator(stcs_).data)

    # Test whether correct number of trials was returned
    epochs.drop_bad()
    assert_true(len(epochs.events) == len(stcs))

    # Average the single trial estimates
    stc_avg = np.zeros_like(stc.data)
    for this_stc in stcs:
        stc_avg += this_stc.crop(0, None).data
    stc_avg /= len(stcs)

    idx = np.argmax(np.max(stc_avg, axis=1))
    max_stc = stc_avg[idx]
    tmax = stc.times[np.argmax(max_stc)]

    assert_true(0.045 < tmax < 0.06)  # incorrect due to limited # of epochs
    assert_true(12 < np.max(max_stc) < 18.5)


@testing.requires_testing_data
def test_dics_source_power():
    """Test DICS source power computation
    """
    raw, epochs, evoked, data_csd, noise_csd, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol = _get_data()

    stc_source_power = dics_source_power(epochs.info, forward, noise_csd,
                                         data_csd, label=label)

    max_source_idx = np.argmax(stc_source_power.data)
    max_source_power = np.max(stc_source_power.data)

    # TODO: Maybe these could be more directly compared to dics() results?
    assert_true(max_source_idx == 0)
    assert_true(0.5 < max_source_power < 1.15)

    # Test picking normal orientation and using a list of CSD matrices
    stc_normal = dics_source_power(epochs.info, forward_surf_ori,
                                   [noise_csd] * 2, [data_csd] * 2,
                                   pick_ori="normal", label=label)

    assert_true(stc_normal.data.shape == (stc_source_power.data.shape[0], 2))

    # The normal orientation results should always be smaller than free
    # orientation results
    assert_true((np.abs(stc_normal.data[:, 0]) <=
                 stc_source_power.data[:, 0]).all())

    # Test if fixed forward operator is detected when picking normal
    # orientation
    assert_raises(ValueError, dics_source_power, raw.info, forward_fixed,
                  noise_csd, data_csd, pick_ori="normal")

    # Test if non-surface oriented forward operator is detected when picking
    # normal orientation
    assert_raises(ValueError, dics_source_power, raw.info, forward, noise_csd,
                  data_csd, pick_ori="normal")

    # Test if volume forward operator is detected when picking normal
    # orientation
    assert_raises(ValueError, dics_source_power, epochs.info, forward_vol,
                  noise_csd, data_csd, pick_ori="normal")

    # Test detection of different number of CSD matrices provided
    assert_raises(ValueError, dics_source_power, epochs.info, forward,
                  [noise_csd] * 2, [data_csd] * 3)

    # Test detection of different frequencies in noise and data CSD objects
    noise_csd.frequencies = [1, 2]
    data_csd.frequencies = [1, 2, 3]
    assert_raises(ValueError, dics_source_power, epochs.info, forward,
                  noise_csd, data_csd)

    # Test detection of uneven frequency spacing
    data_csds = [cp.deepcopy(data_csd) for i in range(3)]
    frequencies = [1, 3, 4]
    for freq, data_csd in zip(frequencies, data_csds):
        data_csd.frequencies = [freq]
    noise_csds = data_csds
    with warnings.catch_warnings(record=True) as w:
        dics_source_power(epochs.info, forward, noise_csds, data_csds)
    assert_equal(len(w), 1)


@testing.requires_testing_data
def test_tf_dics():
    """Test TF beamforming based on DICS
    """
    tmin, tmax, tstep = -0.2, 0.2, 0.1
    raw, epochs, _, _, _, label, forward, _, _, _ =\
        _get_data(tmin, tmax, read_all_forward=False, compute_csds=False)

    freq_bins = [(4, 20), (30, 55)]
    win_lengths = [0.2, 0.2]
    reg = 0.001

    noise_csds = []
    for freq_bin, win_length in zip(freq_bins, win_lengths):
        noise_csd = csd_epochs(epochs, mode='fourier',
                               fmin=freq_bin[0], fmax=freq_bin[1],
                               fsum=True, tmin=tmin,
                               tmax=tmin + win_length)
        noise_csds.append(noise_csd)

    stcs = tf_dics(epochs, forward, noise_csds, tmin, tmax, tstep, win_lengths,
                   freq_bins, reg=reg, label=label)

    assert_true(len(stcs) == len(freq_bins))
    assert_true(stcs[0].shape[1] == 4)

    # Manually calculating source power in several time windows to compare
    # results and test overlapping
    source_power = []
    time_windows = [(-0.1, 0.1), (0.0, 0.2)]
    for time_window in time_windows:
        data_csd = csd_epochs(epochs, mode='fourier',
                              fmin=freq_bins[0][0],
                              fmax=freq_bins[0][1], fsum=True,
                              tmin=time_window[0], tmax=time_window[1])
        noise_csd = csd_epochs(epochs, mode='fourier',
                               fmin=freq_bins[0][0],
                               fmax=freq_bins[0][1], fsum=True,
                               tmin=-0.2, tmax=0.0)
        data_csd.data /= data_csd.n_fft
        noise_csd.data /= noise_csd.n_fft
        stc_source_power = dics_source_power(epochs.info, forward, noise_csd,
                                             data_csd, reg=reg, label=label)
        source_power.append(stc_source_power.data)

    # Averaging all time windows that overlap the time period 0 to 100 ms
    source_power = np.mean(source_power, axis=0)

    # Selecting the first frequency bin in tf_dics results
    stc = stcs[0]

    # Comparing tf_dics results with dics_source_power results
    assert_array_almost_equal(stc.data[:, 2], source_power[:, 0])

    # Test if using unsupported max-power orientation is detected
    assert_raises(ValueError, tf_dics, epochs, forward, noise_csds, tmin, tmax,
                  tstep, win_lengths, freq_bins=freq_bins,
                  pick_ori='max-power')

    # Test if incorrect number of noise CSDs is detected
    assert_raises(ValueError, tf_dics, epochs, forward, [noise_csds[0]], tmin,
                  tmax, tstep, win_lengths, freq_bins=freq_bins)

    # Test if freq_bins and win_lengths incompatibility is detected
    assert_raises(ValueError, tf_dics, epochs, forward, noise_csds, tmin, tmax,
                  tstep, win_lengths=[0, 1, 2], freq_bins=freq_bins)

    # Test if time step exceeding window lengths is detected
    assert_raises(ValueError, tf_dics, epochs, forward, noise_csds, tmin, tmax,
                  tstep=0.15, win_lengths=[0.2, 0.1], freq_bins=freq_bins)

    # Test if incorrect number of mt_bandwidths is detected
    assert_raises(ValueError, tf_dics, epochs, forward, noise_csds, tmin, tmax,
                  tstep, win_lengths, freq_bins, mode='multitaper',
                  mt_bandwidths=[20])

    # Pass only one epoch to test if subtracting evoked responses yields zeros
    stcs = tf_dics(epochs[0], forward, noise_csds, tmin, tmax, tstep,
                   win_lengths, freq_bins, subtract_evoked=True, reg=reg,
                   label=label)

    assert_array_almost_equal(stcs[0].data, np.zeros_like(stcs[0].data))


run_tests_if_main()
