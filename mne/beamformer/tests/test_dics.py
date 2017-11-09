# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Britta Westner
#
# License: BSD 3 clause
from __future__ import print_function
import warnings
import os.path as op
import copy as cp

from pytest import raises
from numpy.testing import assert_array_equal, assert_allclose
import numpy as np

import mne
from mne.datasets import testing
from mne.beamformer import (make_dics, apply_dics, apply_dics_epochs,
                            apply_dics_csd, dics, dics_epochs,
                            dics_source_power, tf_dics)
from mne.time_frequency import csd_epochs
from mne.utils import run_tests_if_main
from mne.externals.six import advance_iterator

# Note that if this is the first test file, this will apply to all subsequent
# tests in a full nosetest:
warnings.simplefilter('always')  # ensure we can verify expected warnings

# Silence these warnings
warnings.simplefilter('ignore', category=DeprecationWarning)

data_path = testing.data_path(download=False)
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_fwd_vol = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc-meg-vol-7-fwd.fif')
fname_event = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc_raw-eve.fif')

subjects_dir = op.join(data_path, 'subjects')
fname_label = op.join(subjects_dir, 'sample', 'label', 'aparc',
                      'rostralmiddlefrontal-lh.label')


def _load_forward():
    """Load forward models"""
    fwd_free = mne.read_forward_solution(fname_fwd)
    fwd_free = mne.pick_types_forward(fwd_free, meg=True, eeg=False)
    fwd_free = mne.convert_forward_solution(fwd_free, surf_ori=False)
    fwd_surf = mne.convert_forward_solution(fwd_free, surf_ori=True,
                                            use_cps=False)
    fwd_fixed = mne.convert_forward_solution(fwd_free, force_fixed=True,
                                             use_cps=False)
    fwd_vol = mne.read_forward_solution(fname_fwd_vol)
    label = mne.read_label(fname_label)

    return fwd_free, fwd_surf, fwd_fixed, fwd_vol, label


def _simulate_data(fwd):
    """Simulate an oscillator on the cortex."""
    source_vertno = 146374  # Somewhere on the frontal lobe

    sfreq = 50.  # Hz.
    times = np.arange(10 * sfreq) / sfreq  # 10 seconds of data
    signal = np.sin(20 * 2 * np.pi * times)  # 20 Hz oscillator
    signal[:len(times) // 2] *= 2  # Make signal louder at the beginning
    signal *= 1e-9  # Scale to be in the ballpark of MEG data

    # Construct a SourceEstimate object that describes the signal at the
    # cortical level.
    stc = mne.SourceEstimate(
        signal[np.newaxis, :],
        vertices=[[source_vertno], []],
        tmin=0,
        tstep=1 / sfreq,
        subject='sample',
    )

    # Create an info object that holds information about the sensors
    info = mne.create_info(fwd['info']['ch_names'], sfreq, ch_types='grad')
    info.update(fwd['info'])  # Merge in sensor position information

    # Run the simulated signal through the forward model, obtaining
    # simulated sensor data.
    raw = mne.apply_forward_raw(fwd, stc, info)

    # Add a little noise
    random = np.random.RandomState(42)
    noise = random.randn(*raw._data.shape) * 1e-14
    raw._data += noise

    # Define a single epoch
    epochs = mne.Epochs(raw, [[0, 0, 1]], event_id=1, tmin=0,
                        tmax=raw.times[-1], preload=True)
    evoked = epochs.average()

    # Compute the cross-spectral density matrix
    csd = csd_epochs(epochs, mode='cwt_morlet', frequencies=[10, 20],
                     fsum=False, cwt_n_cycles=[5, 10], decim=10)

    return epochs, evoked, csd, source_vertno


@testing.requires_testing_data
def test_make_dics():
    """Test making DICS beamformer filters."""
    # We only test proper handling of parameters here. Testing the results is
    # done in apply_dics_timeseries and apply_dics_csd.

    fwd_free, fwd_surf, fwd_fixed, fwd_vol, label = _load_forward()
    epochs, _, csd, _ = _simulate_data(fwd_fixed)

    raises(ValueError, make_dics, epochs.info, fwd_fixed, csd,
           pick_ori="notexistent")

    # Test if fixed forward operator is detected when picking normal
    # orientation
    raises(ValueError, make_dics, epochs.info, fwd_fixed, csd,
           pick_ori="normal")

    # Test if non-surface oriented forward operator is detected when picking
    # normal orientation
    raises(ValueError, make_dics, epochs.info, fwd_free, csd,
           pick_ori="normal")

    # Test if volume forward operator is detected when picking normal
    # orientation
    raises(ValueError, make_dics, epochs.info, fwd_vol, csd, pick_ori="normal")

    # Sanity checks on the returned filters
    n_freq = len(csd.frequencies)
    vertices = np.intersect1d(label.vertices, fwd_free['src'][0]['vertno'])
    n_verts = len(vertices)
    n_orient = 3
    n_channels = csd.n_series

    filters = make_dics(epochs.info, fwd_surf, csd, label=label, pick_ori=None)
    assert filters['weights'].shape == (n_freq, n_verts * n_orient, n_channels)
    assert np.iscomplexobj(filters['weights'])
    assert filters['csd'] == csd
    assert filters['ch_names'] == csd.names
    assert_array_equal(filters['proj'], np.eye(n_channels))
    assert_array_equal(filters['vertices'][0], vertices)
    assert_array_equal(filters['vertices'][1], [])  # Label was on the LH
    assert filters['subject'] == fwd_free['src'][0]['subject_his_id']
    assert filters['n_orient'] == n_orient

    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        pick_ori='normal')
    n_orient = 1
    assert filters['weights'].shape == (n_freq, n_verts * n_orient, n_channels)
    assert filters['n_orient'] == n_orient

    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        pick_ori='max-power')
    n_orient = 1
    assert filters['weights'].shape == (n_freq, n_verts * n_orient, n_channels)
    assert filters['n_orient'] == n_orient

    # Test using a real-valued filter
    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        pick_ori='normal', real_filter=True)
    assert not np.iscomplexobj(filters['weights'])


@testing.requires_testing_data
def test_apply_dics_csd():
    """Test applying a DICS beamformer to a CSD matrix."""
    fwd_free, fwd_surf, fwd_fixed, fwd_vol, label = _load_forward()
    epochs, _, csd, source_vertno = _simulate_data(fwd_fixed)
    vertices = np.intersect1d(label.vertices, fwd_free['src'][0]['vertno'])
    source_ind = vertices.tolist().index(source_vertno)
    reg = 1  # Lots of regularization for our toy dataset

    # Try different types of forward models
    for fwd in [fwd_free, fwd_surf, fwd_fixed]:
        filters = make_dics(epochs.info, fwd, csd, label=label, reg=reg)
        power, f = apply_dics_csd(csd, filters)
        assert f == [10, 20]

        # Did we find the true source at 20 Hz?
        assert np.argmax(power.data[:, 1]) == source_ind

        # Is the signal stronger at 20 Hz than 10?
        assert power.data[source_ind, 1] > power.data[source_ind, 0]

    # Try picking different orientations
    for pick_ori in [None, 'normal', 'max-power']:
        filters = make_dics(epochs.info, fwd_surf, csd, label=label, reg=reg,
                            pick_ori=pick_ori)
        power, f = apply_dics_csd(csd, filters)
        assert f == [10, 20]
        assert np.argmax(power.data[:, 1]) == source_ind
        assert power.data[source_ind, 1] > power.data[source_ind, 0]

    # Test using a real-valued filter
    filters_real = make_dics(epochs.info, fwd_surf, csd, label=label, reg=reg,
                             real_filter=True)
    power, f = apply_dics_csd(csd, filters_real)
    assert f == [10, 20]
    assert np.argmax(power.data[:, 1]) == source_ind
    assert power.data[source_ind, 1] > power.data[source_ind, 0]

    # Test computing source power on a volume source space
    filters_vol = make_dics(epochs.info, fwd_vol, csd, reg=reg)
    power, f = apply_dics_csd(csd, filters_vol)
    vol_source_ind = 3851  # FIXME: not make this hardcoded
    assert f == [10, 20]
    assert np.argmax(power.data[:, 1]) == vol_source_ind
    assert power.data[vol_source_ind, 1] > power.data[vol_source_ind, 0]


@testing.requires_testing_data
def test_apply_dics_timeseries():
    """Test DICS applied to timeseries data."""
    fwd_free, fwd_surf, fwd_fixed, fwd_vol, label = _load_forward()
    epochs, evoked, csd, source_vertno = _simulate_data(fwd_fixed)
    vertices = np.intersect1d(label.vertices, fwd_free['src'][0]['vertno'])
    source_ind = vertices.tolist().index(source_vertno)
    reg = 2  # Lots of regularization for our toy dataset

    multiple_filters = make_dics(evoked.info, fwd_surf, csd, label=label,
                                 reg=reg)

    # Sanity checks on the resulting STC after applying DICS on evoked
    stcs = apply_dics(evoked, multiple_filters)
    assert isinstance(stcs, list)
    assert len(stcs) == len(multiple_filters['weights'])
    assert_array_equal(stcs[0].vertices[0], multiple_filters['vertices'][0])
    assert_array_equal(stcs[0].vertices[1], multiple_filters['vertices'][1])
    assert_allclose(stcs[0].times, evoked.times)

    # Applying filters for multiple frequencies on epoch data should fail
    raises(ValueError, apply_dics_epochs, epochs, multiple_filters)

    # From now on, only apply filters with a single frequency (20 Hz).
    csd20 = csd.pick_frequency(20)
    filters = make_dics(evoked.info, fwd_surf, csd20, label=label, reg=reg)

    # Sanity checks on the resulting STC after applying DICS on epochs.
    stcs = apply_dics_epochs(epochs, filters)
    assert isinstance(stcs, list)
    assert len(stcs) == 1
    assert_array_equal(stcs[0].vertices[0], filters['vertices'][0])
    assert_array_equal(stcs[0].vertices[1], filters['vertices'][1])
    assert_allclose(stcs[0].times, epochs.times)

    # Test if wrong channel selection is detected in application of filter
    evoked_ch = cp.deepcopy(evoked)
    evoked_ch.pick_channels(evoked_ch.ch_names[:-1])
    raises(ValueError, apply_dics, evoked_ch, filters)

    # Try different types of forward models
    for fwd in [fwd_free, fwd_surf, fwd_fixed]:
        filters = make_dics(evoked.info, fwd, csd20, label=label, reg=reg)

        # Apply filters to evoked
        stc = apply_dics(evoked, filters)
        stc = (stc ** 2).mean()
        assert np.argmax(stc.data) == source_ind

        # Apply filters to epochs
        stcs = apply_dics_epochs(epochs, filters)
        stc = (stcs[0] ** 2).mean()
        assert np.argmax(stc.data) == source_ind

    # Try picking different orientations
    for pick_ori in [None, 'normal', 'max-power']:
        filters = make_dics(evoked.info, fwd_surf, csd20, label=label, reg=reg,
                            pick_ori=pick_ori)

        # Apply to evoked
        stc = apply_dics(evoked, filters)
        stc = (stc ** 2).mean()
        assert np.argmax(stc.data) == source_ind

        # Apply to epochs
        stcs = apply_dics_epochs(epochs, filters)
        stc = (stcs[0] ** 2).mean()
        assert np.argmax(stc.data) == source_ind

    # Test using a real-valued filter
    filters_real = make_dics(evoked.info, fwd_surf, csd20, label=label,
                             reg=reg, real_filter=True)
    assert not np.iscomplexobj(filters_real['weights'])
    stc = apply_dics(evoked, filters_real)
    stc = (stc ** 2).mean()
    assert np.argmax(stc.data) == source_ind

    # Test returning a generator
    stcs = apply_dics_epochs(epochs, filters, return_generator=False)
    stcs_gen = apply_dics_epochs(epochs, filters, return_generator=True)
    assert_array_equal(stcs[0].data, advance_iterator(stcs_gen).data)

    # Test computing timecourses on a volume source space
    filters_vol = make_dics(evoked.info, fwd_vol, csd20, reg=reg)
    stc = apply_dics(evoked, filters_vol)
    stc = (stc ** 2).mean()
    assert np.argmax(stc.data) == 3851  # TODO: don't make this hard coded


@testing.requires_testing_data
def test_tf_dics():
    """Test 5D time-frequency beamforming based on DICS."""
    fwd_free, fwd_surf, fwd_fixed, fwd_vol, label = _load_forward()
    epochs, evoked, csd, source_vertno = _simulate_data(fwd_fixed)
    vertices = np.intersect1d(label.vertices, fwd_free['src'][0]['vertno'])
    source_ind = vertices.tolist().index(source_vertno)
    reg = 1  # Lots of regularization for our toy dataset

    tmin = 0
    tmax = 9
    tstep = 4
    win_lengths = [5, 5]
    frequencies = [10, 20]
    freq_bins = [(8, 12), (18, 22)]

    # Compute DICS for two time windows and two frequencies
    for mode in ['fourier', 'multitaper', 'cwt_morlet']:
        stcs = tf_dics(epochs, fwd_surf, tmin, tmax, tstep, win_lengths,
                       mode=mode, freq_bins=freq_bins, frequencies=frequencies,
                       decim=10, reg=reg, label=label)

        # Did we find the true source at 20 Hz?
        assert np.argmax(stcs[1].data[:, 0]) == source_ind
        assert np.argmax(stcs[1].data[:, 1]) == source_ind

        # 20 Hz power should decrease over time
        assert stcs[1].data[source_ind, 0] > stcs[1].data[source_ind, 1]

        # 20 Hz power should be more than 10 Hz power at the true source
        assert stcs[1].data[source_ind, 0] > stcs[0].data[source_ind, 0]

    # Manually compute source power and compare with the last tf_dics result
    source_power = []
    time_windows = [(0, 5), (4, 9)]
    for time_window in time_windows:
        csd = csd_epochs(epochs, mode='cwt_morlet',
                         frequencies=[frequencies[1]], tmin=time_window[0],
                         tmax=time_window[1], decim=10)
        filters = make_dics(epochs.info, fwd_surf, csd, reg=reg, label=label)
        stc_source_power, _ = apply_dics_csd(csd, filters)
        source_power.append(stc_source_power.data)

    # Comparing tf_dics results with dics_source_power results
    assert_allclose(stcs[1].data, np.array(source_power).squeeze().T)

    # Test if freq_bins and win_lengths incompatibility is detected
    raises(ValueError, tf_dics, epochs, fwd_surf, tmin, tmax, tstep,
           win_lengths=[0, 1, 2], frequencies=frequencies)

    # Test if time step exceeding window lengths is detected
    raises(ValueError, tf_dics, epochs, fwd_surf, tmin, tmax, tstep=0.15,
           win_lengths=[0.2, 0.1], frequencies=frequencies)

    # Test if incorrect number of mt_bandwidths is detected
    raises(ValueError, tf_dics, epochs, fwd_surf, tmin, tmax, tstep,
           win_lengths=win_lengths, freq_bins=[frequencies], mode='multitaper',
           mt_bandwidths=[20, 30])

    # Test if subtracting evoked responses yields NaN's, since we only have one
    # epoch.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stcs = tf_dics(epochs, fwd_surf, tmin, tmax, tstep, win_lengths,
                       mode='cwt_morlet', frequencies=frequencies,
                       subtract_evoked=True, reg=reg, label=label, decim=20)
    assert len(w) == 60  # One warning for each vertex
    assert np.all(np.isnan(stcs[0].data))


###############################################################################
# Below are tests for the old DICS code. We can remove this for MNE 0.16.
def _get_data(tmin=-0.11, tmax=0.15, read_all_forward=True, compute_csds=True):
    """Read in real MEG data. Used to test deprecated dics_* functions."""
    """Read in data used in tests."""
    if read_all_forward:
        fwd_free, fwd_surf, fwd_fixed, fwd_vol, _ = _load_forward()
    label_fname = op.join(data_path, 'MEG', 'sample', 'labels', 'Aud-lh.label')
    label = mne.read_label(label_fname)
    events = mne.read_events(fname_event)[:10]
    raw = mne.io.read_raw_fif(fname_raw, preload=False)
    raw.add_proj([], remove_existing=True)  # we'll subselect so remove proj
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
    evoked = epochs.average().crop(0, None)

    # Computing the data and noise cross-spectral density matrices
    if compute_csds:
        data_csd = csd_epochs(epochs, mode='multitaper', tmin=0.045,
                              tmax=None, fmin=8, fmax=12,
                              mt_bandwidth=72.72)
        noise_csd = csd_epochs(epochs, mode='multitaper', tmin=None,
                               tmax=0, fmin=8, fmax=12,
                               mt_bandwidth=72.72)
    else:
        data_csd, noise_csd = None, None

    return (raw, epochs, evoked, data_csd, noise_csd, label, fwd_free,
            fwd_surf, fwd_fixed, fwd_vol)


@testing.requires_testing_data
def test_dics():
    """Test DICS with evoked data and single trials."""
    raw, epochs, evoked, data_csd, noise_csd, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol = _get_data()
    epochs.crop(0, None)
    reg = 0.5  # Heavily regularize due to low SNR

    for real_filter in (True, False):
        stc = dics(evoked, forward, noise_csd=noise_csd, data_csd=data_csd,
                   label=label, real_filter=real_filter, reg=reg)
        stc_pow = np.sum(stc.data, axis=1)
        idx = np.argmax(stc_pow)
        max_stc = stc.data[idx]
        tmax = stc.times[np.argmax(max_stc)]

        # Incorrect due to limited number of epochs
        assert 0.04 < tmax < 0.06
        assert 3. < np.max(max_stc) < 6.

    # Test picking normal orientation
    stc_normal = dics(evoked, forward_surf_ori, noise_csd, data_csd,
                      pick_ori="normal", label=label, real_filter=True,
                      reg=reg)
    assert stc_normal.data.min() < 0  # this doesn't take abs
    stc_normal = dics(evoked, forward_surf_ori, noise_csd, data_csd,
                      pick_ori="normal", label=label, reg=reg)
    assert stc_normal.data.min() >= 0  # this does take abs

    # The amplitude of normal orientation results should always be smaller than
    # free orientation results
    assert (np.abs(stc_normal.data) <= stc.data).all()

    # Test if fixed forward operator is detected when picking normal
    # orientation
    raises(ValueError, dics_epochs, epochs, forward_fixed, noise_csd, data_csd,
           pick_ori="normal")

    # Test if non-surface oriented forward operator is detected when picking
    # normal orientation
    raises(ValueError, dics_epochs, epochs, forward, noise_csd, data_csd,
           pick_ori="normal")

    # Test if volume forward operator is detected when picking normal
    # orientation
    raises(ValueError, dics_epochs, epochs, forward_vol, noise_csd, data_csd,
           pick_ori="normal")

    # Now test single trial using fixed orientation forward solution
    # so we can compare it to the evoked solution
    stcs = dics_epochs(epochs, forward_fixed, noise_csd, data_csd, label=label)

    # Testing returning of generator
    stcs_ = dics_epochs(epochs, forward_fixed, noise_csd, data_csd,
                        return_generator=True, label=label)
    assert_array_equal(stcs[0].data, advance_iterator(stcs_).data)

    # Test whether correct number of trials was returned
    epochs.drop_bad()
    assert len(epochs.events) == len(stcs)

    # Average the single trial estimates
    stc_avg = np.zeros_like(stc.data)
    for this_stc in stcs:
        stc_avg += this_stc.data
    stc_avg /= len(stcs)

    idx = np.argmax(np.max(stc_avg, axis=1))
    max_stc = stc_avg[idx]
    tmax = stc.times[np.argmax(max_stc)]

    assert 0.120 < tmax < 0.150  # incorrect due to limited #
    assert 12 < np.max(max_stc) < 18.5


@testing.requires_testing_data
def test_dics_source_power():
    """Test old DICS source power computation."""
    raw, epochs, evoked, data_csd, noise_csd, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol = _get_data()
    epochs.crop(0, None)
    reg = 0.05

    stc_source_power = dics_source_power(epochs.info, forward, noise_csd,
                                         data_csd, label=label, reg=reg)

    max_source_idx = np.argmax(stc_source_power.data)
    max_source_power = np.max(stc_source_power.data)

    # TODO: Maybe these could be more directly compared to dics() results?
    assert max_source_idx == 1
    assert 0.004 < max_source_power < 0.005

    # Test picking normal orientation
    stc_normal = dics_source_power(epochs.info, forward_surf_ori, noise_csd,
                                   data_csd, pick_ori="normal", label=label,
                                   reg=reg)
    assert stc_normal.data.shape == stc_source_power.data.shape

    # The normal orientation results should always be smaller than free
    # orientation results
    assert (np.abs(stc_normal.data) <= stc_source_power.data).all()

    # Test if fixed forward operator is detected when picking normal
    # orientation
    raises(ValueError, dics_source_power, raw.info, forward_fixed, noise_csd,
           data_csd, pick_ori="normal")

    # Test if non-surface oriented forward operator is detected when picking
    # normal orientation
    raises(ValueError, dics_source_power, raw.info, forward, noise_csd,
           data_csd, pick_ori="normal")

    # Test if volume forward operator is detected when picking normal
    # orientation
    raises(ValueError, dics_source_power, epochs.info, forward_vol, noise_csd,
           data_csd, pick_ori="normal")

    # Test detection of different frequencies in noise and data CSD objects
    noise_csd.frequencies = [1, 2]
    data_csd.frequencies = [1, 2, 3]
    raises(ValueError, dics_source_power, epochs.info, forward, noise_csd,
           data_csd)

    # Test detection of uneven frequency spacing
    data_csd.frequencies = [1, 3, 4]
    data_csd._data = data_csd._data.repeat(3, axis=1)
    noise_csd = data_csd
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dics_source_power(epochs.info, forward, noise_csd, data_csd)
    assert len(w) == 2  # Also deprecation warning


run_tests_if_main()
