# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Britta Westner
#
# License: BSD 3 clause
from __future__ import print_function
import warnings
import os.path as op
import copy as cp

import pytest
from pytest import raises
from numpy.testing import assert_array_equal, assert_allclose
import numpy as np

import mne
from mne.datasets import testing
from mne.beamformer import (make_dics, apply_dics, apply_dics_epochs,
                            apply_dics_csd, dics, dics_epochs,
                            dics_source_power, tf_dics)
from mne.time_frequency import csd_multitaper, csd_morlet
from mne.utils import run_tests_if_main
from mne.externals.six import advance_iterator
from mne.proj import compute_proj_evoked, make_projector

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
    csd = csd_morlet(epochs, frequencies=[10, 20], n_cycles=[5, 10], decim=10)

    return epochs, evoked, csd, source_vertno


def _test_weight_norm(filters):
    """Test weight normalization."""
    for ws in filters['weights']:
        ws = ws.reshape(-1, filters['n_orient'], ws.shape[1])
        for w in ws:
            assert_allclose(np.trace(w.dot(w.T)), 1)


@pytest.mark.slowtest
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

    # Test invalid combinations of parameters
    raises(NotImplementedError, make_dics, epochs.info, fwd_free, csd,
           reduce_rank=True, pick_ori=None)
    raises(NotImplementedError, make_dics, epochs.info, fwd_free, csd,
           reduce_rank=True, pick_ori='max-power', inversion='single')

    # Sanity checks on the returned filters
    n_freq = len(csd.frequencies)
    vertices = np.intersect1d(label.vertices, fwd_free['src'][0]['vertno'])
    n_verts = len(vertices)
    n_orient = 3
    n_channels = csd.n_channels

    # Test return values
    filters = make_dics(epochs.info, fwd_surf, csd, label=label, pick_ori=None,
                        weight_norm='unit-noise-gain')
    assert filters['weights'].shape == (n_freq, n_verts * n_orient, n_channels)
    assert np.iscomplexobj(filters['weights'])
    assert filters['csd'] == csd
    assert filters['ch_names'] == csd.ch_names
    assert_array_equal(filters['proj'], np.eye(n_channels))
    assert_array_equal(filters['vertices'][0], vertices)
    assert_array_equal(filters['vertices'][1], [])  # Label was on the LH
    assert filters['subject'] == fwd_free['src'][0]['subject_his_id']
    assert filters['pick_ori'] is None
    assert filters['n_orient'] == n_orient
    assert filters['inversion'] == 'single'
    assert filters['normalize_fwd']
    assert filters['weight_norm'] == 'unit-noise-gain'
    _test_weight_norm(filters)

    # Test picking orientations. Also test weight norming under these different
    # conditions.
    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        pick_ori='normal', weight_norm='unit-noise-gain')
    n_orient = 1
    assert filters['weights'].shape == (n_freq, n_verts * n_orient, n_channels)
    assert filters['n_orient'] == n_orient
    _test_weight_norm(filters)

    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        pick_ori='max-power', weight_norm='unit-noise-gain')
    n_orient = 1
    assert filters['weights'].shape == (n_freq, n_verts * n_orient, n_channels)
    assert filters['n_orient'] == n_orient
    _test_weight_norm(filters)

    # Test using a real-valued filter
    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        pick_ori='normal', real_filter=True)
    assert not np.iscomplexobj(filters['weights'])

    # Test forward normalization. When inversion='single', the power of a
    # unit-noise CSD should be 1, even without weight normalization.
    csd_noise = csd.copy()
    inds = np.triu_indices(csd.n_channels)
    # Using [:, :] syntax for in-place broadcasting
    csd_noise._data[:, :] = np.eye(csd.n_channels)[inds][:, np.newaxis]
    filters = make_dics(epochs.info, fwd_surf, csd_noise, label=label,
                        weight_norm=None, normalize_fwd=True)
    w = filters['weights'][0][:3]
    assert_allclose(np.diag(w.dot(w.T)), 1.0, rtol=1e-6, atol=0)

    # Test turning off both forward and weight normalization
    filters = make_dics(epochs.info, fwd_surf, csd_noise, label=label,
                        weight_norm=None, normalize_fwd=False)
    w = filters['weights'][0][:3]
    assert not np.allclose(np.diag(w.dot(w.T)), 1.0, rtol=1e-2, atol=0)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_apply_dics_csd():
    """Test applying a DICS beamformer to a CSD matrix."""
    fwd_free, fwd_surf, fwd_fixed, fwd_vol, label = _load_forward()
    epochs, _, csd, source_vertno = _simulate_data(fwd_fixed)
    vertices = np.intersect1d(label.vertices, fwd_free['src'][0]['vertno'])
    source_ind = vertices.tolist().index(source_vertno)
    reg = 1  # Lots of regularization for our toy dataset

    # Construct an identity "noise" CSD, which we will use to test the
    # 'unit-noise-gain' setting.
    csd_noise = csd.copy()
    inds = np.triu_indices(csd.n_channels)
    # Using [:, :] syntax for in-place broadcasting
    csd_noise._data[:, :] = np.eye(csd.n_channels)[inds][:, np.newaxis]

    # Try different types of forward models
    for fwd in [fwd_free, fwd_surf, fwd_fixed]:
        filters = make_dics(epochs.info, fwd, csd, label=label, reg=reg,
                            inversion='single')
        power, f = apply_dics_csd(csd, filters)
        assert f == [10, 20]

        # Did we find the true source at 20 Hz?
        assert np.argmax(power.data[:, 1]) == source_ind

        # Is the signal stronger at 20 Hz than 10?
        assert power.data[source_ind, 1] > power.data[source_ind, 0]

    # Try picking different orientations and inversion modes
    for pick_ori in [None, 'normal', 'max-power']:
        for inversion in ['single', 'matrix']:
            # Matrix inversion mode needs more regularization for this toy
            # dataset.
            if inversion == 'matrix':
                reg_ = 5
            else:
                reg_ = reg

            filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                                reg=reg_, pick_ori=pick_ori,
                                inversion=inversion,
                                weight_norm='unit-noise-gain')
            power, f = apply_dics_csd(csd, filters)
            assert f == [10, 20]
            assert np.argmax(power.data[:, 1]) == source_ind
            assert power.data[source_ind, 1] > power.data[source_ind, 0]

            # Test unit-noise-gain weighting
            noise_power, f = apply_dics_csd(csd_noise, filters)
            assert np.allclose(noise_power.data, 1)

            # Test filter with forward normalization instead of weight
            # normalization
            filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                                reg=reg_, pick_ori=pick_ori,
                                inversion=inversion, weight_norm=None,
                                normalize_fwd=True)
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

    # Test rank reduction
    filters_real = make_dics(epochs.info, fwd_surf, csd, label=label, reg=5,
                             pick_ori='max-power', inversion='matrix',
                             reduce_rank=True)
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
    reg = 5  # Lots of regularization for our toy dataset

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

    # Did we find the source?
    stc = (stcs[0] ** 2).mean()
    assert np.argmax(stc.data) == source_ind

    # Apply filters to evoked
    stc = apply_dics(evoked, filters)
    stc = (stc ** 2).mean()
    assert np.argmax(stc.data) == source_ind

    # Test if wrong channel selection is detected in application of filter
    evoked_ch = cp.deepcopy(evoked)
    evoked_ch.pick_channels(evoked_ch.ch_names[:-1])
    raises(ValueError, apply_dics, evoked_ch, filters)

    # Test whether projections are applied, by adding a custom projection
    filters_noproj = make_dics(evoked.info, fwd_surf, csd20, label=label)
    stc_noproj = apply_dics(evoked, filters_noproj)
    evoked_proj = evoked.copy()
    p = compute_proj_evoked(evoked_proj, n_grad=1, n_mag=0, n_eeg=0)
    proj_matrix = make_projector(p, evoked_proj.ch_names)[0]
    evoked_proj.info['projs'] += p
    filters_proj = make_dics(evoked_proj.info, fwd_surf, csd20, label=label)
    assert_array_equal(filters_proj['proj'], proj_matrix)
    stc_proj = apply_dics(evoked_proj, filters_proj)
    assert np.any(np.not_equal(stc_noproj.data, stc_proj.data))

    # Test detecting incompatible projections
    filters_proj['proj'] = filters_proj['proj'][:-1, :-1]
    raises(ValueError, apply_dics, evoked_proj, filters_proj)

    # Test returning a generator
    stcs = apply_dics_epochs(epochs, filters, return_generator=False)
    stcs_gen = apply_dics_epochs(epochs, filters, return_generator=True)
    assert_array_equal(stcs[0].data, advance_iterator(stcs_gen).data)

    # Test computing timecourses on a volume source space
    filters_vol = make_dics(evoked.info, fwd_vol, csd20, reg=reg)
    stc = apply_dics(evoked, filters_vol)
    stc = (stc ** 2).mean()
    assert np.argmax(stc.data) == 3851  # TODO: don't make this hard coded


@pytest.mark.slowtest
@testing.requires_testing_data
def test_tf_dics():
    """Test 5D time-frequency beamforming based on DICS."""
    fwd_free, fwd_surf, fwd_fixed, fwd_vol, label = _load_forward()
    epochs, evoked, _, source_vertno = _simulate_data(fwd_fixed)
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
        stcs = tf_dics(epochs, fwd_surf, None, tmin, tmax, tstep, win_lengths,
                       mode=mode, freq_bins=freq_bins, frequencies=frequencies,
                       decim=10, reg=reg, label=label)

        # Did we find the true source at 20 Hz?
        assert np.argmax(stcs[1].data[:, 0]) == source_ind
        assert np.argmax(stcs[1].data[:, 1]) == source_ind

        # 20 Hz power should decrease over time
        assert stcs[1].data[source_ind, 0] > stcs[1].data[source_ind, 1]

        # 20 Hz power should be more than 10 Hz power at the true source
        assert stcs[1].data[source_ind, 0] > stcs[0].data[source_ind, 0]

    # Manually compute source power and compare with the last tf_dics result.
    source_power = []
    time_windows = [(0, 5), (4, 9)]
    for time_window in time_windows:
        csd = csd_morlet(epochs, frequencies=[frequencies[1]],
                         tmin=time_window[0], tmax=time_window[1], decim=10)
        csd = csd.sum()
        csd._data /= csd.n_fft
        filters = make_dics(epochs.info, fwd_surf, csd, reg=reg, label=label)
        stc_source_power, _ = apply_dics_csd(csd, filters)
        source_power.append(stc_source_power.data)

    # Comparing tf_dics results with dics_source_power results
    assert_allclose(stcs[1].data, np.array(source_power).squeeze().T, atol=0)

    # Test using noise csds. We're going to use identity matrices. That way,
    # since we're using unit-noise-gain weight normalization, there should be
    # no effect.
    stcs = tf_dics(epochs, fwd_surf, None, tmin, tmax, tstep, win_lengths,
                   mode='cwt_morlet', frequencies=frequencies, decim=10,
                   reg=reg, label=label, normalize_fwd=False,
                   weight_norm='unit-noise-gain')
    noise_csd = csd.copy()
    inds = np.triu_indices(csd.n_channels)
    # Using [:, :] syntax for in-place broadcasting
    noise_csd._data[:, :] = 2 * np.eye(csd.n_channels)[inds][:, np.newaxis]
    noise_csd.n_fft = 2  # Dividing by n_fft should yield an identity CSD
    noise_csds = [noise_csd, noise_csd]  # Two frequency bins
    stcs_norm = tf_dics(epochs, fwd_surf, noise_csds, tmin, tmax, tstep,
                        win_lengths, mode='cwt_morlet',
                        frequencies=frequencies, decim=10, reg=reg,
                        label=label, normalize_fwd=False,
                        weight_norm='unit-noise-gain')
    assert_allclose(stcs_norm[0].data, stcs[0].data, atol=0)
    assert_allclose(stcs_norm[1].data, stcs[1].data, atol=0)

    # Test invalid parameter combinations
    raises(ValueError, tf_dics, epochs, fwd_surf, None, tmin, tmax, tstep,
           win_lengths, mode='fourier', freq_bins=None)
    raises(ValueError, tf_dics, epochs, fwd_surf, None, tmin, tmax, tstep,
           win_lengths, mode='cwt_morlet', frequencies=None)

    # Test if incorrect number of noise CSDs is detected
    raises(ValueError, tf_dics, epochs, fwd_surf, [noise_csds[0]], tmin, tmax,
           tstep, win_lengths, freq_bins=freq_bins)

    # Test if freq_bins and win_lengths incompatibility is detected
    raises(ValueError, tf_dics, epochs, fwd_surf, None, tmin, tmax, tstep,
           win_lengths=[0, 1, 2], freq_bins=freq_bins)

    # Test if time step exceeding window lengths is detected
    raises(ValueError, tf_dics, epochs, fwd_surf, None, tmin, tmax, tstep=0.15,
           win_lengths=[0.2, 0.1], freq_bins=freq_bins)

    # Test if incorrent number of n_ffts is detected
    raises(ValueError, tf_dics, epochs, fwd_surf, None, tmin, tmax, tstep,
           win_lengths, freq_bins=freq_bins, n_ffts=[1])

    # Test if incorrect number of mt_bandwidths is detected
    raises(ValueError, tf_dics, epochs, fwd_surf, None, tmin, tmax, tstep,
           win_lengths=win_lengths, freq_bins=freq_bins, mode='multitaper',
           mt_bandwidths=[20])

    # Test if subtracting evoked responses yields NaN's, since we only have one
    # epoch. Suppress division warnings.
    with warnings.catch_warnings(record=True):
        stcs = tf_dics(epochs, fwd_surf, None, tmin, tmax, tstep, win_lengths,
                       mode='cwt_morlet', frequencies=frequencies,
                       subtract_evoked=True, reg=reg, label=label, decim=20)
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
        data_csd = csd_multitaper(epochs, tmin=0.045, tmax=None, fmin=8,
                                  fmax=12, bandwidth=72.72).sum()
        noise_csd = csd_multitaper(epochs, tmin=None, tmax=0, fmin=8, fmax=12,
                                   bandwidth=72.72).sum()
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
