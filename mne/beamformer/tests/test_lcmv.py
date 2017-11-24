from copy import deepcopy
import os.path as op

import pytest
from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_almost_equal)
import warnings

import mne
from mne import compute_covariance
from mne.datasets import testing
from mne.beamformer import (make_lcmv, apply_lcmv, apply_lcmv_raw, lcmv,
                            lcmv_epochs, lcmv_raw, tf_lcmv)
from mne.beamformer._lcmv import _lcmv_source_power, _reg_pinv, _eig_inv
from mne.externals.six import advance_iterator
from mne.utils import run_tests_if_main


data_path = testing.data_path(download=False)
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_fwd_vol = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc-meg-vol-7-fwd.fif')
fname_event = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc_raw-eve.fif')
fname_label = op.join(data_path, 'MEG', 'sample', 'labels', 'Aud-lh.label')

warnings.simplefilter('always')  # enable b/c these tests throw warnings


def _read_forward_solution_meg(*args, **kwargs):
    fwd = mne.read_forward_solution(*args)
    fwd = mne.convert_forward_solution(fwd, **kwargs)
    return mne.pick_types_forward(fwd, meg=True, eeg=False)


def _get_data(tmin=-0.1, tmax=0.15, all_forward=True, epochs=True,
              epochs_preload=True, data_cov=True):
    """Read in data used in tests."""
    label = mne.read_label(fname_label)
    events = mne.read_events(fname_event)
    raw = mne.io.read_raw_fif(fname_raw, preload=True)
    forward = mne.read_forward_solution(fname_fwd)
    if all_forward:
        forward_surf_ori = _read_forward_solution_meg(
            fname_fwd, surf_ori=True)
        forward_fixed = _read_forward_solution_meg(
            fname_fwd, force_fixed=True, surf_ori=True, use_cps=False)
        forward_vol = _read_forward_solution_meg(fname_fwd_vol)
    else:
        forward_surf_ori = None
        forward_fixed = None
        forward_vol = None

    event_id, tmin, tmax = 1, tmin, tmax

    # Setup for reading the raw data
    raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bad channels
    # Set up pick list: MEG - bad channels
    left_temporal_channels = mne.read_selection('Left-temporal')
    picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=True,
                           eog=True, ref_meg=False, exclude='bads',
                           selection=left_temporal_channels)
    raw.pick_channels([raw.ch_names[ii] for ii in picks])
    raw.info.normalize_proj()  # avoid projection warnings

    if epochs:
        # Read epochs
        epochs = mne.Epochs(
            raw, events, event_id, tmin, tmax, proj=True,
            baseline=(None, 0), preload=epochs_preload,
            reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
        if epochs_preload:
            epochs.resample(200, npad=0, n_jobs=2)
        epochs.crop(0, None)
        evoked = epochs.average()
        info = evoked.info
    else:
        epochs = None
        evoked = None
        info = raw.info

    noise_cov = mne.read_cov(fname_cov)
    noise_cov['projs'] = []  # avoid warning
    with warnings.catch_warnings(record=True):  # bad proj
        noise_cov = mne.cov.regularize(noise_cov, info, mag=0.05, grad=0.05,
                                       eeg=0.1, proj=True)
    if data_cov:
        with warnings.catch_warnings(record=True):  # too few samples
            data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.145)
    else:
        data_cov = None

    return raw, epochs, evoked, data_cov, noise_cov, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol


@pytest.mark.slowtest
@testing.requires_testing_data
def test_lcmv():
    """Test LCMV with evoked data and single trials."""
    raw, epochs, evoked, data_cov, noise_cov, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol = _get_data()

    for fwd in [forward, forward_vol]:
        stc = lcmv(evoked, fwd, noise_cov, data_cov, reg=0.01,
                   max_ori_out='signed')
        stc.crop(0.02, None)

        stc_pow = np.sum(np.abs(stc.data), axis=1)
        idx = np.argmax(stc_pow)
        max_stc = stc.data[idx]
        tmax = stc.times[np.argmax(max_stc)]

        assert_true(0.09 < tmax < 0.105, tmax)
        assert_true(0.9 < np.max(max_stc) < 3., np.max(max_stc))

        if fwd is forward:
            # Test picking normal orientation (surface source space only)
            stc_normal = lcmv(evoked, forward_surf_ori, noise_cov,
                              data_cov, reg=0.01, pick_ori="normal",
                              max_ori_out='signed')
            stc_normal.crop(0.02, None)

            stc_pow = np.sum(np.abs(stc_normal.data), axis=1)
            idx = np.argmax(stc_pow)
            max_stc = stc_normal.data[idx]
            tmax = stc_normal.times[np.argmax(max_stc)]

            assert_true(0.04 < tmax < 0.11, tmax)
            assert_true(0.4 < np.max(max_stc) < 2., np.max(max_stc))

            # The amplitude of normal orientation results should always be
            # smaller than free orientation results
            assert_true((np.abs(stc_normal.data) <= stc.data).all())

        # Test picking source orientation maximizing output source power
        stc_max_power = lcmv(evoked, fwd, noise_cov, data_cov, reg=0.01,
                             pick_ori="max-power", max_ori_out='signed')
        stc_max_power.crop(0.02, None)
        stc_pow = np.sum(np.abs(stc_max_power.data), axis=1)
        idx = np.argmax(stc_pow)
        max_stc = np.abs(stc_max_power.data[idx])
        tmax = stc.times[np.argmax(max_stc)]

        assert_true(0.08 < tmax < 0.11, tmax)
        assert_true(0.8 < np.max(max_stc) < 3., np.max(max_stc))

        stc_max_power.data[:, :] = np.abs(stc_max_power.data)

        if fwd is forward:
            # Maximum output source power orientation results should be
            # similar to free orientation results in areas with channel
            # coverage
            label = mne.read_label(fname_label)
            mean_stc = stc.extract_label_time_course(label, fwd['src'],
                                                     mode='mean')
            mean_stc_max_pow = \
                stc_max_power.extract_label_time_course(label, fwd['src'],
                                                        mode='mean')
            assert_true((np.abs(mean_stc - mean_stc_max_pow) < 0.5).all())

        # Test NAI weight normalization:
        stc_nai = lcmv(evoked, fwd, noise_cov=noise_cov, data_cov=data_cov,
                       reg=0.01, pick_ori='max-power', weight_norm='nai',
                       max_ori_out='signed')
        stc_nai.crop(0.02, None)

        # Test whether unit-noise-gain solution is a scaled version of NAI
        pearsoncorr = np.corrcoef(np.concatenate(np.abs(stc_nai.data)),
                                  np.concatenate(stc_max_power.data))
        assert_almost_equal(pearsoncorr[0, 1], 1.)

    # Test sphere head model with unit-noise gain beamformer and orientation
    # selection and rank reduction of the leadfield
    sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.080)
    src = mne.setup_volume_source_space(subject=None, pos=15., mri=None,
                                        sphere=(0.0, 0.0, 0.0, 80.0),
                                        bem=None, mindist=5.0, exclude=2.0)

    fwd_sphere = mne.make_forward_solution(evoked.info, trans=None, src=src,
                                           bem=sphere, eeg=False, meg=True)

    # Test that we get an error is not reducing rank
    assert_raises(ValueError, lcmv, evoked, fwd_sphere, noise_cov, data_cov,
                  reg=0.1, weight_norm='unit-noise-gain', pick_ori="max-power",
                  reduce_rank=False)

    # Now let's reduce it
    stc_sphere = lcmv(evoked, fwd_sphere, noise_cov, data_cov, reg=0.1,
                      weight_norm='unit-noise-gain', pick_ori="max-power",
                      reduce_rank=True, max_ori_out='signed')
    stc_sphere = np.abs(stc_sphere)
    stc_sphere.crop(0.02, None)

    stc_pow = np.sum(stc_sphere.data, axis=1)
    idx = np.argmax(stc_pow)
    max_stc = stc_sphere.data[idx]
    tmax = stc_sphere.times[np.argmax(max_stc)]

    assert_true(0.08 < tmax < 0.11, tmax)
    assert_true(0.4 < np.max(max_stc) < 2., np.max(max_stc))

    # Test if fixed forward operator is detected when picking normal or
    # max-power orientation
    assert_raises(ValueError, lcmv, evoked, forward_fixed, noise_cov, data_cov,
                  reg=0.01, pick_ori="normal")
    assert_raises(ValueError, lcmv, evoked, forward_fixed, noise_cov, data_cov,
                  reg=0.01, pick_ori="max-power", max_ori_out='signed')

    # Test if non-surface oriented forward operator is detected when picking
    # normal orientation
    assert_raises(ValueError, lcmv, evoked, forward, noise_cov, data_cov,
                  reg=0.01, pick_ori="normal")

    # Test if volume forward operator is detected when picking normal
    # orientation
    assert_raises(ValueError, lcmv, evoked, forward_vol, noise_cov, data_cov,
                  reg=0.01, pick_ori="normal")

    # Test if missing of data covariance matrix is detected
    assert_raises(ValueError, lcmv, evoked, forward_vol, noise_cov=noise_cov,
                  data_cov=None, reg=0.01, pick_ori="max-power",
                  max_ori_out='signed')

    # Test if missing of noise covariance matrix is detected when more than
    # one channel type is present in the data
    assert_raises(ValueError, lcmv, evoked, forward_vol, noise_cov=None,
                  data_cov=data_cov, reg=0.01, pick_ori="max-power",
                  max_ori_out='signed')

    # Test if not-yet-implemented orientation selections raise error with
    # neural activity index
    assert_raises(NotImplementedError, lcmv, evoked, forward_surf_ori,
                  noise_cov, data_cov, reg=0.01, pick_ori="normal",
                  weight_norm='nai')
    assert_raises(NotImplementedError, lcmv, evoked, forward_vol, noise_cov,
                  data_cov, reg=0.01, pick_ori=None, weight_norm='nai')

    # Test if no weight-normalization and max-power source orientation throw
    # an error
    assert_raises(NotImplementedError, lcmv, evoked, forward_vol, noise_cov,
                  data_cov, reg=0.01, pick_ori="max-power", weight_norm=None,
                  max_ori_out='signed')

    # Test if wrong channel selection is detected in application of filter
    evoked_ch = deepcopy(evoked)
    evoked_ch.pick_channels(evoked_ch.ch_names[1:])
    filters = make_lcmv(evoked.info, forward_vol, data_cov, reg=0.01,
                        noise_cov=noise_cov)
    assert_raises(ValueError, apply_lcmv, evoked_ch, filters,
                  max_ori_out='signed')

    # Test if discrepancies in channel selection of data and fwd model are
    # handled correctly in apply_lcmv
    # make filter with data where first channel was removed
    filters = make_lcmv(evoked_ch.info, forward_vol, data_cov, reg=0.01,
                        noise_cov=noise_cov)
    # applying that filter to the full data set should automatically exclude
    # this channel from the data
    stc = apply_lcmv(evoked, filters, max_ori_out='signed')
    # the result should be equal to applying this filter to a dataset without
    # this channel:
    stc_ch = apply_lcmv(evoked_ch, filters, max_ori_out='signed')
    assert_array_almost_equal(stc.data, stc_ch.data)

    # Test if non-matching SSP projection is detected in application of filter
    raw_proj = deepcopy(raw)
    raw_proj.del_proj()
    assert_raises(ValueError, apply_lcmv_raw, raw_proj, filters,
                  max_ori_out='signed')

    # Test if setting reduce_rank to True returns a NotImplementedError
    # when no orientation selection is done or pick_ori='normal'
    assert_raises(NotImplementedError, lcmv, evoked, forward_vol, noise_cov,
                  data_cov, pick_ori=None, weight_norm='nai', reduce_rank=True,
                  max_ori_out='signed')
    assert_raises(NotImplementedError, lcmv, evoked, forward_surf_ori,
                  noise_cov, data_cov, pick_ori='normal', weight_norm='nai',
                  reduce_rank=True, max_ori_out='signed')

    # Now test single trial using fixed orientation forward solution
    # so we can compare it to the evoked solution
    stcs = lcmv_epochs(epochs, forward_fixed, noise_cov, data_cov,
                       reg=0.01, max_ori_out='signed')
    stcs_ = lcmv_epochs(epochs, forward_fixed, noise_cov, data_cov,
                        reg=0.01, return_generator=True,
                        max_ori_out='signed')
    assert_array_equal(stcs[0].data, advance_iterator(stcs_).data)

    epochs.drop_bad()
    assert_true(len(epochs.events) == len(stcs))

    # average the single trial estimates
    stc_avg = np.zeros_like(stcs[0].data)
    for this_stc in stcs:
        stc_avg += this_stc.data
    stc_avg /= len(stcs)

    # compare it to the solution using evoked with fixed orientation
    stc_fixed = lcmv(evoked, forward_fixed, noise_cov, data_cov, reg=0.01,
                     max_ori_out='signed')
    assert_array_almost_equal(stc_avg, stc_fixed.data)

    # use a label so we have few source vertices and delayed computation is
    # not used
    stcs_label = lcmv_epochs(epochs, forward_fixed, noise_cov, data_cov,
                             reg=0.01, label=label, max_ori_out='signed')

    assert_array_almost_equal(stcs_label[0].data, stcs[0].in_label(label).data)


@testing.requires_testing_data
def test_lcmv_raw():
    """Test LCMV with raw data."""
    raw, _, _, _, noise_cov, label, forward, _, _, _ =\
        _get_data(all_forward=False, epochs=False, data_cov=False)

    tmin, tmax = 0, 20
    start, stop = raw.time_as_index([tmin, tmax])

    # use only the left-temporal MEG channels for LCMV
    data_cov = mne.compute_raw_covariance(raw, tmin=tmin, tmax=tmax)
    stc = lcmv_raw(raw, forward, noise_cov, data_cov, reg=0.01,
                   label=label, start=start, stop=stop,
                   max_ori_out='signed')

    assert_array_almost_equal(np.array([tmin, tmax]),
                              np.array([stc.times[0], stc.times[-1]]),
                              decimal=2)

    # make sure we get an stc with vertices only in the lh
    vertno = [forward['src'][0]['vertno'], forward['src'][1]['vertno']]
    assert_true(len(stc.vertices[0]) == len(np.intersect1d(vertno[0],
                                                           label.vertices)))
    assert_true(len(stc.vertices[1]) == 0)


@testing.requires_testing_data
def test_lcmv_source_power():
    """Test LCMV source power computation."""
    raw, epochs, evoked, data_cov, noise_cov, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol = _get_data()

    stc_source_power = _lcmv_source_power(epochs.info, forward, noise_cov,
                                          data_cov, label=label,
                                          weight_norm='unit-noise-gain')

    max_source_idx = np.argmax(stc_source_power.data)
    max_source_power = np.max(stc_source_power.data)

    assert_true(max_source_idx == 0, max_source_idx)
    assert_true(0.4 < max_source_power < 2.4, max_source_power)

    # Test picking normal orientation and using a list of CSD matrices
    stc_normal = _lcmv_source_power(
        epochs.info, forward_surf_ori, noise_cov, data_cov,
        pick_ori="normal", label=label, weight_norm='unit-noise-gain')

    # The normal orientation results should always be smaller than free
    # orientation results
    assert_true((np.abs(stc_normal.data[:, 0]) <=
                 stc_source_power.data[:, 0]).all())

    # Test if fixed forward operator is detected when picking normal
    # orientation
    assert_raises(ValueError, _lcmv_source_power, raw.info, forward_fixed,
                  noise_cov, data_cov, pick_ori="normal")

    # Test if non-surface oriented forward operator is detected when picking
    # normal orientation
    assert_raises(ValueError, _lcmv_source_power, raw.info, forward, noise_cov,
                  data_cov, pick_ori="normal")

    # Test if volume forward operator is detected when picking normal
    # orientation
    assert_raises(ValueError, _lcmv_source_power, epochs.info, forward_vol,
                  noise_cov, data_cov, pick_ori="normal")


@testing.requires_testing_data
def test_tf_lcmv():
    """Test TF beamforming based on LCMV."""
    label = mne.read_label(fname_label)
    events = mne.read_events(fname_event)
    raw = mne.io.read_raw_fif(fname_raw, preload=True)
    forward = mne.read_forward_solution(fname_fwd)

    event_id, tmin, tmax = 1, -0.2, 0.2

    # Setup for reading the raw data
    raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

    # Set up pick list: MEG - bad channels
    left_temporal_channels = mne.read_selection('Left-temporal')
    picks = mne.pick_types(raw.info, meg=True, eeg=False,
                           stim=True, eog=True, exclude='bads',
                           selection=left_temporal_channels)
    raw.pick_channels([raw.ch_names[ii] for ii in picks])
    raw.info.normalize_proj()  # avoid projection warnings
    del picks

    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        baseline=None, preload=False,
                        reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
    epochs.load_data()

    freq_bins = [(4, 12), (15, 40)]
    time_windows = [(-0.1, 0.1), (0.0, 0.2)]
    win_lengths = [0.2, 0.2]
    tstep = 0.1
    reg = 0.05

    source_power = []
    noise_covs = []
    for (l_freq, h_freq), win_length in zip(freq_bins, win_lengths):
        raw_band = raw.copy()
        raw_band.filter(l_freq, h_freq, method='iir', n_jobs=1,
                        iir_params=dict(output='ba'))
        epochs_band = mne.Epochs(
            raw_band, epochs.events, epochs.event_id, tmin=tmin, tmax=tmax,
            baseline=None, proj=True)
        with warnings.catch_warnings(record=True):  # not enough samples
            noise_cov = compute_covariance(epochs_band, tmin=tmin, tmax=tmin +
                                           win_length)
        noise_cov = mne.cov.regularize(
            noise_cov, epochs_band.info, mag=reg, grad=reg, eeg=reg,
            proj=True)
        noise_covs.append(noise_cov)
        del raw_band  # to save memory

        # Manually calculating source power in on frequency band and several
        # time windows to compare to tf_lcmv results and test overlapping
        if (l_freq, h_freq) == freq_bins[0]:
            for time_window in time_windows:
                with warnings.catch_warnings(record=True):  # bad samples
                    data_cov = compute_covariance(epochs_band,
                                                  tmin=time_window[0],
                                                  tmax=time_window[1])
                with warnings.catch_warnings(record=True):  # bad proj
                    stc_source_power = _lcmv_source_power(
                        epochs.info, forward, noise_cov, data_cov,
                        reg=reg, label=label, weight_norm='unit-noise-gain')
                source_power.append(stc_source_power.data)

    assert_raises(ValueError, tf_lcmv, epochs, forward, noise_covs, tmin, tmax,
                  tstep, win_lengths, freq_bins, reg=reg, label=label)
    stcs = tf_lcmv(epochs, forward, noise_covs, tmin, tmax, tstep,
                   win_lengths, freq_bins, reg=reg, label=label, raw=raw)

    assert_true(len(stcs) == len(freq_bins))
    assert_true(stcs[0].shape[1] == 4)

    # Averaging all time windows that overlap the time period 0 to 100 ms
    source_power = np.mean(source_power, axis=0)

    # Selecting the first frequency bin in tf_lcmv results
    stc = stcs[0]

    # Comparing tf_lcmv results with _lcmv_source_power results
    assert_array_almost_equal(stc.data[:, 2], source_power[:, 0])

    # Test if using unsupported max-power orientation is detected
    assert_raises(ValueError, tf_lcmv, epochs, forward, noise_covs, tmin, tmax,
                  tstep, win_lengths, freq_bins=freq_bins,
                  pick_ori='max-power')

    # Test if incorrect number of noise CSDs is detected
    # Test if incorrect number of noise covariances is detected
    assert_raises(ValueError, tf_lcmv, epochs, forward, [noise_covs[0]], tmin,
                  tmax, tstep, win_lengths, freq_bins)

    # Test if freq_bins and win_lengths incompatibility is detected
    assert_raises(ValueError, tf_lcmv, epochs, forward, noise_covs, tmin, tmax,
                  tstep, win_lengths=[0, 1, 2], freq_bins=freq_bins)

    # Test if time step exceeding window lengths is detected
    assert_raises(ValueError, tf_lcmv, epochs, forward, noise_covs, tmin, tmax,
                  tstep=0.15, win_lengths=[0.2, 0.1], freq_bins=freq_bins)

    # Test if missing of noise covariance matrix is detected when more than
    # one channel type is present in the data
    assert_raises(ValueError, tf_lcmv, epochs, forward, noise_covs=None,
                  tmin=tmin, tmax=tmax, tstep=tstep, win_lengths=win_lengths,
                  freq_bins=freq_bins)

    # Test if unsupported weight normalization specification is detected
    assert_raises(ValueError, tf_lcmv, epochs, forward, noise_covs, tmin, tmax,
                  tstep, win_lengths, freq_bins, weight_norm='nai')

    # Test correct detection of preloaded epochs objects that do not contain
    # the underlying raw object
    epochs_preloaded = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                                  baseline=(None, 0), preload=True)
    epochs_preloaded._raw = None
    with warnings.catch_warnings(record=True):  # not enough samples
        assert_raises(ValueError, tf_lcmv, epochs_preloaded, forward,
                      noise_covs, tmin, tmax, tstep, win_lengths, freq_bins)

    with warnings.catch_warnings(record=True):  # not enough samples
        # Pass only one epoch to test if subtracting evoked
        # responses yields zeros
        stcs = tf_lcmv(epochs[0], forward, noise_covs, tmin, tmax, tstep,
                       win_lengths, freq_bins, subtract_evoked=True, reg=reg,
                       label=label, raw=raw)

    assert_array_almost_equal(stcs[0].data, np.zeros_like(stcs[0].data))


def test_reg_pinv():
    """Test regularization and inversion of covariance matrix."""
    # create rank-deficient array
    a = np.array([[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]])

    # Test if rank-deficient matrix without regularization throws
    # specific warning
    with warnings.catch_warnings(record=True) as w:
        _reg_pinv(a, reg=0.)
    assert_true(any('deficient' in str(ww.message) for ww in w))


def test_eig_inv():
    """Test matrix pseudoinversion with setting smallest eigenvalue to zero."""
    # create rank-deficient array
    a = np.array([[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]])

    # test inversion
    a_inv = np.linalg.pinv(a)
    a_inv_eig = _eig_inv(a, 2)

    assert_almost_equal(a_inv, a_inv_eig)


run_tests_if_main()
