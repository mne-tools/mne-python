import os.path as op

from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

import mne
from mne.datasets import sample
from mne.beamformer import lcmv, lcmv_epochs, lcmv_raw
from mne.beamformer._lcmv import _lcmv_source_power
from mne.source_estimate import SourceEstimate, VolSourceEstimate


data_path = sample.data_path()
fname_data = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-meg-oct-6-fwd.fif')
fname_fwd_vol = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis-meg-vol-7-fwd.fif')
fname_event = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-eve.fif')
label = 'Aud-lh'
fname_label = op.join(data_path, 'MEG', 'sample', 'labels', '%s.label' % label)


def _get_data(tmin=-0.1, tmax=0.15, all_forward=True, epochs=True,
              data_cov=True):
    """Read in data used in tests
    """
    label = mne.read_label(fname_label)
    events = mne.read_events(fname_event)
    raw = mne.fiff.Raw(fname_raw, preload=False)
    forward = mne.read_forward_solution(fname_fwd)
    if all_forward:
        forward_surf_ori = mne.read_forward_solution(fname_fwd, surf_ori=True)
        forward_fixed = mne.read_forward_solution(fname_fwd, force_fixed=True,
                                                  surf_ori=True)
        forward_vol = mne.read_forward_solution(fname_fwd_vol, surf_ori=True)
    else:
        forward_surf_ori = None
        forward_fixed = None
        forward_vol = None

    event_id, tmin, tmax = 1, tmin, tmax

    # Setup for reading the raw data
    raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

    if epochs:
        # Set up pick list: MEG - bad channels
        left_temporal_channels = mne.read_selection('Left-temporal')
        picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False,
                                    stim=True, eog=True, exclude='bads',
                                    selection=left_temporal_channels)

        # Read epochs
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                            picks=picks, baseline=(None, 0), preload=True,
                            reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
        epochs.resample(200, npad=0, n_jobs=2)
        evoked = epochs.average()
        info = evoked.info
    else:
        epochs = None
        evoked = None
        info = raw.info

    noise_cov = mne.read_cov(fname_cov)
    noise_cov = mne.cov.regularize(noise_cov, info,
                                   mag=0.05, grad=0.05, eeg=0.1, proj=True)
    if data_cov:
        data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.15)
    else:
        data_cov = None

    return raw, epochs, evoked, data_cov, noise_cov, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol


def test_lcmv():
    """Test LCMV with evoked data and single trials
    """
    raw, epochs, evoked, data_cov, noise_cov, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol = _get_data()

    for fwd in [forward, forward_vol]:
        stc = lcmv(evoked, fwd, noise_cov, data_cov, reg=0.01)

        if fwd is forward:
            assert_true(isinstance(stc, SourceEstimate))
        else:
            assert_true(isinstance(stc, VolSourceEstimate))

        stc_pow = np.sum(stc.data, axis=1)
        idx = np.argmax(stc_pow)
        max_stc = stc.data[idx]
        tmax = stc.times[np.argmax(max_stc)]

        assert_true(0.09 < tmax < 0.105)
        assert_true(1.9 < np.max(max_stc) < 3.)

        if fwd is forward:
            # Test picking normal orientation (surface source space only)
            stc_normal = lcmv(evoked, forward_surf_ori, noise_cov, data_cov,
                              reg=0.01, pick_ori="normal")

            stc_pow = np.sum(np.abs(stc_normal.data), axis=1)
            idx = np.argmax(stc_pow)
            max_stc = stc_normal.data[idx]
            tmax = stc_normal.times[np.argmax(max_stc)]

            assert_true(0.09 < tmax < 0.11)
            assert_true(1. < np.max(max_stc) < 2.)

            # The amplitude of normal orientation results should always be
            # smaller than free orientation results
            assert_true((np.abs(stc_normal.data) <= stc.data).all())

        # Test picking source orientation maximizing output source power
        stc_max_power = lcmv(evoked, fwd, noise_cov, data_cov, reg=0.01,
                             pick_ori="max-power")
        stc_pow = np.sum(stc_max_power.data, axis=1)
        idx = np.argmax(stc_pow)
        max_stc = stc_max_power.data[idx]
        tmax = stc.times[np.argmax(max_stc)]

        assert_true(0.09 < tmax < 0.1)
        assert_true(2. < np.max(max_stc) < 3.)

        # Maximum output source power orientation results should be similar to
        # free orientation results
        assert_true((stc_max_power.data - stc.data < 0.5).all())

    # Test if fixed forward operator is detected when picking normal or
    # max-power orientation
    assert_raises(ValueError, lcmv, evoked, forward_fixed, noise_cov, data_cov,
                  reg=0.01, pick_ori="normal")
    assert_raises(ValueError, lcmv, evoked, forward_fixed, noise_cov, data_cov,
                  reg=0.01, pick_ori="max-power")

    # Test if non-surface oriented forward operator is detected when picking
    # normal orientation
    assert_raises(ValueError, lcmv, evoked, forward, noise_cov, data_cov,
                  reg=0.01, pick_ori="normal")

    # Test if volume forward operator is detected when picking normal
    # orientation
    assert_raises(ValueError, lcmv, evoked, forward_vol, noise_cov, data_cov,
                  reg=0.01, pick_ori="normal")

    # Now test single trial using fixed orientation forward solution
    # so we can compare it to the evoked solution
    stcs = lcmv_epochs(epochs, forward_fixed, noise_cov, data_cov, reg=0.01)
    stcs_ = lcmv_epochs(epochs, forward_fixed, noise_cov, data_cov, reg=0.01,
                        return_generator=True)
    assert_array_equal(stcs[0].data, stcs_.next().data)

    epochs.drop_bad_epochs()
    assert_true(len(epochs.events) == len(stcs))

    # average the single trial estimates
    stc_avg = np.zeros_like(stcs[0].data)
    for this_stc in stcs:
        stc_avg += this_stc.data
    stc_avg /= len(stcs)

    # compare it to the solution using evoked with fixed orientation
    stc_fixed = lcmv(evoked, forward_fixed, noise_cov, data_cov, reg=0.01)
    assert_array_almost_equal(stc_avg, stc_fixed.data)

    # use a label so we have few source vertices and delayed computation is
    # not used
    stcs_label = lcmv_epochs(epochs, forward_fixed, noise_cov, data_cov,
                             reg=0.01, label=label)

    assert_array_almost_equal(stcs_label[0].data, stcs[0].in_label(label).data)


def test_lcmv_raw():
    """Test LCMV with raw data
    """
    raw, _, _, _, noise_cov, label, forward, _, _, _ =\
        _get_data(all_forward=False, epochs=False, data_cov=False)

    tmin, tmax = 0, 20
    start, stop = raw.time_as_index([tmin, tmax])

    # use only the left-temporal MEG channels for LCMV
    left_temporal_channels = mne.read_selection('Left-temporal')
    picks = mne.fiff.pick_types(raw.info, meg=True, exclude='bads',
                                selection=left_temporal_channels)

    data_cov = mne.compute_raw_data_covariance(raw, tmin=tmin, tmax=tmax)

    stc = lcmv_raw(raw, forward, noise_cov, data_cov, reg=0.01, label=label,
                   start=start, stop=stop, picks=picks)

    assert_array_almost_equal(np.array([tmin, tmax]),
                              np.array([stc.times[0], stc.times[-1]]),
                              decimal=2)

    # make sure we get an stc with vertices only in the lh
    vertno = [forward['src'][0]['vertno'], forward['src'][1]['vertno']]
    assert_true(len(stc.vertno[0]) == len(np.intersect1d(vertno[0],
                                                         label.vertices)))
    assert_true(len(stc.vertno[1]) == 0)
    # TODO: test more things


def test_lcmv_source_power():
    """Test LCMV source power computation
    """
    raw, epochs, evoked, data_cov, noise_cov, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol = _get_data()

    stc_source_power = _lcmv_source_power(epochs.info, forward, noise_cov,
                                          data_cov, label=label)

    max_source_idx = np.argmax(stc_source_power.data)
    max_source_power = np.max(stc_source_power.data)

    # TODO: Try a direct comparison to results of lcmv
    assert_true(max_source_idx == 24)
    assert_true(2.2 < max_source_power < 2.4)

    # Test picking normal orientation and using a list of CSD matrices
    stc_normal = _lcmv_source_power(epochs.info, forward_surf_ori, noise_cov,
                                    data_cov, pick_ori="normal", label=label)

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
