import os.path as op

from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

import mne
from mne.datasets import sample
from mne.beamformer import lcmv, lcmv_epochs, lcmv_raw


data_path = sample.data_path()
fname_data = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis-ave.fif')
fname_raw = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_raw.fif')
fname_cov = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis-meg-oct-6-fwd.fif')
fname_fwd_vol = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis-meg-vol-7-fwd.fif')
fname_event = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_raw-eve.fif')
label = 'Aud-lh'
fname_label = op.join(data_path, 'MEG', 'sample', 'labels', '%s.label' % label)

label = mne.read_label(fname_label)
noise_cov = mne.read_cov(fname_cov)
# preloading raw here increases mem requirements by 400 mb for all nosetests
# that include this file's parent directory :(
raw = mne.fiff.Raw(fname_raw, preload=False)
forward = mne.read_forward_solution(fname_fwd)
forward_surf_ori = mne.read_forward_solution(fname_fwd, surf_ori=True)
forward_fixed = mne.read_forward_solution(fname_fwd, force_fixed=True,
                                          surf_ori=True)
forward_vol = mne.read_forward_solution(fname_fwd_vol, surf_ori=True)
events = mne.read_events(fname_event)


def test_lcmv():
    """Test LCMV with evoked data and single trials
    """
    event_id, tmin, tmax = 1, -0.1, 0.15

    # Setup for reading the raw data
    raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

    # Set up pick list: EEG + MEG - bad channels (modify to your needs)
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

    noise_cov = mne.read_cov(fname_cov)
    noise_cov = mne.cov.regularize(noise_cov, evoked.info,
                                   mag=0.05, grad=0.05, eeg=0.1, proj=True)

    data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.15)
    stc = lcmv(evoked, forward, noise_cov, data_cov, reg=0.01)

    stc_pow = np.sum(stc.data, axis=1)
    idx = np.argmax(stc_pow)
    max_stc = stc.data[idx]
    tmax = stc.times[np.argmax(max_stc)]

    assert_true(0.09 < tmax < 0.1)
    assert_true(2. < np.max(max_stc) < 3.)

    # Test picking source orientation maximizing output source power
    stc_max_power = lcmv(evoked, forward, noise_cov, data_cov, reg=0.01,
                         pick_ori="max-power")

    assert_true((np.abs(stc_max_power.data) <= stc.data + 1).all())

    # Test if fixed forward operator is detected when picking
    # max-power orientation
    assert_raises(ValueError, lcmv, evoked, forward_fixed, noise_cov, data_cov,
                  reg=0.01, pick_ori="max-power")

    # Now test single trial using fixed orientation forward solution
    # so we can compare it to the evoked solution
    stcs = lcmv_epochs(epochs, forward_fixed, noise_cov, data_cov, reg=0.01)
    stcs_ = lcmv_epochs(epochs, forward_fixed, noise_cov, data_cov, reg=0.01,
                        return_generator=True)
    assert_array_equal(stcs[0].data, stcs_.next().data)

    epochs.drop_bad_epochs()
    assert_true(len(epochs.events) == len(stcs))

    # average the single trial estimates
    stc_avg = np.zeros_like(stc.data)
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
    tmin, tmax = 0, 20
    # Setup for reading the raw data
    raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

    # Set up pick list: EEG + MEG - bad channels (modify to your needs)
    left_temporal_channels = mne.read_selection('Left-temporal')
    picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, stim=True,
                                eog=True, exclude='bads',
                                selection=left_temporal_channels)

    noise_cov = mne.read_cov(fname_cov)
    noise_cov = mne.cov.regularize(noise_cov, raw.info,
                                   mag=0.05, grad=0.05, eeg=0.1, proj=True)

    start, stop = raw.time_as_index([tmin, tmax])

    # use only the left-temporal MEG channels for LCMV
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
