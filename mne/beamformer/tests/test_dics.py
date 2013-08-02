import os.path as op

from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import mne
from mne.datasets import sample
from mne.beamformer import dics, dics_epochs
from mne.time_frequency import compute_csd


data_path = sample.data_path()
fname_data = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-meg-oct-6-fwd.fif')
fname_fwd_vol = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis-meg-vol-7-fwd.fif')
fname_event = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-eve.fif')
label = 'Aud-lh'
fname_label = op.join(data_path, 'MEG', 'sample', 'labels', '%s.label' % label)

label = mne.read_label(fname_label)
# preloading raw here increases mem requirements by 400 mb for all nosetests
# that include this file's parent directory :(
raw = mne.fiff.Raw(fname_raw, preload=False)
forward = mne.read_forward_solution(fname_fwd)
forward_surf_ori = mne.read_forward_solution(fname_fwd, surf_ori=True)
forward_fixed = mne.read_forward_solution(fname_fwd, force_fixed=True,
                                          surf_ori=True)
forward_vol = mne.read_forward_solution(fname_fwd_vol, surf_ori=True)
events = mne.read_events(fname_event)


def test_dics():
    """Test DICS with evoked data and single trials
    """
    event_id, tmin, tmax = 1, -0.11, 0.15

    # Setup for reading the raw data
    raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

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

    # Computing the data and noise cross-spectral density matrices
    data_csd = compute_csd(epochs, mode='multitaper', tmin=0.04, tmax=None,
                           fmin=8, fmax=12)
    noise_csd = compute_csd(epochs, mode='multitaper', tmin=None, tmax=0.0,
                            fmin=8, fmax=12)

    stc = dics(evoked, forward, noise_csd=noise_csd, data_csd=data_csd)

    stc_pow = np.sum(stc.data, axis=1)
    idx = np.argmax(stc_pow)
    max_stc = stc.data[idx]
    tmax = stc.times[np.argmax(max_stc)]

    assert_true(0.09 < tmax < 0.11)
    assert_true(12 < np.max(max_stc) < 13)  # TODO: Too large?

    # Test picking normal orientation
    stc_normal = dics(evoked, forward_surf_ori, noise_csd, data_csd,
                      pick_ori="normal")

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
    stcs = dics_epochs(epochs, forward_fixed, noise_csd, data_csd, reg=0.01)

    # Testing returning of generator
    stcs_ = dics_epochs(epochs, forward_fixed, noise_csd, data_csd, reg=0.01,
                        return_generator=True)
    assert_array_equal(stcs[0].data, stcs_.next().data)

    # Test whether correct number of trials was returned
    epochs.drop_bad_epochs()
    assert_true(len(epochs.events) == len(stcs))

    # Average the single trial estimates
    stc_avg = np.zeros_like(stc.data)
    for this_stc in stcs:
        stc_avg += this_stc.data
    stc_avg /= len(stcs)

    idx = np.argmax(np.max(stc_avg, axis=1))
    max_stc = stc_avg[idx]
    tmax = stc.times[np.argmax(max_stc)]

    assert_true(0.09 < tmax < 0.11)
    assert_true(15 < np.max(max_stc) < 16)

    # Use a label so we have few source vertices and delayed computation is
    # not used
    stcs_label = dics_epochs(epochs, forward_fixed, noise_csd, data_csd,
                             reg=0.01, label=label)

    assert_array_almost_equal(stcs_label[0].data, stcs[0].in_label(label).data)
