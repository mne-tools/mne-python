import os.path as op

from nose.tools import assert_true, assert_raises
import numpy as np

import mne
from mne.datasets import sample
from mne.beamformer import dics_epochs
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


def test_dics_epochs():
    """Test DICS with single trials
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
    #evoked = epochs.average()

    # Computing the data and noise cross-spectral density matrices
    data_csd = compute_csd(epochs, mode='multitaper', tmin=0.04, tmax=None,
                           fmin=8, fmax=12)
    noise_csd = compute_csd(epochs, mode='multitaper', tmin=None, tmax=0.0,
                            fmin=8, fmax=12)

    # TODO: This should be done on evoked
    stcs = dics_epochs(epochs, forward, noise_csd=None, data_csd=data_csd,
                       return_generator=True)
    stc = stcs.next()

    stc_pow = np.sum(stc.data, axis=1)
    idx = np.argmax(stc_pow)
    max_stc = stc.data[idx]
    tmax = stc.times[np.argmax(max_stc)]

    # TODO: These should be made reasonable once normalization is implemented
    assert_true(-1 < tmax < 1)
    assert_true(0. < np.max(max_stc) < 20.)

    # Test picking normal orientation
    # TODO: This should be done on evoked
    stcs = dics_epochs(epochs, forward_surf_ori, noise_csd, data_csd,
                       pick_ori="normal", return_generator=True)
    stc_normal = stcs.next()

    # The amplitude of normal orientation results should always be smaller than
    # free orientation results
    assert_true((np.abs(stc_normal.data) <= stc.data).all())

    # TODO: dics_epochs would best be tested by comparing to dics done on
    # evoked data

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
