import os.path as op

from nose.tools import assert_true
import numpy as np
# from numpy.testing import assert_array_almost_equal, assert_equal

import mne
from mne.datasets import sample
from mne.beamformer import lcmv


examples_folder = op.join(op.dirname(__file__), '..', '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname_data = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis-ave.fif')
fname_raw = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_raw.fif')
fname_cov = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis-meg-oct-6-fwd.fif')
fname_event = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_raw-eve.fif')
label = 'Aud-lh'
fname_label = op.join(data_path, 'MEG', 'sample', 'labels', '%s.label' % label)

label = mne.read_label(fname_label)
noise_cov = mne.read_cov(fname_cov)
raw = mne.fiff.Raw(fname_raw)
forward = mne.read_forward_solution(fname_fwd)
events = mne.read_events(fname_event)


def test_lcmv():
    """Test LCMV
    """
    event_id, tmin, tmax = 1, -0.2, 0.2

    # Setup for reading the raw data
    raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

    # Set up pick list: EEG + MEG - bad channels (modify to your needs)
    left_temporal_channels = mne.read_selection('Left-temporal')
    picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, stim=True, eog=True,
                       exclude=raw.info['bads'], selection=left_temporal_channels)

    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks, baseline=(None, 0), preload=True,
                        reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
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
