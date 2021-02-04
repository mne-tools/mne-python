import os.path as op

import numpy as np

import mne
from mne import io, pick_types
from mne.connectivity import spectral_connectivity
from mne.datasets import testing


data_path = testing.data_path(download=False)
fname_raw_testing = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_trunc_raw.fif')

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_fname = op.join(base_dir, 'test-eve.fif')

raw = io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)
picks = pick_types(raw.info, meg=True, eeg=True, stim=True, ecg=True,
                   eog=True, include=['STI 014'], exclude='bads')

reject = dict(grad=1000e-12, mag=4e-12, eeg=80e-6, eog=150e-6)
flat = dict(grad=1e-15, mag=1e-15)


def test_spectral_connectivity():
    """Test spectral.spectral_connectivity with different data objects"""
    event_id, tmin, tmax = 1, -1, 1
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), reject=reject)
    epochs.load_data().pick_types(meg='grad')

    # Parameters for computing connectivity
    fmin, fmax = 8., 13.
    sfreq = raw.info['sfreq']
    tmin, tmax = -0.9, -0.2

    spectral_connectivity(epochs, method='pli', mode='multitaper',
                          sfreq=sfreq, fmin=fmin, fmax=fmax,
                          faverage=True, tmin=tmin, tmax=tmax,
                          mt_adaptive=False, n_jobs=1)

    # Test with a numpy array
    myarray = np.random.rand(3, 203, raw.n_times)
    tmin, tmax = 0.1, 0.8
    spectral_connectivity(myarray, method='pli', mode='multitaper',
                          sfreq=sfreq, fmin=fmin, fmax=fmax,
                          faverage=True, tmin=tmin, tmax=tmax,
                          mt_adaptive=False, n_jobs=1)
