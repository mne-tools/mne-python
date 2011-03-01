import numpy as np
import os.path as op

from numpy.testing import assert_allclose

import mne
from mne import fiff
from mne import time_frequency

raw_fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                'test_raw.fif')
event_fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                'test-eve.fif')

def test_time_frequency():
    """Test IO for STC files
    """
    # Set parameters
    event_id = 1
    tmin = -0.2
    tmax = 0.5

    # Setup for reading the raw data
    raw = fiff.setup_read_raw(raw_fname)
    events = mne.read_events(event_fname)

    include = []
    exclude = raw['info']['bads'] + ['MEG 2443', 'EEG 053'] # bads + 2 more

    # picks MEG gradiometers
    picks = fiff.pick_types(raw['info'], meg='grad', eeg=False,
                                    stim=False, include=include, exclude=exclude)

    picks = picks[:2]
    data, times, channel_names = mne.read_epochs(raw, events, event_id,
                                    tmin, tmax, picks=picks, baseline=(None, 0))
    epochs = np.array([d['epoch'] for d in data]) # as 3D matrix
    evoked_data = np.mean(epochs, axis=0) # compute evoked fields

    frequencies = np.arange(4, 20, 5) # define frequencies of interest
    Fs = raw['info']['sfreq'] # sampling in Hz
    power, phase_lock = time_frequency(epochs, Fs=Fs, frequencies=frequencies,
                                       n_cycles=2, use_fft=True)

    assert power.shape == (len(picks), len(frequencies), len(times))
    assert power.shape == phase_lock.shape
    assert np.sum(phase_lock >= 1) == 0
    assert np.sum(phase_lock <= 0) == 0

    power, phase_lock = time_frequency(epochs, Fs=Fs, frequencies=frequencies,
                                       n_cycles=2, use_fft=False)

    assert power.shape == (len(picks), len(frequencies), len(times))
    assert power.shape == phase_lock.shape
    assert np.sum(phase_lock >= 1) == 0
    assert np.sum(phase_lock <= 0) == 0
    