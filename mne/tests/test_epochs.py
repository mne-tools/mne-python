# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os.path as op

import mne
from mne import fiff

raw_fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                     'test_raw.fif')
event_name = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                     'test-eve.fif')


def test_read_epochs():
    event_id = 1
    tmin = -0.2
    tmax = 0.5

    # Setup for reading the raw data
    raw = fiff.Raw(raw_fname)
    events = mne.read_events(event_name)
    picks = fiff.pick_types(raw.info, meg=True, eeg=False, stim=False,
                            eog=True, include=['STI 014'])
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0))
    epochs.average()


def test_reject_epochs():
    event_id = 1
    tmin = -0.2
    tmax = 0.5

    # Setup for reading the raw data
    raw = fiff.Raw(raw_fname)
    events = mne.read_events(event_name)

    picks = fiff.pick_types(raw.info, meg=True, eeg=True, stim=True,
                            eog=True, include=['STI 014'])
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0))
    n_epochs = len(epochs)
    epochs.reject(grad=1000e-12, mag=4e-12, eeg=80e-6, eog=150e-6)
    n_clean_epochs = len(epochs)
    assert n_epochs > n_clean_epochs
    assert n_clean_epochs == 3
