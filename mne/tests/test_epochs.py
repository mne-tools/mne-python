# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os
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

    #   Setup for reading the raw data
    raw = fiff.setup_read_raw(raw_fname)
    events = mne.read_events(event_name)

    # Set up pick list: MEG + STI 014 - bad channels (modify to your needs)
    include = ['STI 014'];
    want_meg = True
    want_eeg = False
    want_stim = False
    picks = fiff.pick_types(raw['info'], want_meg, want_eeg, want_stim,
                            include, raw['info']['bads'])

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0))
