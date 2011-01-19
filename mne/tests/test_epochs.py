# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import os
import os.path as op

import mne
from mne import fiff

MNE_SAMPLE_DATASET_PATH = os.getenv('MNE_SAMPLE_DATASET_PATH')
raw_fname = op.join(MNE_SAMPLE_DATASET_PATH, 'MEG', 'sample',
                                                'sample_audvis_raw.fif')
event_name = op.join(MNE_SAMPLE_DATASET_PATH, 'MEG', 'sample',
                                                'sample_audvis_raw-eve.fif')


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

    data, times, channel_names = mne.read_epochs(raw, events, event_id,
                                                    tmin, tmax, picks=picks)
