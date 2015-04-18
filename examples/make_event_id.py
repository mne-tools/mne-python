"""
===========================
Make an event id for Epochs
===========================

Combine triggers into a single event id while maintaining their value.
"""
# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
tmin, tmax = -1, .2

# Reading events
events = mne.read_events(fname)

# Combine triggers across
event_id = {'aud-l': 1,
            'aud-r': 2,
            'vis-l': 3,
            'vis-r': 4,
            'aud': [1, 2],
            'vis': [3, 4]}

# Make Raw instance
raw = mne.io.read_raw_fif(raw_fname)

# Make Epochs instance
epochs = mne.Epochs(raw, events, event_id, tmin, tmax)

print epochs
