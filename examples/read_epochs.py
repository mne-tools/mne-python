"""
==================================
Reading epochs from a raw FIF file
==================================
"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

print __doc__

import mne
from mne import fiff

###############################################################################
# Set parameters
raw_fname = 'MNE-sample-data/MEG/sample/sample_audvis_raw.fif'
event_name = 'MNE-sample-data/MEG/sample/sample_audvis_raw-eve.fif'
event_id = 1
tmin = -0.2
tmax = 0.5
pick_all = True

#   Setup for reading the raw data
raw = fiff.setup_read_raw(raw_fname)
events = mne.read_events(event_name)

if pick_all:
   # Pick all
   picks = range(raw['info']['nchan'])
else:
   #   Set up pick list: MEG + STI 014 - bad channels (modify to your needs)
   include = ['STI 014'];
   want_meg = True
   want_eeg = False
   want_stim = False
   picks = fiff.pick_types(raw['info'], want_meg, want_eeg, want_stim,
                                include, raw['info']['bads'])

data, times, channel_names = mne.read_epochs(raw, events, event_id,
                                                    tmin, tmax, picks=picks)

# for epoch in data:
    