"""
==================
Find EOG artifacts
==================

Locate peaks of EOG to spot blinks and general EOG artifacts.

"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np
import pylab as pl
import mne
from mne import fiff
from mne.datasets import sample
data_path = sample.data_path('..')

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)

event_id = 998
eog_events = mne.artifacts.find_eog_events(raw, event_id)

# Read epochs
picks = fiff.pick_types(raw.info, meg=False, eeg=False, stim=False, eog=True)
tmin, tmax = -0.2, 0.2
epochs = mne.Epochs(raw, eog_events, event_id, tmin, tmax, picks=picks)
data = epochs.get_data()

print "Number of detected EOG artifacts : %d" % len(data)

###############################################################################
# Plot EOG artifacts
pl.plot(1e3 * epochs.times, np.squeeze(data).T)
pl.xlabel('Times (ms)')
pl.ylabel('EOG (muV)')
pl.show()
