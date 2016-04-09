"""
==================
Find EOG artifacts
==================

Locate peaks of EOG to spot blinks and general EOG artifacts.

"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)


import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname)

event_id = 998
eog_events = mne.preprocessing.find_eog_events(raw, event_id)

# Read epochs
picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=False, eog=True,
                       exclude='bads')
tmin, tmax = -0.2, 0.2
epochs = mne.Epochs(raw, eog_events, event_id, tmin, tmax, picks=picks)
data = epochs.get_data()

print("Number of detected EOG artifacts : %d" % len(data))

###############################################################################
# Plot EOG artifacts
plt.plot(1e3 * epochs.times, np.squeeze(data).T)
plt.xlabel('Times (ms)')
plt.ylabel('EOG (muV)')
plt.show()
