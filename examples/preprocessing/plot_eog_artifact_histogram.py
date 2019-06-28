"""
========================
Show EOG artifact timing
========================

Compute the distribution of timing for EOG artifacts.

"""
# Authors: Eric Larson <larson.eric.d@gmail.com>
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
raw = io.read_raw_fif(raw_fname, preload=True)
events = mne.find_events(raw, 'STI 014')
eog_event_id = 512
eog_events = mne.preprocessing.find_eog_events(raw, eog_event_id)
raw.add_events(eog_events, 'STI 014')

# Read epochs
picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=True, eog=False)
tmin, tmax = -0.2, 0.5
event_ids = {'AudL': 1, 'AudR': 2, 'VisL': 3, 'VisR': 4}
epochs = mne.Epochs(raw, events, event_ids, tmin, tmax, picks=picks)

# Get the stim channel data
pick_ch = mne.pick_channels(epochs.ch_names, ['STI 014'])[0]
data = epochs.get_data()[:, pick_ch, :]
data = np.sum((data.astype(int) & eog_event_id) == eog_event_id, axis=0)

###############################################################################
# Plot EOG artifact distribution
fig, ax = plt.subplots()
ax.stem(1e3 * epochs.times, data, use_line_collection=True)
ax.set(xlabel='Times (ms)',
       ylabel='Blink counts (from %s trials)' % len(epochs))
fig.tight_layout()
