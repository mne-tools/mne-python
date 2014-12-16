"""
=============================
Re-referencing the EEG signal
=============================

Load raw data and apply some EEG referencing schemes
"""
# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample
from matplotlib import pyplot as plt

# Setup for reading the raw data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_id, tmin, tmax = 1, -0.2, 0.5

# Read the raw data
raw = mne.io.Raw(raw_fname, preload=True)
events = mne.read_events(event_fname)

# Apply different EEG referencing schemes
#########################################

# No reference. This assumes that the EEG has already been referenced properly.
# This explicitly prevents MNE from adding a default EEG reference.
raw_none = mne.io.set_eeg_reference(raw, [])[0]

# Average reference. This is normally added by default, but can also be added
# explicitly.
raw_car = mne.io.set_eeg_reference(raw)[0]

# Use the mean of channels EEG 001 and EEG 002 as a reference
raw_custom = mne.io.set_eeg_reference(raw, ['EEG 001', 'EEG 002'])[0]

# Plot the evokeds to see the effect of the EEG reference
#########################################################
picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')
ev_none = mne.Epochs(raw_none, events, event_id, tmin, tmax).average(picks)
ev_car = mne.Epochs(raw_car, events, event_id, tmin, tmax).average(picks)
ev_custom = mne.Epochs(raw_custom, events, event_id, tmin, tmax).average(picks)

plt.figure(figsize=(6, 6))

ax = plt.subplot(3, 1, 1)
ev_none.plot(axes=ax)
plt.title('Original reference')

ax = plt.subplot(3, 1, 2)
ev_car.plot(axes=ax)
plt.title('Average reference')

ax = plt.subplot(3, 1, 3)
ev_custom.plot(axes=ax)
plt.title('Custom reference')

plt.tight_layout()
plt.show()
