"""
=============================
Re-referencing the EEG signal
=============================

Load raw data and apply some EEG referencing schemes.
"""
# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample
from matplotlib import pyplot as plt

print(__doc__)

# Setup for reading the raw data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_id, tmin, tmax = 1, -0.2, 0.5

# Read the raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
events = mne.read_events(event_fname)

# The EEG channels will be plotted to visualize the difference in referencing
# schemes.
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True, exclude='bads')

###############################################################################
# Apply different EEG referencing schemes and plot the resulting evokeds

reject = dict(eeg=180e-6, eog=150e-6)
epochs_params = dict(events=events, event_id=event_id, tmin=tmin, tmax=tmax,
                     picks=picks, reject=reject)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)

# No reference. This assumes that the EEG has already been referenced properly.
# This explicitly prevents MNE from adding a default EEG reference.
raw_no_ref, _ = mne.io.set_eeg_reference(raw, [])
evoked_no_ref = mne.Epochs(raw_no_ref, **epochs_params).average()
del raw_no_ref  # Free memory

evoked_no_ref.plot(axes=ax1, titles=dict(eeg='EEG Original reference'))

# Average reference. This is normally added by default, but can also be added
# explicitly.
raw_car, _ = mne.io.set_eeg_reference(raw)
evoked_car = mne.Epochs(raw_car, **epochs_params).average()
del raw_car

evoked_car.plot(axes=ax2, titles=dict(eeg='EEG Average reference'))

# Use the mean of channels EEG 001 and EEG 002 as a reference
raw_custom, _ = mne.io.set_eeg_reference(raw, ['EEG 001', 'EEG 002'])
evoked_custom = mne.Epochs(raw_custom, **epochs_params).average()
del raw_custom

evoked_custom.plot(axes=ax3, titles=dict(eeg='EEG Custom reference'))

mne.viz.tight_layout()
