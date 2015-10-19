"""
========================================================
Extract epochs, average and save evoked response to disk
========================================================

This script shows how to read the epochs from a raw file given
a list of events. The epochs are averaged to produce evoked
data and then saved to disk.

"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne import io
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5

# Select events to extract epochs from.
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2}

#   Setup for reading the raw data
raw = io.Raw(raw_fname)
events = mne.read_events(event_fname)

#   Plot raw data
fig = raw.plot(events=events, event_color={1: 'cyan', -1: 'lightgray'})

#   Set up pick list: EEG + STI 014 - bad channels (modify to your needs)
include = []  # or stim channels ['STI 014']
raw.info['bads'] += ['EEG 053']  # bads + 1 more

# pick EEG and MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=True,
                       include=include, exclude='bads')
# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(eeg=80e-6, eog=150e-6),
                    preload=True)

# Plot epochs.
epochs.plot(title='Auditory left/right')

# Look at channels that caused dropped events, showing that the subject's
# blinks were likely to blame for most epochs being dropped
epochs.drop_bad_epochs()
epochs.plot_drop_log(subject='sample')

# Average epochs and get evoked data corresponding to the left stimulation
evoked = epochs['Left'].average()

evoked.save('sample_audvis_eeg-ave.fif')  # save evoked data to disk

###############################################################################
# View evoked response

evoked.plot(gfp=True)

###############################################################################
# Save evoked responses for different conditions to disk

# average epochs and get Evoked datasets
evokeds = [epochs[cond].average() for cond in ['Left', 'Right']]

# save evoked data to disk
mne.write_evokeds('sample_auditory_and_visual_eeg-ave.fif', evokeds)
