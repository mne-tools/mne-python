"""
.. _tut_erp:

Event Related Potentials (ERPs) Measures
==================================================

.. contents:: Here we cover the specifics of extracting data from ERPs, namely:
   :local:
   :depth: 1

"""

import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample

###############################################################################
# Create evoked from raw data

# Read in raw data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# Select only the the EEG and EOG channels
raw.pick_types(meg=False, eeg=True, eog=True).load_data()

# Create epochs from events, and reject epochs
events = mne.read_events(event_fname)
event_id = {'auditory/left': 1,
            'auditory/right': 2,
            'visual/left': 3,
            'visual/right': 4}
tmin, tmax = -0.2, 0.5
reject = dict(eeg=180e-6, eog=150e-6)
epochs_params = dict(events=events, event_id=event_id, tmin=tmin, tmax=tmax,
                     reject=reject)
epochs = mne.Epochs(raw, **epochs_params)

# Plot the sensors to show locations
epochs.plot_sensors(show_names=True)

# Make all evokeds
evokeds = dict((cond, epochs[cond].average()) for cond in event_id)
print(evokeds['visual/left'])

# Plot the ERPs for visual stimulation
evokeds['visual/left'].plot_joint(title='Visual Left (EEG)',
                                  times=[.09, .18, .27])
evokeds['visual/right'].plot_joint(title='Visual Right (EEG)',
                                   times=[.09, .18, .27])

###############################################################################
# Obtaining Peak Latency and Amplitude
# ---------------------
#
# One of the most common ERP measures is optaining the peak amplitude and time.
# This usually is done by searching in a user-specifed time frame. We
# will obtain the peak latency and amplitude for the first positive peak for
# the visual trials in each hemisphere. The :func:`mne.io.Evoked.get_peak`
# method to achieve this. Here, we will focus on channels 'EEG 057' and
# 'EEG 059' (left and right hemispheres, respectively).
#
# To mitigate high-frequency noise influencing peak measures, we can low-pass
# filter the data with a cutoff of 20Hz or 30Hz. Here, we will do 20Hz. Note
# that the Raw object of this sample dataset is lowpass filtered at 40Hz.

# First we will extract the data from right trials in contralateral hemisphere

# Define channel to use
chan = 'EEG 054'

# Copy and filter evoked
right_vis_evoked = evokeds['visual/right'].copy()
right_vis_evoked.filter(None, 20)
right_vis_evoked.pick(chan)

# Get peak latency and amplitude at EEG 054
_, lh_lat, lh_amp = right_vis_evoked.get_peak(tmin=.07, tmax=.13,
                                              mode='pos',
                                              return_amplitude=True)

# Convert amplitude from volts to microvolts
lh_microvolts = lh_amp * 1e6

# Plot the evoked trace with a point for the peak
fig, ax = plt.subplots()
fig = right_vis_evoked.plot(picks=chan, axes=ax,
                            titles=f'Visual/Right Peak at {chan}',
                            proj=True)
ax.plot(lh_lat, lh_microvolts, 'ro')
fig

# Next, we will do the same for the left visual trials in
# the right hemisphere
chan = 'EEG 059'
left_vis_evoked = evokeds['visual/right'].copy()
left_vis_evoked.filter(None, 20)
left_vis_evoked.pick(chan)
_, rh_latency, rh_amplitude = left_vis_evoked.get_peak(tmin=.07, tmax=.13,
                                                       mode='pos',
                                                       return_amplitude=True)

# Convert amplitude from volts to microvolts
rh_microvolts = rh_amplitude * 1e6

# Plot the evoked trace with a point for the peak
fig, ax = plt.subplots(nrows=1, ncols=1)
fig = right_vis_evoked.plot(picks=chan, axes=ax,
                            titles=f'Visual/Left Peak at {chan}',
                            proj=True)
ax.plot(rh_latency, rh_microvolts, 'ro')
fig


###############################################################################
# Obtaining Mean Amplitude in a Specified Time Window
# ---------------------------------------------------------------
#
# Another common practice in ERP studies is to define a component (or effect)
# as the mean amplitude within a specified time window. Sometimes, this also
# involves extracting the mean over both time and space (sensors). The
# following demonstrates how to do both. We will focus again on visual trials
# in posterior electrodes.
#
# Below, we extract the mean amplitude from .08s to .12s for visual trials.
# Given how the average amplitude over a time window is extracted, there is no
# need to apply any additional lowpass filtering to the evoked data.

# Define electrode clusters
lh_cluster = [f'EEG {x:03}' for x in [44, 45, 54, 57]]
rh_cluster = [f'EEG {x:03}' for x in [52, 55, 56, 59]]

# Extract mean amplitude and store in a dictionary
tmin, tmax = .08, .12

# Create dictionary with amplitudes converted to microvolts
mean_amplitude = {}
for cond in ['visual/left', 'visual/right']:
    lh_evoked = evokeds[cond].copy().pick(lh_cluster)
    rh_evoked = evokeds[cond].copy().pick(rh_cluster)
    mean_amplitude[cond] = {
        'L-Hemi': lh_evoked.crop(tmin=tmin, tmax=tmax).data.mean() * 1e6,
        'R-Hemi': rh_evoked.crop(tmin=tmin, tmax=tmax).data.mean() * 1e6
    }

# Convert to dataframe and make a bar plot
# Make data frame. Transposed to make hemisphere as columns
amp_df = pd.DataFrame(mean_amplitude).T
amp_df.plot(kind='bar')
