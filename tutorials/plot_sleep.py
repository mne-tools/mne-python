"""
Example on sleep data
=====================

This tutorial explains how to load and read polysomnography recordings and
extract some basic features (relative power in frequency bands) in order to
train a classifier

"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Stanislas Chambon <stan.chambon@gmail.com>
#
# License: BSD Style.

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets.sleep_physionet import fetch_data
from sklearn.ensemble import RandomForestClassifier

psg_fname, hyp_fname = fetch_data(subjects=[0])[0]

raw = mne.io.read_raw_edf(psg_fname, stim_channel=False)
annotations = mne.read_annotations(hyp_fname)

# ##############################################################################

raw.set_annotations(annotations)
raw.plot(duration=60)

mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}
raw.set_channel_types(mapping)

##############################################################################
# Extract 30s events from annotations

annotation_desc_2_event_id = {'Sleep stage W': 1,
                              'Sleep stage 1': 2,
                              'Sleep stage 2': 3,
                              'Sleep stage 3': 4,
                              'Sleep stage 4': 4,
                              'Sleep stage R': 5}

events, event_id = mne.events_from_annotations(
    raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)

del event_id['Sleep stage 4']  # remove duplicated event_id

mne.viz.plot_events(events, event_id=event_id, sfreq=raw.info['sfreq'])

##############################################################################
# Epoching

tmax = 30. - 1. / raw.info['sfreq']  # tmax in included
epochs = mne.Epochs(raw, events, event_id, tmin=0., tmax=tmax, baseline=None)
print(epochs)

##############################################################################
# Extract features from EEG: relative power in specific frequency bands

eeg_channels = ["EEG Fpz-Cz", "EEG Pz-Oz"]
epochs.load_data().pick_channels(eeg_channels)
z_norm = np.linalg.norm(epochs.get_data(), axis=-1)

# specific frequency bands
freq_bands = {"delta": [0.5, 4.5],
              "theta": [4.5, 8.5],
              "alpha": [8.5, 11.5],
              "sigma": [11.5, 15.5],
              "beta": [15.5, 30]}

X = np.zeros((epochs.events.shape[0], len(eeg_channels) * len(freq_bands)))
for idx_band, (band, lims) in enumerate(freq_bands.items()):
    print(idx_band, band, lims)

    z = np.linalg.norm(
        epochs.copy().filter(lims[0], lims[1]).get_data(), axis=-1) / z_norm

    X[:, len(eeg_channels) * idx_band:len(eeg_channels) * (idx_band + 1)] = z

##############################################################################
# Train a classifier on this record
# The classifier can then be used to predict on another record

y = events[:, 2]
clf = RandomForestClassifier().fit(X, y)

##############################################################################
# Plot the power spectrum density (PSD) in each stage

_, ax = plt.subplots()

for stage in zip(epochs.event_id):
    epochs[stage].plot_psd(area_mode=None, ax=ax, fmin=0.1, fmax=20.)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
[line.set_color(color) for line, color in zip(ax.get_lines(), colors)]
plt.legend(list(epochs.event_id.keys()))
