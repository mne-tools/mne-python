"""
Example on sleep data
=====================

This tutorial explains how to load and read polysomnography recordings and
perform sleep stage classification on Sleep Physionet dataset [1]_, [2]_.

We perform the following steps:
  1. we load 2 PSG recordings from 2 different subjects.
  2. we read annotations and EEG time series from the recordings
  3. we extract some features from the EEG data: relative power
    in specific frequency bands: [0.5 - 4.5Hz], ...
  4. we train a RandomForestClassifier on features and annotations from one
    recording
  5. we predict the sleep stages on the second recording.

References
----------
.. [1] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of
       a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity
       of the EEG. IEEE-BME 47(9):1185-1194 (2000).
.. [2] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh,
       Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000)
       PhysioBank, PhysioToolkit, and PhysioNet: Components of a New
       Research Resource for Complex Physiologic Signals.
       Circulation 101(23):e215-e220

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
from sklearn.metrics import accuracy_score

##############################################################################
# Load 2 PSG recordings from 2 different subjects

psg_train, hyp_train = fetch_data(subjects=[0])[0]
psg_test, hyp_test = fetch_data(subjects=[1])[0]


##############################################################################
# read the PSG data and annotations

mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

raw_train = mne.io.read_raw_edf(psg_train, stim_channel=False)
annot_train = mne.read_annotations(hyp_train)

raw_train.set_annotations(annot_train)
raw_train.set_channel_types(mapping)

raw_test = mne.io.read_raw_edf(psg_test, stim_channel=False)
annot_test = mne.read_annotations(hyp_test)

raw_test.set_annotations(annot_test)
raw_test.set_channel_types(mapping)

# plot some data
raw_train.plot(duration=60)


##############################################################################
# Extract 30s events from annotations

annotation_desc_2_event_id = {'Sleep stage W': 1,
                              'Sleep stage 1': 2,
                              'Sleep stage 2': 3,
                              'Sleep stage 3': 4,
                              'Sleep stage 4': 4,
                              'Sleep stage R': 5}

events_train, event_id_train = mne.events_from_annotations(
    raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)

events_test, event_id_test = mne.events_from_annotations(
    raw_test, event_id=annotation_desc_2_event_id, chunk_duration=30.)

del event_id_train['Sleep stage 4']  # remove duplicated event_id
del event_id_test['Sleep stage 4']

# plot events
mne.viz.plot_events(
    events_train, event_id=event_id_train, sfreq=raw_train.info['sfreq'])

##############################################################################
# Epoching

tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included
epochs_train = mne.Epochs(
    raw_train, events_train, event_id_train,
    tmin=0., tmax=tmax, baseline=None)

tmax = 30. - 1. / raw_test.info['sfreq']  # tmax in included
epochs_test = mne.Epochs(
    raw_test, events_test, event_id_test,
    tmin=0., tmax=tmax, baseline=None)

print(epochs_train)
print(epochs_test)

##############################################################################
# Extract features from EEG: relative power in specific frequency bands

eeg_channels = ["EEG Fpz-Cz", "EEG Pz-Oz"]

# specific frequency bands
freq_bands = {"delta": [0.5, 4.5],
              "theta": [4.5, 8.5],
              "alpha": [8.5, 11.5],
              "sigma": [11.5, 15.5],
              "beta": [15.5, 30]}

# extract features from training data
epochs_train.load_data().pick_channels(eeg_channels)
z_norm = np.linalg.norm(epochs_train.get_data(), axis=-1)

X_train = np.zeros(
    (epochs_train.events.shape[0], len(eeg_channels) * len(freq_bands)))
for idx_band, (band, lims) in enumerate(freq_bands.items()):
    print(idx_band, band, lims)

    z = np.linalg.norm(
        epochs_train.copy().filter(lims[0], lims[1]).get_data(),
        axis=-1) / z_norm

    X_train[
        :, len(eeg_channels) * idx_band:len(eeg_channels) * (idx_band + 1)] = z

# extract features from testing data
epochs_test.load_data().pick_channels(eeg_channels)
z_norm = np.linalg.norm(epochs_test.get_data(), axis=-1)

X_test = np.zeros(
    (epochs_test.events.shape[0], len(eeg_channels) * len(freq_bands)))
for idx_band, (band, lims) in enumerate(freq_bands.items()):
    print(idx_band, band, lims)

    z = np.linalg.norm(
        epochs_test.copy().filter(lims[0], lims[1]).get_data(),
        axis=-1) / z_norm

    X_test[
        :, len(eeg_channels) * idx_band:len(eeg_channels) * (idx_band + 1)] = z

# format annotations
y_train = events_train[:, 2]
y_test = events_test[:, 2]

##############################################################################
# Train a classifier and predict on test recording

clf = RandomForestClassifier().fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test))
print("Accuracy score: {}".format(acc))


##############################################################################
# Plot the power spectrum density (PSD) in each stage

_, ax = plt.subplots()

for stage in zip(epochs_train.event_id):
    epochs_train[stage].plot_psd(area_mode=None, ax=ax, fmin=0.1, fmax=20.)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
[line.set_color(color) for line, color in zip(ax.get_lines(), colors)]
plt.legend(list(epochs_train.event_id.keys()))
