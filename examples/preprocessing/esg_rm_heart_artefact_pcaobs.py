"""
.. _ex-pcaobs:

==============================================================================================
Principal Component Analysis - Optimal Basis Sets (PCA-OBS) for removal of cardiac artefact
==============================================================================================

This script shows an example of how to use an adaptation of PCA-OBS
:footcite:`NiazyEtAl2005`. PCA-OBS was originally designed to remove
the ballistocardiographic artefact in simultaneous EEG-fMRI. Here, it
has been adapted to remove the delay between the detected R-peak and the
ballistocardiographic artefact such that the algorithm can be applied to
remove the cardiac artefact in EEG (electroencephalogrpahy) and ESG
(electrospinography) data. We will illustrate how it works by applying the
algorithm to ESG data, where the effect of removal is most pronounced.

See: https://www.biorxiv.org/content/10.1101/2024.09.05.611423v1
for more details on the dataset and application for ESG data.

"""

# Authors: Emma Bailey <bailey@cbs.mpg.de>, Steinn Hauser Magnusson <hausersteinn@gmail.com>
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from matplotlib import pyplot as plt
from mne.preprocessing.pca_obs import pca_obs
from mne.preprocessing import find_ecg_events, fix_stim_artifact
from mne.io import read_raw_eeglab
from scipy.signal import firls
import numpy as np
from mne import Epochs, events_from_annotations, concatenate_raws

###############################################################################
# Download sample subject data from OpenNeuro if you haven't already
# This will download simultaneous EEG and ESG data from a single participant after
# median nerve stimulation of the left wrist
# Set the target directory to your desired location
import openneuro as on
import glob
target_dir = '/data/pt_02569/test_data'
file_list = glob.glob(target_dir + '/sub-001/eeg/*median*.set')
if file_list:
    print('Data is already downloaded')
else:
    on.download(dataset='ds004388', target_dir=target_dir, include='sub-001/*median*_eeg*')

###############################################################################
# Define the esg channels (arranged in two patches over the neck and lower back)
# Also include the ECG channel for artefact correction
esg_chans = ["S35", "S24", "S36", "Iz", "S17", "S15", "S32", "S22", "S19", "S26", "S28",
             "S9", "S13", "S11", "S7", "SC1", "S4", "S18", "S8", "S31", "SC6", "S12",
             "S16", "S5", "S30", "S20", "S34", "S21", "S25", "L1", "S29", "S14", "S33",
             "S3", "L4", "S6", "S23", 'ECG']

# Sampling rate
fs = 1000

# Interpolation window for ESG data to remove stimulation artefact
tstart_esg = -0.007
tmax_esg = 0.007

# Define timing of heartbeat epochs
iv_baseline = [-400 / 1000, -300 / 1000]
iv_epoch = [-400 / 1000, 600 / 1000]

###############################################################################
# Read in each of the four blocks and concatenate the raw structures after performing
# some minimal preprocessing including removing the stimulation artefact, downsampling
# and filtering
block_files = glob.glob(target_dir + '/sub-001/eeg/*median*.set')
block_files = sorted(block_files)

for count, block_file in enumerate(block_files):
    raw = read_raw_eeglab(block_file, eog=(), preload=True, uint16_codec=None, verbose=None)

    # Isolate the ESG channels only
    raw.pick(esg_chans)

    # Find trigger timings to remove the stimulation artefact
    events, event_dict = events_from_annotations(raw)
    trigger_name = 'Median - Stimulation'

    fix_stim_artifact(raw, events=events, event_id=event_dict[trigger_name], tmin=tstart_esg, tmax=tmax_esg, mode='linear',
                      stim_channel=None)

    # Downsample the data
    raw.resample(fs)

    # Append blocks of the same condition
    if count == 0:
        raw_concat = raw
    else:
        concatenate_raws([raw_concat, raw])

###############################################################################
# Find ECG events and add to the raw structure as event annotations
ecg_events, ch_ecg, average_pulse = find_ecg_events(raw_concat, ch_name="ECG")
ecg_event_samples = np.asarray([[ecg_event[0] for ecg_event in ecg_events]])  # Samples only

qrs_event_time = [x / fs for x in ecg_event_samples.reshape(-1)]  # Divide by sampling rate to make times
duration = np.repeat(0.0, len(ecg_event_samples))
description = ['qrs'] * len(ecg_event_samples)

raw_concat.annotations.append(qrs_event_time, duration, description, ch_names=[esg_chans]*len(qrs_event_time))

###############################################################################
# Create filter coefficients
a = [0, 0, 1, 1]
f = [0, 0.4 / (fs / 2), 0.9 / (fs / 2), 1]  # 0.9 Hz highpass filter
ord = round(3 * fs / 0.5)
fwts = firls(ord + 1, f, a)

###############################################################################
# Create evoked response about the detected R-peaks before cardiac artefact correction
# Apply PCA-OBS to remove the cardiac artefact
# Create evoked response about the detected R-peaks after cardiac artefact correction
events, event_ids = events_from_annotations(raw_concat)
event_id_dict = {key: value for key, value in event_ids.items() if key == "qrs"}
epochs = Epochs(
    raw_concat,
    events,
    event_id=event_id_dict,
    tmin=iv_epoch[0],
    tmax=iv_epoch[1],
    baseline=tuple(iv_baseline),
)
evoked_before = epochs.average()

# Apply function - modifies the data in place
raw_concat.apply_function(
    pca_obs,
    picks=esg_chans,
    n_jobs=len(esg_chans),
    # args sent to PCA_OBS
    qrs=ecg_event_samples,
    filter_coords=fwts,
)

epochs = Epochs(
    raw_concat,
    events,
    event_id=event_id_dict,
    tmin=iv_epoch[0],
    tmax=iv_epoch[1],
    baseline=tuple(iv_baseline),
)
evoked_after = epochs.average()

###############################################################################
# Comparison image
fig, axes = plt.subplots(1, 1)
axes.plot(evoked_before.times, evoked_before.get_data().T, color="black")
axes.set_ylim([-0.0005, 0.001])
axes.plot(evoked_after.times, evoked_after.get_data().T, color="green")
axes.set_title("Before (black) versus after (green)")
plt.tight_layout()
plt.show()

# %%
# References
# ----------
# .. footbibliography::
