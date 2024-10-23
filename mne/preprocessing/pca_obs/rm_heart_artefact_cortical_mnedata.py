# Checking algorithm implementation with EEG data form MNE sample datasets
# Following this tutorial data: https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html#what-is-ssp

import os

import matplotlib.pyplot as plt
import numpy as np
from PCA_OBS import *
from scipy.signal import firls

from mne import Epochs
from mne.datasets.sample import data_path
from mne.io import read_raw_fif
from mne.preprocessing import (
    find_ecg_events,
)

if __name__ == "__main__":
    sample_data_folder = data_path(path="/data/pt_02569/mne_test_data/")
    sample_data_raw_file = os.path.join(
        sample_data_folder, "MEG", "sample", "sample_audvis_raw.fif"
    )
    # here we crop and resample just for speed
    raw = read_raw_fif(sample_data_raw_file, preload=True)

    # Find ECG events - no ECG channel in data, uses synthetic
    (
        ecg_events,
        ch_ecg,
        average_pulse,
    ) = find_ecg_events(raw)
    # Extract just sample timings of ecg events
    ecg_event_samples = np.asarray([[ecg_event[0] for ecg_event in ecg_events]])
    # print(ecg_events)

    # Create filter coefficients
    fs = raw.info["sfreq"]
    a = [0, 0, 1, 1]
    f = [0, 0.4 / (fs / 2), 0.9 / (fs / 2), 1]  # 0.9 Hz highpass filter
    # f = [0 0.4 / (fs / 2) 0.5 / (fs / 2) 1] # 0.5 Hz highpass filter
    ord = round(3 * fs / 0.5)
    fwts = firls(ord + 1, f, a)

    # For heartbeat epochs
    iv_baseline = [-300 / 1000, -200 / 1000]
    iv_epoch = [-400 / 1000, 600 / 1000]

    # run PCA_OBS
    # Algorithm is extremely sensitive to accurate R-peak timings, won't work as well with the artificial ECG
    # channel estimation as we have here
    PCA_OBS_kwargs = dict(qrs=ecg_event_samples, filter_coords=fwts, sr=fs)

    epochs = Epochs(
        raw, ecg_events, tmin=iv_epoch[0], tmax=iv_epoch[1], baseline=tuple(iv_baseline)
    )
    evoked_before = epochs.average()

    # Apply function should modifies the data in raw in place
    raw.apply_function(PCA_OBS, picks="eeg", **PCA_OBS_kwargs, n_jobs=10)
    epochs = Epochs(
        raw, ecg_events, tmin=iv_epoch[0], tmax=iv_epoch[1], baseline=tuple(iv_baseline)
    )
    evoked_after = epochs.average()

    # Comparison image
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(evoked_before.times, evoked_before.get_data().T)
    axes[0].set_ylim([-1e-5, 3e-5])
    axes[0].set_title("Before PCA-OBS")
    axes[1].plot(evoked_after.times, evoked_after.get_data().T)
    axes[1].set_ylim([-1e-5, 3e-5])
    axes[1].set_title("After PCA-OBS")
    plt.tight_layout()

    # Comparison image
    fig, axes = plt.subplots(1, 1)
    axes.plot(evoked_before.times, evoked_before.get_data().T, color="black")
    axes.set_ylim([-1e-5, 3e-5])
    axes.plot(evoked_after.times, evoked_after.get_data().T, color="green")
    axes.set_title("Before (black) versus after (green)")
    plt.tight_layout()
    plt.show()
