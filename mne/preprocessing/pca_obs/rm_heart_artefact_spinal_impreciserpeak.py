# Calls PCA_OBS which in turn calls fit_ecgTemplate to remove the heart artefact via PCA_OBS (Principal Component
# Analysis, Optimal Basis Sets)

from matplotlib import pyplot as plt
from mne.preprocessing.pca_obs.PCA_OBS import PCA_OBS
from mne.preprocessing.pca_obs import ESG_CHANS
from mne.preprocessing.pca_obs import (
    fs,
    iv_baseline,
    iv_epoch,
    raw,
)
from scipy.signal import firls
import numpy as np
from mne import Epochs, events_from_annotations
from mne.io import read_raw_fif
from mne.preprocessing import find_ecg_events

if __name__ == "__main__":

    # Find ECG events - no ECG channel in data, uses synthetic
    (
        ecg_events,
        ch_ecg,
        average_pulse,
    ) = find_ecg_events(raw, ch_name="ECG")
    # Extract just sample timings of ecg events
    ecg_event_samples = np.asarray([[ecg_event[0] for ecg_event in ecg_events]])

    # Create filter coefficients
    a = [0, 0, 1, 1]
    f = [0, 0.4 / (fs / 2), 0.9 / (fs / 2), 1]  # 0.9 Hz highpass filter
    # f = [0 0.4 / (fs / 2) 0.5 / (fs / 2) 1] # 0.5 Hz highpass filter
    ord = round(3 * fs / 0.5)
    fwts = firls(ord + 1, f, a)

    # run PCA_OBS
    events, event_ids = events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == "qrs"}
    epochs = Epochs(
        raw,
        events,
        event_id=event_id_dict,
        tmin=iv_epoch[0],
        tmax=iv_epoch[1],
        baseline=tuple(iv_baseline),
    )
    evoked_before = epochs.average()

    # Apply function should modifies the data in raw in place
    raw.apply_function(
        PCA_OBS, 
        picks=ESG_CHANS, 
        n_jobs=len(ESG_CHANS), 
        # args sent to PCA_OBS
        qrs=ecg_event_samples, 
        filter_coords=fwts, 
    )
    epochs = Epochs(
        raw,
        events,
        event_id=event_id_dict,
        tmin=iv_epoch[0],
        tmax=iv_epoch[1],
        baseline=tuple(iv_baseline),
    )
    evoked_after = epochs.average()

    # Comparison image
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(evoked_before.times, evoked_before.get_data().T)
    axes[0].set_ylim([-0.0005, 0.001])
    axes[0].set_title("Before PCA-OBS")
    axes[1].plot(evoked_after.times, evoked_after.get_data().T)
    axes[1].set_ylim([-0.0005, 0.001])
    axes[1].set_title("After PCA-OBS")
    plt.tight_layout()

    # Comparison image
    fig, axes = plt.subplots(1, 1)
    axes.plot(evoked_before.times, evoked_before.get_data().T, color="black")
    axes.set_ylim([-0.0005, 0.001])
    axes.plot(evoked_after.times, evoked_after.get_data().T, color="green")
    axes.set_title("Before (black) versus after (green)")
    plt.tight_layout()
    plt.show()
