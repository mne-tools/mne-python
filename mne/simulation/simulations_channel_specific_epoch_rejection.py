"""Dipole fitting with epochs that contain NaNs.

Compare dipole fitting error using:

1. Standard: Sample covariance (epochs with NaNs dropped)
    "Only use data where all channels are good"
    => reject bad segments across all channels

2. New: Pairwise covariance (allows for NaNs in epochs)
    "For ech pair of channels, use all time points where both are valid"
    => Channel FP1 is bad for some time - exclude for pairs involving FP1
    => other channel pairs still use those time points

Phantom dipole simulation using channel-specific epoch rejection.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mne
from mne import find_events, fit_dipole
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif

# Load phantom data
data_path = bst_phantom_elekta.data_path(verbose=True)
raw_fname = data_path / "kojak_all_200nAm_pp_no_chpi_no_ms_raw.fif"
raw = read_raw_fif(raw_fname, preload=True)
raw.info["bads"] = ["MEG1933", "MEG2421"]
events = find_events(raw, "STI201")
tmin, tmax = -0.1, 0.1
bmax = -0.05
peak_t = 0.036
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.08)
actual_pos, actual_ori = mne.dipole.get_phantom_dipoles()
true_amp = 100.0  # nAm


def compute_errors(dip):
    """Dipole fitting errors based on Elekta tutorial."""
    pos_err = 1000 * np.sqrt(np.sum((dip.pos - true_pos) ** 2, axis=-1))
    ori_err = np.rad2deg(np.arccos(np.abs(np.sum(dip.ori * true_ori, axis=1))))
    amp_err = true_amp - np.array(dip.amplitude) / 1e-9
    return pos_err, ori_err, amp_err


results = {"pairwise": [], "sample": []}

for dipole_idx in range(1, 32):
    epochs = mne.Epochs(
        raw,
        events,
        event_id=dipole_idx,
        tmin=tmin,
        tmax=tmax,
        baseline=(None, bmax),
        preload=True,
        reject_by_annotation=True,
    )

    # drop contaminated edges
    epochs = epochs[1:-1]

    ch_names = np.array(epochs.ch_names)

    # --- MEG channels only ---
    meg_picks = mne.pick_types(epochs.info, meg=True)
    meg_chs = ch_names[meg_picks]

    loc_ids = np.array([ch[:6] for ch in meg_chs])
    unique_locs = np.unique(loc_ids)

    # take first half of sensor locations (triplets)
    bad_locs = unique_locs[: len(unique_locs) // 2]

    n_epochs = len(epochs)
    n_channels = len(ch_names)

    # Test with one third of epochs marked as bad
    bad_epochs = np.arange(n_epochs // 3)

    mask = np.zeros((n_epochs, n_channels), dtype=bool)

    # apply triplet-consistent masking (mag + grad channels)
    for i, ch in enumerate(ch_names):
        if ch in meg_chs:
            loc = ch[:6]
            if loc in bad_locs:
                mask[bad_epochs, i] = True

    # apply mask
    epochs.drop_bad_epochs_by_channel(mask)

    # detect channels with any NaNs
    nan_chs = mask.any(axis=0)

    # only plot sensor layout once
    if dipole_idx == 1:
        # make a copy of info
        info_copy = epochs.info.copy()

        # define bad channels only in the copy
        info_copy["bads"] = [
            ch for ch, is_nan in zip(epochs.ch_names, nan_chs) if is_nan
        ]

        # create a temporary Epochs object for plotting only
        epochs_plot = epochs.copy()
        epochs_plot.info = info_copy

        # plot
        epochs_plot.plot_sensors(kind="topomap", show_names=False)
        # mainly frontal channels, slightly more channels on the left

    # cov = mne.compute_covariance(epochs)
    # array must not contain infs or NaNs

    # We calculate covariance on baseline window.
    # Condition 1: Pairwise covariance (NaN-aware)
    epochs_data_baseline = epochs.copy().crop(None, bmax).get_data()
    n_epochs, n_channels, n_times = epochs_data_baseline.shape

    # swap channels and time and concatenate epochs
    X_t = epochs_data_baseline.transpose(0, 2, 1).reshape(
        n_epochs * n_times, n_channels
    )

    # pairwise covariance is implemented in pandas covariance
    df = pd.DataFrame(X_t)
    cov_pairwise = df.cov(min_periods=2)  # how many valid samples

    # degrees of freedom are important but not sure yet
    nfree = n_epochs * (n_times - 1)  # samples per channel - 1

    # setup covariance object for dipole
    cov_pairwise_mne = mne.Covariance(
        cov_pairwise.values, names=epochs.ch_names, nfree=nfree, projs=[], bads=[]
    )

    # create copy of epochs object
    epochs_nan = epochs.copy()

    # drop the epochs that contain NaNs
    bad_epochs = np.any(np.isnan(epochs.get_data()), axis=(1, 2))
    epochs.drop(bad_epochs)

    # create evoked from clean epochs
    evoked_clean = epochs.copy().crop(peak_t, peak_t).average()

    # evoked from epochs without dropping NaNs
    evoked_nan = epochs_nan.copy().crop(peak_t, peak_t).average()

    # now fit dipole with pairwise covariance and evoked with NaNs
    dip_pairwise, _ = fit_dipole(evoked_nan, cov_pairwise_mne, sphere)

    # Standard: Sample covariance (dropped bad epochs)
    cov_sample = mne.compute_covariance(epochs, tmax=bmax, method="empirical")

    # fit dipole on data with dropped bad epochs
    dip_sample, _ = fit_dipole(evoked_nan, cov_sample, sphere)

    # get true dipole parameters
    true_pos = actual_pos[dipole_idx]
    true_ori = actual_ori[dipole_idx]

    # compute and store error
    results["pairwise"].append(compute_errors(dip_pairwise))
    results["sample"].append(compute_errors(dip_sample))

# Convert results to arrays
pairwise = np.array(results["pairwise"])
sample = np.array(results["sample"])

# Plot
metrics = ["Position error (mm)", "Orientation error (deg)", "Amplitude error (nAm)"]

fig, axes = plt.subplots(3, 1, figsize=(8, 10), layout="constrained")

for i, ax in enumerate(axes):
    ax.plot(pairwise[:, i], label="Pairwise covariance", marker="o")
    ax.plot(sample[:, i], label="Sample covariance", marker="o")
    ax.set_title(metrics[i])
    ax.set_xlabel("Dipole index")
    ax.legend()

plt.show()
