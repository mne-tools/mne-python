"""
Dipole fitting with epochs that contain NaNs.

Compare dipole fitting error using:
1. New: Pairwise covariance (allows for NaNs in epochs)
2. Standard: Sample covariance (epochs with NaNs dropped)

Phantom dipole simulation using channel-specific epoch rejection.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mne
from mne import find_events, fit_dipole
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif

# Load phantom data
# -----------------

data_path = bst_phantom_elekta.data_path(verbose=True)
raw_fname = data_path / "kojak_all_200nAm_pp_no_chpi_no_ms_raw.fif"
raw = read_raw_fif(raw_fname, preload=True)
raw.info["bads"] = ["MEG1933", "MEG2421"]
events = find_events(raw, "STI201")
tmin, tmax = -0.1, 0.1
bmax = -0.05
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.08)
actual_pos, actual_ori = mne.dipole.get_phantom_dipoles()
true_amp = 200.0  # nAm

rng = np.random.RandomState(42)

# Store for results
results = {"pairwise": [], "sample": []}

# Loop over dipoles
# -----------------
for dipole_idx in range(1, 5):
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

    # drop first and last epoch
    epochs = epochs[1:-1]

    n_epochs, n_channels = len(epochs), len(epochs.ch_names)

    # channel-specific epoch rejection
    # ---------------------------------------------------
    mask = np.zeros((n_epochs, n_channels), dtype=bool)
    mask[: n_epochs // 3, : n_channels // 3] = True
    epochs.drop_bad_epochs_by_channel(mask)

    # cov = mne.compute_covariance(epochs)
    # array must not contain infs or NaNs

    # Condition 1: Pairwise covariance (NaN-aware)
    epochs_data = epochs.copy().crop(None, bmax).get_data()
    n_epochs, n_channels, n_times = epochs_data.shape

    X = epochs_data.reshape(n_epochs * n_times, n_channels)

    # pairwise covariance is implemented in pandas
    df = pd.DataFrame(X)
    cov_pairwise = df.cov(min_periods=2)  # how many valid samples

    # cov_pairwise_mne = mne.Covariance(
    #   cov_pairwise.values,
    #   names=epochs.ch_names,
    #   nfree=X.shape[0] - 1,
    #   projs=[],
    #   bads=[]
    # )

    cov_pairwise_mne = mne.Covariance(
        cov_pairwise.values,
        names=epochs.ch_names,
        nfree=np.sum(~np.isnan(X[:, 0])) - 1,
        projs=[],
        bads=[],
    )
    plt.imshow(cov_pairwise, aspect="auto")
    plt.colorbar()
    plt.title("Pairwise covariance")
    plt.xlabel("Channels")
    plt.ylabel("Channels")
    plt.show()

    # drop the 10 bad epochs that contain NaNs
    bad_epochs = np.any(np.isnan(epochs_data), axis=(1, 2))

    epochs.drop(bad_epochs)

    evoked_clean = epochs.copy().crop(0.035, 0.035).average()

    dip_pairwise, _ = fit_dipole(evoked_clean, cov_pairwise_mne, sphere)

    # Condition 2: Sample covariance (dropped bad epochs)
    cov_sample = mne.compute_covariance(epochs, tmax=bmax, method="empirical")

    dip_sample, _ = fit_dipole(evoked_clean, cov_sample, sphere)

    # True dipole parameters
    true_pos = actual_pos[dipole_idx]
    true_ori = actual_ori[dipole_idx]

    # ---------------------------------------------------
    # Error computation helper
    # ---------------------------------------------------
    def compute_errors(dip):
        """Dipole fitting errors."""
        pos_err = 1000 * np.sqrt(np.sum((dip.pos - true_pos) ** 2, axis=-1))
        ori_err = np.rad2deg(np.arccos(np.abs(np.sum(dip.ori * true_ori, axis=1))))
        amp_err = true_amp - np.array(dip.amplitude) / 1e-9
        return pos_err, ori_err, amp_err

    results["pairwise"].append(compute_errors(dip_pairwise))
    results["sample"].append(compute_errors(dip_sample))


# -----------------------------
# Convert results to arrays
# -----------------------------
pairwise = np.array(results["pairwise"])
sample = np.array(results["sample"])


# -----------------------------
# Plot comparison
# -----------------------------
metrics = ["Position error (mm)", "Orientation error (deg)", "Amplitude error (nAm)"]

fig, axes = plt.subplots(3, 1, figsize=(8, 10), layout="constrained")

for i, ax in enumerate(axes):
    ax.plot(pairwise[:, i], label="Pairwise covariance", marker="o")
    ax.plot(sample[:, i], label="Sample covariance", marker="o")
    ax.set_title(metrics[i])
    ax.set_xlabel("Dipole index")
    ax.legend()

plt.show()
