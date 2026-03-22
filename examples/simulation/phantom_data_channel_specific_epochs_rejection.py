import matplotlib.pyplot as plt
import numpy as np

import mne
from mne import find_events, fit_dipole
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif

# true dipole positions and origins
true_pos, true_ori = mne.dipole.get_phantom_dipoles()

# Load Phantom MEG Data
data_path = bst_phantom_elekta.data_path(verbose=True)
raw_fname = data_path / "kojak_all_200nAm_pp_no_chpi_no_ms_raw.fif"
raw = read_raw_fif(raw_fname, preload=True)
raw.info["bads"] = ["MEG1933", "MEG2421"]  # known bad channels

# Find events for phantom stimulation
events = find_events(raw, "STI201")

# Epoching around stimulus
tmin, tmax = -0.1, 0.1
bmax = -0.05  # baseline

# Eventually we want to loop over all simulated dipole events
event_id = 1
picks = mne.pick_types(
    raw.info,
    meg=True,  # pick MEG channels
    eeg=False,
    stim=False,
    exclude="bads",  # drop bad channels immediately
)
epochs = mne.Epochs(
    raw, events, event_id=event_id, tmin=tmin, tmax=tmax, picks=picks, preload=True
)
# drop first and last epoch (might be corrupted)
epochs = epochs[1:-1]

n_epochs, n_channels, n_times = epochs.get_data().shape

print(f"Data shape: epochs={n_epochs}, channels={n_channels}, times={n_times}")

# Estimate dipole positions
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.08)
trans = mne.transforms.Transform("head", "mri", np.eye(4))

# Compute covariance for baseline
cov_full = mne.compute_covariance(epochs, tmax=bmax)

# fit dipole on all epochs available
t_peak = 0.036  # max global field power (see brainstorm_phantom_elekta)
evoked = epochs.average().crop(t_peak, t_peak)
dipole_allepochs, _ = fit_dipole(evoked, cov_full, sphere)
est_pos = dipole_allepochs.pos

# Simulate channel-specific epoch rejection
rng = np.random.RandomState(42)
# [0.1, 0.3, 0.5, 0.7]
drop_probs = [0.7]  # fraction of epochs dropped per channel

# store localisation error
errors_a, errors_b = [], []
imbalance_metrics = []

for p in drop_probs:
    # mask bad epochs per channel
    mask = rng.rand(n_epochs, n_channels) < p
    epochs_nan = epochs.copy().drop_bad_epochs_by_channel(mask)

    # Compute per-channel number of valid epochs
    per_channel_nave = (~np.isnan(epochs_nan.get_data()).any(axis=2)).sum(axis=0)

    # P-norm computation
    effective_nave = (np.mean(np.sqrt(per_channel_nave))) ** 2

    # Channel imbalance metric (std across channels)
    imbalance = np.std(per_channel_nave)
    imbalance_metrics.append(imbalance)

    evoked = epochs_nan.average()

    # Compute covariance using MNE
    # cov = mne.compute_covariance(epochs_nan, method="empirical")
    # compute covariance does not allow for NaNs in the data

    # Option A:
    # Compute covariance on all epochs
    # covariance computation drops bad channels and stimulus channels
    cov_a = mne.compute_covariance(epochs, method="empirical", tmax=bmax)

    # fit dipole on evoked data after excluding epochs per channel
    dipole_a, _ = fit_dipole(evoked, cov_a, sphere)

    # Dipole localization error relative to dipole estimated from all epochs
    error_a = np.linalg.norm(dipole_a.pos - est_pos)
    errors_a.append(error_a)

    # Option B: Scale covariance
    # Compute covariance on all epochs
    cov_b = mne.compute_covariance(epochs, method="empirical", tmax=bmax)

    # scale covariance to reflect effective per-channel nave
    cov_b["data"] *= per_channel_nave  # scale matrix

    # fit dipole on evoked data after excluding epochs per channel
    dipole_b, _ = fit_dipole(evoked, cov_b, sphere)

    # Dipole localization error relative to dipole estimated from all epochs
    error_b = np.linalg.norm(dipole_b.pos - est_pos)
    errors_b.append(error_b)

    # Option C: Covariance estimation after rejecting bad epochs per channel


# Plot results from Option A
plt.figure(figsize=(7, 5))
plt.plot(imbalance_metrics, errors_a, "o-", linewidth=2, markersize=8)
for i, p in enumerate(drop_probs):
    plt.text(imbalance_metrics[i] + 0.05, errors_a[i] + 0.05, f"p={p:.1f}")
plt.xlabel("Channel imbalance (std of valid epochs per channel)")
plt.ylabel("Dipole localization error (m)")
plt.title("Option A")
plt.show()

# Option B
plt.figure(figsize=(7, 5))
plt.plot(imbalance_metrics, errors_b, "o-", linewidth=2, markersize=8)
for i, p in enumerate(drop_probs):
    plt.text(imbalance_metrics[i] + 0.05, errors_b[i] + 0.05, f"p={p:.1f}")
plt.xlabel("Channel imbalance (std of valid epochs per channel)")
plt.ylabel("Dipole localization error (m)")
plt.title("Option B")
plt.show()
