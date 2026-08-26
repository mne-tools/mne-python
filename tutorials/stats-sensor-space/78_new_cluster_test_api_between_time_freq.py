"""
.. _tut-new-cluster-test-api-between-tfr:

===========================================================================
New cluster test API: between-conditions cluster statistic on single trials
===========================================================================

This tutorial reproduces :ref:`tut-cluster-tfr` using the new
:func:`~mne.stats.cluster_test` API instead of
:func:`~mne.stats.permutation_cluster_test`. It compares clusters in
time-frequency power estimates between two conditions using a non-parametric
permutation procedure.

The procedure consists of:

  - extracting epochs for 2 conditions
  - computing single trial power estimates
  - baseline correcting the power estimates (power ratios)
  - building a :class:`pandas.DataFrame` with one row per condition, and
    running :func:`~mne.stats.cluster_test` to see if the power
    estimates are significantly different between conditions
"""
# Authors: The MNE-Python contributors.
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mne
from mne.datasets import sample
from mne.stats import cluster_test

# %%
# Set parameters
data_path = sample.data_path()
meg_path = data_path / "MEG" / "sample"
raw_fname = meg_path / "sample_audvis_raw.fif"
event_fname = meg_path / "sample_audvis_raw-eve.fif"
tmin, tmax = -0.2, 0.5

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

raw.info["bads"] += ["MEG 2443", "EEG 053"]  # bads + 2 more

# picks MEG gradiometers
picks = mne.pick_types(
    raw.info, meg="grad", eeg=False, eog=True, stim=False, exclude="bads"
)

ch_name = "MEG 1332"  # restrict example to one channel

# Load condition 1
reject = dict(grad=4000e-13, eog=150e-6)
event_id = 1
epochs_condition_1 = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    picks=picks,
    baseline=(None, 0),
    reject=reject,
    preload=True,
)
epochs_condition_1.pick([ch_name])

# Load condition 2
event_id = 2
epochs_condition_2 = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    picks=picks,
    baseline=(None, 0),
    reject=reject,
    preload=True,
)
epochs_condition_2.pick([ch_name])

# %%
# Factor to downsample the temporal dimension of the TFR. Decimation occurs
# after frequency decomposition and can be used to reduce memory usage (and
# possibly computational time of downstream operations such as nonparametric
# statistics) if you don't need high spectrotemporal resolution.
decim = 2
freqs = np.arange(7, 30, 3)  # define frequencies of interest
n_cycles = 1.5
tfr_kwargs = dict(
    method="morlet",
    freqs=freqs,
    n_cycles=n_cycles,
    decim=decim,
    return_itc=False,
    average=False,
)

tfr_epochs_1 = epochs_condition_1.compute_tfr(**tfr_kwargs)
tfr_epochs_2 = epochs_condition_2.compute_tfr(**tfr_kwargs)

tfr_epochs_1.apply_baseline(mode="ratio", baseline=(None, 0))
tfr_epochs_2.apply_baseline(mode="ratio", baseline=(None, 0))

# %%
# Prepare the dataframe for the new cluster test API
# ----------------------------------------------------
# We use one row per condition, each holding the full (multi-epoch)
# :class:`~mne.time_frequency.EpochsTFR` object for that condition. Because
# there is no ``within_id`` and there are exactly 2 groups,
# :func:`~mne.stats.cluster_test` performs an unpaired test,
# equivalent to the 1-way ANOVA (F-test) that
# :func:`~mne.stats.permutation_cluster_test` performs by default.
df = pd.DataFrame(
    dict(power=[tfr_epochs_1, tfr_epochs_2], condition=["condition_1", "condition_2"])
)
formula = "power ~ condition"

# %%
# Compute statistic
# -----------------
threshold = 6.0
cluster_result = cluster_test(
    df,
    formula,
    threshold=threshold,
    tail=0,
    n_permutations=100,
    out_type="mask",
    rng=np.random.default_rng(seed=8675309),
)

# %%
# View time-frequency plots
# -------------------------
# The single channel we picked doesn't lend itself well to
# :meth:`~mne.stats.ClusterResult.plot_cluster_time_frequency`
# (which shows a topomap of the cluster's spatial extent), so here we build a
# plot directly from :class:`~mne.stats.ClusterResult`'s raw
# attributes, just like the original tutorial did with the plain arrays
# returned by :func:`~mne.stats.permutation_cluster_test`. The main
# difference is dimension order: ``cluster_result.stat_obs`` has shape
# ``(times, freqs, channels)`` rather than ``(freqs, times)``.
times = 1e3 * epochs_condition_1.times  # change unit to ms

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 4), layout="constrained")

# Compute the difference in evoked power to determine which condition was
# greater, since the F-test only tells us there *is* a difference
power_1 = tfr_epochs_1.data[:, 0].mean(axis=0)  # avg over epochs, 1 channel
power_2 = tfr_epochs_2.data[:, 0].mean(axis=0)
signs = np.sign(power_1 - power_2).T  # transpose to (times, freqs)

F_obs = cluster_result.stat_obs[..., 0]  # only 1 channel; now (times, freqs)
F_obs_plot = np.nan * np.ones_like(F_obs)
for c, p_val in zip(cluster_result.clusters, cluster_result.cluster_p_values):
    if p_val <= 0.05:
        c = c[..., 0]  # squeeze the (singleton) channel dimension
        F_obs_plot[c] = F_obs[c] * signs[c]

# transpose everything to (freqs, times) for display, matching the original
ax.imshow(
    F_obs.T,
    extent=[times[0], times[-1], freqs[0], freqs[-1]],
    aspect="auto",
    origin="lower",
    cmap="gray",
)
max_F = np.nanmax(np.abs(F_obs_plot))
ax.imshow(
    F_obs_plot.T,
    extent=[times[0], times[-1], freqs[0], freqs[-1]],
    aspect="auto",
    origin="lower",
    cmap="RdBu_r",
    vmin=-max_F,
    vmax=max_F,
)

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title(f"Induced power ({ch_name})")

# plot evoked
evoked_condition_1 = epochs_condition_1.average()
evoked_condition_2 = epochs_condition_2.average()
evoked_contrast = mne.combine_evoked(
    [evoked_condition_1, evoked_condition_2], weights=[1, -1]
)
evoked_contrast.plot(axes=ax2, time_unit="s")
