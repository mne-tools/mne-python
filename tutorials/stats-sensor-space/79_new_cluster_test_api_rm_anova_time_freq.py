"""
.. _tut-new-cluster-test-api-rm-anova-tfr:

=====================================================================
New cluster test API: mass-univariate two-way repeated measures ANOVA
=====================================================================

This tutorial reproduces :ref:`tut-timefreq-twoway-anova` using the new
:func:`~mne.stats.cluster_test` API instead of
:func:`~mne.stats.permutation_cluster_test` with a hand-rolled ``stat_fun``.
As in the original tutorial, the model assumes two fully crossed factors --
perceptual modality (auditory vs. visual) and location of stimulus
presentation (left vs. right) -- and single trials are used as replications
("subjects") while iterating over time-frequency bins for a single channel.

:func:`~mne.stats.cluster_test` supports this design directly:
a formula naming a single interaction term, e.g. ``"power ~ modality:location"``,
combined with ``within_id`` naming the replication/subject column, dispatches
internally to :func:`~mne.stats.f_mway_rm` with the correct ``factor_levels``
and ``effects`` -- no custom ``stat_fun`` needed.

We conclude, as in the original, by comparing the cluster-corrected result to
multiple-comparisons correction via False Discovery Rate.
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
from mne.stats import cluster_test, f_mway_rm, f_threshold_mway_rm, fdr_correction
from mne.time_frequency import AverageTFRArray

# %%
# Set parameters
# --------------
data_path = sample.data_path()
meg_path = data_path / "MEG" / "sample"
raw_fname = meg_path / "sample_audvis_raw.fif"
event_fname = meg_path / "sample_audvis_raw-eve.fif"
tmin, tmax = -0.2, 0.5

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

raw.info["bads"] += ["MEG 2443"]  # bads

# picks MEG gradiometers
picks = mne.pick_types(
    raw.info, meg="grad", eeg=False, eog=True, stim=False, exclude="bads"
)

ch_name = "MEG 1332"

# Load conditions
reject = dict(grad=4000e-13, eog=150e-6)
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    picks=picks,
    baseline=(None, 0),
    preload=True,
    reject=reject,
)
epochs.pick([ch_name])  # restrict example to one channel

# %%
# We have to make sure all conditions have the same counts, as the ANOVA
# expects a fully balanced data matrix and does not forgive imbalances that
# generously (risk of type-I error).
epochs.equalize_event_counts(event_id)

decim = 2
freqs = np.arange(7, 30, 3)  # define frequencies of interest
n_cycles = freqs / freqs[0]
zero_mean = False  # don't correct morlet wavelet to be of mean zero

# %%
# Create TFR representations for all conditions
# ---------------------------------------------
epochs_power = list()
for condition in [epochs[k] for k in event_id]:
    this_tfr = condition.compute_tfr(
        "morlet",
        freqs,
        n_cycles=n_cycles,
        decim=decim,
        average=False,
        zero_mean=zero_mean,
        return_itc=False,
    )
    this_tfr.apply_baseline(mode="ratio", baseline=(None, 0))
    this_power = this_tfr.data[:, 0, :, :]  # we only have one channel.
    epochs_power.append(this_power)

# keep a reference TFR object around, purely to reuse its info/times/freqs
# below (all 4 conditions share the same channel, times, and freqs)
tfr_info, tfr_times, tfr_freqs = this_tfr.info, this_tfr.times, this_tfr.freqs

# %%
# Setup repeated measures ANOVA
# -----------------------------
#
# As in the original tutorial, we first compute naive (uncorrected)
# mass-univariate F-images for all three effects, using
# :func:`~mne.stats.f_mway_rm` directly -- this part doesn't involve
# clustering at all, so it is unchanged from the original.

n_conditions = len(epochs.event_id)
n_replications = epochs.events.shape[0] // n_conditions

factor_levels = [2, 2]  # number of levels in each factor
effects = "A*B"  # compute all effects: main effects A, B, and interaction A:B
n_freqs = len(freqs)
times = 1e3 * epochs.times[::decim]
n_times = len(times)

# assemble the data matrix and swap axes so replications are the first
# dimension and conditions are the second dimension
data = np.swapaxes(np.asarray(epochs_power), 1, 0)
print(data.shape)  # replications x conditions x (freqs x times)

fvals, pvals = f_mway_rm(data, factor_levels, effects=effects)

effect_labels = ["modality", "location", "modality by location"]

fig, axes = plt.subplots(3, 1, figsize=(6, 6), layout="constrained")

for effect, sig, effect_label, ax in zip(fvals, pvals, effect_labels, axes):
    ax.imshow(
        effect,
        cmap="gray",
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
    )
    effect[sig >= 0.05] = np.nan
    c = ax.imshow(
        effect,
        cmap="autumn",
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f'Time-locked response for "{effect_label}" ({ch_name})')

# %%
# Account for multiple comparisons using FDR versus permutation clustering
# --------------------------------------------------------------------------
#
# Now we restrict the analysis to the interaction effect and correct for
# multiple comparisons using a cluster-based permutation test, via the new
# :func:`~mne.stats.cluster_test` API.
#
# Build a dataframe with one row per single trial: the trial's time-frequency
# power (as an :class:`~mne.time_frequency.AverageTFRArray`, since each row
# is a single "replication"), which factor levels it belongs to
# (``modality``, ``location``), and a ``replication`` index. Trials are
# already balanced across conditions (see ``equalize_event_counts`` above),
# so we simply use each condition's trial order (0, 1, ..., n_replications -
# 1) as the replication index -- exactly the same positional pairing that
# :func:`~mne.stats.f_mway_rm` uses internally in the original tutorial.
rows = list()
for cond_name, cond_data in zip(event_id, epochs_power):
    modality, location = cond_name.split("_")
    for i in range(n_replications):
        rows.append(
            dict(
                power=AverageTFRArray(
                    info=tfr_info,
                    # restore the (singleton) channel axis dropped above
                    data=cond_data[i][np.newaxis],
                    times=tfr_times,
                    freqs=tfr_freqs,
                ),
                modality=modality,
                location=location,
                replication=i,
            )
        )
df = pd.DataFrame(rows)

# The ANOVA returns a tuple of f-values and p-values; f_threshold_mway_rm
# gives us the cluster-forming threshold from the f-values' null distribution
pthresh = 0.001  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_replications, factor_levels, "A:B", pthresh)
tail = 1  # f-test, so tail > 0
n_permutations = 256  # Save some time (the test won't be too sensitive ...)

cluster_result = cluster_test(
    df,
    "power ~ modality:location",
    within_id="replication",
    threshold=f_thresh,
    tail=tail,
    n_permutations=n_permutations,
    out_type="mask",
    seed=0,
)
print(cluster_result.stat_name)  # "F-statistic (repeated-measures ANOVA)"

# %%
# Create new stats image with only significant clusters. As in
# :ref:`tut-new-cluster-test-api-between-tfr`, we use
# ``cluster_result``'s raw attributes directly since this is single-channel
# data -- ``cluster_result.stat_obs`` has shape ``(times, freqs, 1)``, so we
# squeeze out (and transpose away) the channel dimension to match the
# ``(freqs, times)`` images plotted above.
F_obs = cluster_result.stat_obs[..., 0].T
F_obs_plot = np.full_like(F_obs, np.nan)
for c, p_val in zip(cluster_result.clusters, cluster_result.cluster_p_values):
    if p_val <= 0.05:
        c = c[..., 0].T
        F_obs_plot[c] = F_obs[c]

fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
for f_image, cmap in zip([F_obs, F_obs_plot], ["gray", "autumn"]):
    c = ax.imshow(
        f_image,
        cmap=cmap,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
    )

fig.colorbar(c, ax=ax)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title(
    f'Time-locked response for "modality by location" ({ch_name})\n'
    "cluster-level corrected (p <= 0.05)"
)

# %%
# Now using FDR:
#
# .. note:: We use ``F_obs`` (``cluster_result``'s own observed statistic,
#    computed fresh above) rather than ``fvals[2]`` as the gray background
#    image here, because the naive-F-image plotting loop above mutated
#    ``fvals`` in place (setting non-significant bins to ``nan``).

mask, _ = fdr_correction(pvals[2])
F_obs_plot2 = F_obs.copy()
F_obs_plot2[~mask.reshape(F_obs_plot2.shape)] = np.nan

fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
for f_image, cmap in zip([F_obs, F_obs_plot2], ["gray", "autumn"]):
    c = ax.imshow(
        f_image,
        cmap=cmap,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
    )

fig.colorbar(c, ax=ax)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title(
    f'Time-locked response for "modality by location" ({ch_name})\n'
    "FDR corrected (p <= 0.05)"
)

# %%
# Both cluster-level and FDR correction help get rid of potential
# false-positives that we saw in the naive f-images. The cluster permutation
# correction is biased toward time-frequencies with contiguous areas of high
# or low power, which is likely appropriate given the highly correlated nature
# of this data.
