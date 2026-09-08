"""
.. _tut-new-cluster-test-api:

===============================================================
Group-level cluster permutation testing with formula contrasts
===============================================================

Run a cluster-based permutation test on evoked data from several subjects,
specifying the contrast with a Wilkinson (R-style) formula. By the end you
will have run a paired *t*-test across subjects and inspected the cluster
permutation results.

You will:

- load evoked data from multiple subjects
- build a long-format dataframe with one row per subject and condition
- run :func:`mne.stats.cluster_test` with a Wilkinson-notation formula
- inspect the cluster permutation results
"""
# Author: Carina Forster <carinaforster0611@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%
# Load the required packages
# --------------------------

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mne

# %%
# Load evoked data for multiple subjects
# --------------------------------------
#
# We use the P3b data from the freely available ERP CORE dataset: a visual
# oddball task contrasting rare *target* stimuli with frequent *non-target*
# stimuli. Each subject has an evoked response for both conditions; five
# subjects are included here.

# path to the P3 dataset
path_to_p3 = mne.datasets.misc.data_path() / "ERP_CORE" / "P3"

# participant IDs available in this dataset (15 to 19)
participant_ids = range(15, 20)

# load each subject's evoked data into a list
evokeds_allsubs = []
for pid in participant_ids:
    # filename with the ID zero-padded to three digits
    filename_p3 = f"sub-{pid:03d}_ses-P3_task-P3_ave.fif"
    p3_file_path = Path(path_to_p3) / filename_p3
    evokeds = mne.read_evokeds(p3_file_path)
    evokeds_allsubs.append(evokeds)

# split the two conditions into separate per-subject lists
target_only = [evoked[0] for evoked in evokeds_allsubs]
non_target_only = [evoked[1] for evoked in evokeds_allsubs]

# %%
# Inspect the contrast before testing
# -----------------------------------
#
# Before running any statistics, look at the effect you are about to test.
# We form the per-subject difference (target minus non-target) and plot its
# grand average. A positive deflection means targets evoke the stronger
# response.

diff_evoked = [
    mne.combine_evoked([evoked_target, evoked_non_target], weights=[1, -1])
    for evoked_target, evoked_non_target in zip(target_only, non_target_only)
]

grand_avg_diff = mne.grand_average(diff_evoked)
grand_avg_diff.plot()
grand_avg_diff.plot_topomap()

# %%
# You should see the largest difference around 400 ms over central-parietal
# channels -- the expected P3b effect, stronger for target stimuli. This is the
# contrast the cluster test will evaluate formally.
#
# ``diff_evoked`` is used only for this visualization. The cluster test below
# works from a dataframe holding *both* conditions and forms the contrast from
# the formula.

# %%
# Build the dataframe for the cluster test
# ----------------------------------------
#
# The formula interface takes a long-format :class:`pandas.DataFrame` with one
# row per observation. Each row holds one subject's evoked response for one
# condition, so every subject contributes two rows (target and non-target).
# Every subject must contribute the same set of conditions. The columns are:
#
# - ``evoked``: the single-subject :class:`~mne.Evoked` object
# - ``condition``: the condition label, referenced by the formula
# - ``subject_index``: identifies which observations are paired within a subject

evokeds_conditions = target_only + non_target_only
conditions = ["target"] * len(target_only) + ["non-target"] * len(non_target_only)
subject_index = list(participant_ids) * 2

df = pd.DataFrame(
    {
        "evoked": evokeds_conditions,
        "condition": conditions,
        "subject_index": subject_index,
    }
)
df

# %%
# You should see the largest difference around 400 ms over central-parietal
# channels -- the expected P3b effect, stronger for target stimuli. This is the
# contrast the cluster test will evaluate formally.
#
# ``diff_evoked`` is used only for this visualization. The cluster test below
# works from a dataframe holding *both* conditions and forms the contrast from
# the formula.

# %%
# The sign of the contrast follows the order of the condition levels. We set
# "target" as the first level so the difference is formed as target minus
# non-target (positive = stronger response to targets), matching the grand
# average we plotted above.

# TODO: do this within cluster test?
df["condition"] = pd.Categorical(
    df["condition"], categories=["target", "non-target"], ordered=True
)

df

# %%
# Run the cluster test with a formula
# -----------------------------------
#
# The contrast is written as a Wilkinson (R-style) formula, the same notation
# used by R's ``lmer``/``glmer``. Here ``"evoked ~ condition"`` models the
# evoked response as a function of condition: ``condition`` is categorical and
# is dummy-coded automatically, and an intercept is included implicitly.
# Passing ``within_id="subject_index"`` makes this a within-subject (paired)
# test: the two conditions are subtracted within each subject and the resulting
# differences are tested against zero (a one-sample t-test), with the null
# distribution built by sign-flipping those per-subject differences. Because we
# set ``target`` as the first condition level above, the difference is formed as
# target minus non-target. TODO: should be a parameter in cluster_test?

formula = "evoked ~ condition"

cluster_result = mne.stats.cluster_test(
    df=df, formula=formula, within_id="subject_index"
)

print(f"Smallest cluster p-value: {cluster_result.cluster_p_values.min():.4f}")

# %%
# The smallest cluster p-value is about 0.06, so no cluster is significant at
# alpha = 0.05 -- and with five subjects none ever could be. Here is why.
#
# The null distribution is built by sign-flipping the five per-subject
# difference scores. There are ``2 ** 5 = 32`` ways to assign signs, but
# flipping every sign only mirrors the partition, so just ``2 ** (5 - 1) = 16``
# are distinct; excluding the observed arrangement leaves
# ``2 ** (5 - 1) - 1 = 15`` permutations. Because the test is exact, all 15 are
# evaluated (you will see ``15/15`` in the progress log) rather than sampled at
# random.
#
# The finest p-value this can resolve is ``1 / (15 + 1) = 0.0625``, so even the
# most extreme possible cluster lands just above 0.05. The near-0.0625 result
# means the observed cluster *was* the most extreme one -- there is simply not
# enough data to reach significance. Detecting an effect here would need more
# subjects; that, not the specific p-value, is the takeaway.

# %%
# Inspect the results
# -------------------
#
# The result object carries the observed cluster-level statistics. We plot the
# observed t-values as a heatmap with time on the x-axis and channel names on
# the y-axis. Because the contrast is target minus non-target, positive t-values
# mean a stronger response to targets, matching the difference plotted earlier.

print(
    f"Number of permutations run: {cluster_result.n_permutations}"
)  # TODO: fix this in separate PR

# times (in seconds) and channel names come from the evoked data
times = grand_avg_diff.times
ch_names = grand_avg_diff.ch_names

# stat_obs holds the observed t-values; ensure it is arranged as (channels, times)
stat_obs = cluster_result.stat_obs
if stat_obs.shape != (len(ch_names), len(times)):
    stat_obs = stat_obs.T

# symmetric colour limits so the diverging colormap is centred on zero
vlim = np.abs(stat_obs).max()

fig, ax = plt.subplots(layout="constrained")
im = ax.imshow(
    stat_obs,
    aspect="auto",
    origin="lower",
    extent=[times[0], times[-1], 0, len(ch_names)],
    cmap="RdBu_r",
    vmin=-vlim,
    vmax=vlim,
)
ax.set_yticks(np.arange(len(ch_names)) + 0.5)
ax.set_yticklabels(ch_names)
ax.set_xlabel("time (s)")
ax.set_ylabel("channel")
ax.set_title("Observed cluster statistic (target - non-target)")
fig.colorbar(im, ax=ax, label="t-value")
