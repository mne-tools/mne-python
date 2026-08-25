"""
.. _tut-new-cluster-test-api-1samp-tfr:

======================================================================
New cluster test API: 1-sample cluster statistic on single trial power
======================================================================

This tutorial reproduces :ref:`tut-cluster-one-samp-tfr` using the new
:func:`~mne.stats.cluster_test` API instead of
:func:`~mne.stats.permutation_cluster_1samp_test`. It estimates significant
clusters in time-frequency power estimates using a non-parametric permutation
procedure.

The procedure consists of:

  - extracting epochs
  - computing single trial power estimates
  - baseline correcting the power estimates (power ratios)
  - building a :class:`pandas.DataFrame` with one row holding all epochs
  - running :func:`~mne.stats.cluster_test` to see if the ratio
    deviates from 1
  - plotting the resulting cluster with
    :meth:`~mne.stats.ClusterResult.plot_cluster_time_frequency`

Here, the unit of observation is epochs from a specific study subject.
However, the same logic applies when the unit of observation is a number of
study subjects each of whom contribute their own averaged data (i.e., an
average of their epochs). This would then be considered an analysis at the
"2nd level" -- see :ref:`tut-new-cluster-test-api` for an example of a 2nd
level analysis with the new API.

For more information on cluster-based permutation testing in MNE-Python, see
also: :ref:`tut-cluster-spatiotemporal-sensor`.
"""
# Authors: The MNE-Python contributors.
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import numpy as np
import pandas as pd
import scipy.stats

import mne
from mne.datasets import sample
from mne.stats import cluster_test

# %%
# Set parameters
# --------------

data_path = sample.data_path()
meg_path = data_path / "MEG" / "sample"
raw_fname = meg_path / "sample_audvis_raw.fif"
tmin, tmax, event_id = -0.3, 0.6, 1

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname)
events = mne.find_events(raw, stim_channel="STI 014")

raw.info["bads"] += ["MEG 2443", "EEG 053"]  # bads + 2 more

# for speed, we'll only look at right-temporal gradiometers (and EOG)
picks_eog = mne.pick_types(raw.info, eog=True)
picks_grad = mne.pick_types(raw.info, meg="grad", exclude="bads")
picks_rtemp = mne.pick_channels(
    raw.info["ch_names"], mne.read_vectorview_selection("Right-temporal"), ordered=True
)
picks = list((set(picks_rtemp) & set(picks_grad)) | set(picks_eog))

# Load condition 1
event_id = 1
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    picks=picks,
    baseline=(None, 0),
    preload=True,
    reject=dict(grad=4000e-13, eog=150e-6),
)

evoked = epochs.average()

# Factor to down-sample the temporal dimension of the TFR. Decimation occurs
# after frequency decomposition and can be used to reduce memory usage (and
# possibly computational time of downstream operations such as nonparametric
# statistics) if you don't need high spectrotemporal resolution.
decim = 5

# define frequencies of interest
freqs = np.arange(8, 40, 2)

# run the TFR decomposition -- note ``average=False``, so we keep all trials
tfr_epochs = epochs.compute_tfr(
    "morlet",
    freqs,
    n_cycles=4.0,
    decim=decim,
    average=False,
    return_itc=False,
    n_jobs=None,
)

# Baseline power
tfr_epochs.apply_baseline(mode="logratio", baseline=(-0.100, 0))

# Crop in time to keep only what is between 0 and 400 ms
evoked.crop(-0.1, 0.4)
tfr_epochs.crop(-0.1, 0.4)

# %%
# Define adjacency for statistics
# -------------------------------
# To perform a cluster-based permutation test, we need a suitable definition
# for the adjacency of sensors, time points, and frequency bins. Just like
# with the old API, :func:`~mne.stats.cluster_test` does not
# build this adjacency for us -- we compute the sensor adjacency, then
# combine it with a "lattice" adjacency for the time-frequency plane
# ourselves, exactly like in :ref:`tut-cluster-one-samp-tfr`.
#
# The one thing that *does* differ from the old API is dimension order: the
# new API always represents a single observation's data as
# ``(times, freqs, channels)`` (i.e., the channel axis is always last), no
# matter what order the underlying MNE-Python object stores its data in. So
# where the old tutorial builds ``combine_adjacency(sensor_adjacency,
# n_freqs, n_times)``, here we build it as ``combine_adjacency(n_times,
# n_freqs, sensor_adjacency)``.
sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(tfr_epochs.info, "grad")
use_idx = [ch_names.index(ch_name) for ch_name in tfr_epochs.ch_names]
sensor_adjacency = sensor_adjacency[use_idx][:, use_idx]
assert sensor_adjacency.shape == (len(tfr_epochs.ch_names), len(tfr_epochs.ch_names))

adjacency = mne.stats.combine_adjacency(
    len(tfr_epochs.times), len(tfr_epochs.freqs), sensor_adjacency
)
assert (
    adjacency.shape[0]
    == adjacency.shape[1]
    == len(tfr_epochs.times) * len(tfr_epochs.freqs) * len(tfr_epochs.ch_names)
)

# %%
# Prepare the dataframe for the new cluster test API
# ----------------------------------------------------
# The new API always needs a dependent variable ("data") column and an
# independent variable column, matched by a Wilkinson ``formula``. For a
# genuine one-sample test like this one -- where we simply want to know
# whether the (already baseline-normalized) power deviates from 1 -- there is
# no real "condition" to compare against. We represent this by putting *all*
# epochs into a single :class:`~mne.time_frequency.EpochsTFR` in one row, with
# a placeholder, single-valued "group" column. When the independent variable
# has only one unique value, :func:`~mne.stats.cluster_test`
# recognizes this as a 1-sample design and runs a paired t-test against zero,
# exactly like :func:`~mne.stats.permutation_cluster_1samp_test` did above.
df = pd.DataFrame(dict(power=[tfr_epochs], group=["induced"]))
formula = "power ~ group"

# %%
# Compute statistic
# -----------------
# For forming clusters, we need to specify a critical test statistic
# threshold, exactly as in the old API.

# We want a two-tailed test
tail = 0

degrees_of_freedom = len(epochs) - 1
t_thresh = scipy.stats.t.ppf(1 - 0.001 / 2, df=degrees_of_freedom)

# Warning: 50 is way too small for a real-world analysis (where values of
# 5000 or higher are used), but here we use it to increase computation speed.
n_permutations = 50

cluster_result = cluster_test(
    df,
    formula,
    threshold=t_thresh,
    tail=tail,
    adjacency=adjacency,
    n_permutations=n_permutations,
    seed=0,
    verbose=True,
)

# %%
# View time-frequency plots
# -------------------------
# We now visualize the most significant cluster using
# :meth:`~mne.stats.ClusterResult.plot_cluster_time_frequency`. By
# default (``cluster_idx=0``) this shows the cluster with the largest *mass*
# (the sum of the observed statistic within the cluster -- the same quantity
# used internally to compute the cluster p-values), not necessarily the one
# with the lowest p-value. Other significant clusters can be inspected by
# passing a different ``cluster_idx``, ranked the same way.
#
# Like :ref:`tut-cluster-one-samp-tfr`, the plot shows the spectrogram for the
# single channel with the most extreme statistic within the chosen cluster
# (here, automatically picked from among the cluster's channels rather than
# hand-picked), with that cluster's time-frequency extent highlighted -- next
# to a topomap of the statistic averaged over the cluster's full
# time-frequency extent, showing which other channels were also part of it.
# If another significant cluster also has a member point on the displayed
# channel, it is highlighted there too, so no significant effect on that
# channel is hidden.
#
# .. warning:: Talking about "significant clusters" can be convenient, but
#              you must be aware of all associated caveats! For example, it
#              is **invalid** to interpret the cluster p value as being
#              spatially or temporally specific. See the comprehensive
#              `FieldTrip tutorial <ft_cluster_>`_ for more information.
#
# .. include:: ../../links.inc
print(f"The lowest cluster p-value is: {cluster_result.cluster_p_values.min()}")
cluster_result.plot_cluster_time_frequency(tfr_epochs, cluster_idx=0)
print(f"The second lowest cluster p-value is: {cluster_result.cluster_p_values[1]}")
cluster_result.plot_cluster_time_frequency(tfr_epochs, cluster_idx=1)

# %%
# As in :ref:`tut-cluster-one-samp-tfr`, it is also informative to look at the
# evoked (i.e., phase-locked) response over the same channels and time window,
# for context: the induced power increase detected above need not be
# accompanied by an evoked response, and vice versa.
evoked.plot()
