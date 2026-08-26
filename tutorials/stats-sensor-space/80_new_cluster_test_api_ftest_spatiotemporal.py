"""
.. _tut-new-cluster-test-api-ftest-spatiotemporal:

===========================================================================
New cluster test API: spatiotemporal permutation F-test on full sensor data
===========================================================================

This tutorial reproduces :ref:`tut-cluster-spatiotemporal-sensor` using the
new :func:`~mne.stats.cluster_test` API instead of
:func:`~mne.stats.spatio_temporal_cluster_test`. It tests for differential
evoked responses in at least one of four conditions using a permutation
clustering test, with the FieldTrip neighbor templates used to determine
sensor adjacency.

Here, the unit of observation is epochs from a specific study subject.
However, the same logic applies when the unit of observation is a number of
study subjects each of whom contribute their own averaged data (i.e., an
average of their epochs). This would then be considered an analysis at the
"2nd level".

See the `FieldTrip tutorial <ft_cluster_>`_ for a caveat regarding the
possible interpretation of "significant" clusters.

For more information on cluster-based permutation testing in MNE-Python, see
also: :ref:`tut-new-cluster-test-api-1samp-tfr`.

.. include:: ../../links.inc
"""
# Authors: The MNE-Python contributors.
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mne
from mne.channels import find_ch_adjacency
from mne.datasets import sample
from mne.stats import cluster_test, combine_adjacency
from mne.viz import plot_compare_evokeds

# %%
# Set parameters
# --------------
data_path = sample.data_path()
meg_path = data_path / "MEG" / "sample"
raw_fname = meg_path / "sample_audvis_filt-0-40_raw.fif"
event_fname = meg_path / "sample_audvis_filt-0-40_raw-eve.fif"
event_id = {"Aud/L": 1, "Aud/R": 2, "Vis/L": 3, "Vis/R": 4}
tmin = -0.2
tmax = 0.5

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 25)
events = mne.read_events(event_fname)

# %%
# Read epochs for the channel of interest
# ---------------------------------------

picks = mne.pick_types(raw.info, meg="mag", eog=True)

reject = dict(mag=4e-12, eog=150e-6)
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    picks=picks,
    decim=2,  # just for speed!
    baseline=None,
    reject=reject,
    preload=True,
)

epochs.drop_channels(["EOG 061"])
epochs.equalize_event_counts(event_id)

# %%
# Prepare the dataframe for the new cluster test API
# ----------------------------------------------------
# We use one row per condition, each holding the full :class:`~mne.Epochs`
# subset for that condition. With 4 groups (and no ``within_id``),
# :func:`~mne.stats.cluster_test` performs a 1-way F-test,
# equivalent to what :func:`~mne.stats.spatio_temporal_cluster_test` did by
# default in the original tutorial.
df = pd.DataFrame(
    dict(data=[epochs[event_name] for event_name in event_id], condition=list(event_id))
)
formula = "data ~ condition"

# %%
# Find the FieldTrip neighbor definition to setup sensor adjacency
# ----------------------------------------------------------------
adjacency, ch_names = find_ch_adjacency(epochs.info, ch_type="mag")

print(type(adjacency))  # it's a sparse matrix!

mne.viz.plot_ch_adjacency(epochs.info, adjacency, ch_names)

# %%
# Compute permutation statistic
# -----------------------------
#
# How does it work? We use clustering to "bind" together features which are
# similar. Our features are the magnetic fields measured over our sensor
# array at different times. This reduces the multiple comparison problem.
# To compute the actual test-statistic, we first sum all F-values in all
# clusters. We end up with one statistic for each cluster.
# Then we generate a distribution from the data by shuffling our conditions
# between our samples and recomputing our clusters and the test statistics.
# We test for the significance of a given cluster by computing the probability
# of observing a cluster of that size
# :footcite:`MarisOostenveld2007,Sassenhagen2019`.

# We are running an F test, so we look at the upper tail
tail = 1

# We want to set a critical test statistic (here: F), to determine when
# clusters are being formed. Using Scipy's percent point function of the F
# distribution, we can conveniently select a threshold that corresponds to
# some alpha level that we arbitrarily pick.
alpha_cluster_forming = 0.001

# For an F test we need the degrees of freedom for the numerator
# (number of conditions - 1) and the denominator (number of observations
# - number of conditions):
n_conditions = len(event_id)
n_observations = len(epochs) // n_conditions
dfn = n_conditions - 1
dfd = n_observations - n_conditions

f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)

# run the cluster based permutation analysis
cluster_result = cluster_test(
    df,
    formula,
    n_permutations=1000,
    threshold=f_thresh,
    tail=tail,
    adjacency=adjacency,
    rng=0,
)

# %%
# .. note:: Note how we only specified an adjacency for sensors! As with the
#           old API, because our data per observation is 2D (times x
#           channels), an adjacency for time points was automatically taken
#           into account (this is also called "lattice adjacency"). For 3D
#           data per observation (e.g., times x frequencies x channels), we
#           need to use :func:`mne.stats.combine_adjacency`, as shown further
#           below.
#
# Visualize clusters
# ------------------
# ``cluster_result.stat_obs`` and ``cluster_result.clusters`` use the same
# ``(times, channels)`` axis order as the old API's ``F_obs``/``clusters``
# here (this only differs for 3D, time-frequency data -- see below and
# :ref:`tut-new-cluster-test-api-1samp-tfr`). Since the 4 conditions here
# aren't organized as paired lists of per-subject evokeds,
# :meth:`~mne.stats.ClusterResult.plot_cluster_time_sensor`
# doesn't apply (it assumes exactly 2 such conditions); instead we adapt the
# manual plotting code from the original tutorial almost unchanged.

p_accept = 0.01
good_cluster_inds = np.where(cluster_result.cluster_p_values < p_accept)[0]

colors = {"Aud": "crimson", "Vis": "steelblue"}
linestyles = {"L": "-", "R": "--"}

evokeds = {cond: epochs[cond].average() for cond in event_id}

for i_clu, clu_idx in enumerate(good_cluster_inds):
    time_inds, space_inds = np.squeeze(cluster_result.clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    f_map = cluster_result.stat_obs[time_inds, ...].mean(axis=0)

    sig_times = epochs.times[time_inds]

    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3), layout="constrained")

    f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epochs.info, tmin=0)
    f_evoked.plot_topomap(
        times=0,
        mask=mask,
        axes=ax_topo,
        cmap="Reds",
        vlim=(np.min, np.max),
        # the data are an F-statistic, not a physical measurement, so disable
        # plot_topomap()'s unit-conversion scaling (e.g. 1e15 for mag)
        scalings=1.0,
        show=False,
        colorbar=False,
        mask_params=dict(markersize=10),
    )
    image = ax_topo.images[0]

    ax_topo.set_title("")

    divider = make_axes_locatable(ax_topo)

    ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        "Averaged F-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
    )

    ax_signals = divider.append_axes("right", size="300%", pad=1.2)
    title = f"Cluster #{i_clu + 1}, {len(ch_inds)} sensor"
    if len(ch_inds) > 1:
        title += "s (mean)"
    plot_compare_evokeds(
        evokeds,
        title=title,
        picks=ch_inds,
        axes=ax_signals,
        colors=colors,
        linestyles=linestyles,
        show=False,
        split_legend=True,
        truncate_yaxis="auto",
    )

    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx(
        (ymin, ymax), sig_times[0], sig_times[-1], color="orange", alpha=0.3
    )

plt.show()

# %%
# Permutation statistic for time-frequencies
# ------------------------------------------
#
# Let's do the same thing with the time-frequency decomposition of the data
# (see :ref:`tut-sensors-time-freq` for a tutorial and
# :ref:`ex-tfr-comparison` for a comparison of time-frequency methods) to
# show how cluster permutations can be done on higher-dimensional data. This
# time we only compare 2 of the 4 conditions.

decim = 4
freqs = np.arange(7, 30, 3)  # define frequencies of interest
n_cycles = freqs / freqs[0]

tfrs = dict()
for condition in ("Aud/L", "Vis/L"):
    this_tfr = epochs[condition].compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        decim=decim,
        average=False,
        return_itc=False,
    )
    this_tfr.apply_baseline(mode="ratio", baseline=(None, 0))
    tfrs[condition] = this_tfr

df_tfr = pd.DataFrame(dict(power=list(tfrs.values()), condition=list(tfrs)))

# %%
# Remember the note above on the adjacency matrix: for 3D data, as here, we
# must use :func:`mne.stats.combine_adjacency` to extend the sensor-based
# adjacency to incorporate the time-frequency plane as well.
#
# The one thing that differs from the old API here is dimension order: the
# new API always represents a single observation's time-frequency data as
# ``(times, freqs, channels)`` (channels last), so where the original
# tutorial builds ``combine_adjacency(n_freqs, n_times, adjacency)``, here we
# build it as ``combine_adjacency(n_times, n_freqs, adjacency)`` -- see also
# :ref:`tut-new-cluster-test-api-1samp-tfr`.
tfr_adjacency = combine_adjacency(len(this_tfr.times), len(freqs), adjacency)

# %%
# Now we can run the cluster permutation test, but first we have to set a
# threshold. This example decimates in time and uses few frequencies so we
# need to increase the threshold from the default value in order to have
# differentiated clusters (i.e., so that our algorithm doesn't just find one
# large cluster). For a more principled method of setting this parameter,
# threshold-free cluster enhancement may be used. See :ref:`disc-stats` for a
# discussion.

# This time we don't calculate a threshold based on the F distribution.
# We might as well select an arbitrary threshold for cluster forming
tfr_threshold = 15.0

cluster_result_tfr = cluster_test(
    df_tfr,
    "power ~ condition",
    n_permutations=1000,
    threshold=tfr_threshold,
    tail=1,
    adjacency=tfr_adjacency,
    rng=0,
)

# %%
# Finally, we can plot our results using
# :meth:`~mne.stats.ClusterResult.plot_cluster_time_frequency`. By
# default (``cluster_idx=0``) it shows the largest significant cluster (an
# F-statistic is non-negative, so ranking by cluster mass here simply means
# ranking by size): a topomap of the statistic averaged over that cluster's
# time-frequency extent, next to a spectrogram for the single channel (among
# the cluster's channels) with the most extreme statistic, highlighting that
# cluster's time-frequency extent (and that of any other significant cluster
# that also has a member point on the same channel). As in the original
# tutorial, keep in mind that each sensor has its own significant
# time-frequencies, but, in order to display a single spectrogram, all the
# time-frequencies that are significant on the displayed sensor are shown.
# This is a difficulty inherent to visualizing high-dimensional data and
# should be taken into consideration when interpreting results.
print(f"The lowest cluster p-value is: {cluster_result_tfr.cluster_p_values.min()}")
cluster_result_tfr.plot_cluster_time_frequency(this_tfr)

# %%
# References
# ----------
# .. footbibliography::
