# -*- coding: utf-8 -*-
"""
.. _tut-cluster-spatiotemporal-sensor:

=====================================================
Spatiotemporal permutation F-test on full sensor data
=====================================================

Tests for differential evoked responses in at least
one condition using a permutation clustering test.
The FieldTrip neighbor templates will be used to determine
the adjacency between sensors. This serves as a spatial prior
to the clustering. Spatiotemporal clusters will then
be visualized using custom matplotlib code.

Here, the unit of observation is epochs from a specific study subject.
However, the same logic applies when the unit observation is
a number of study subject each of whom contribute their own averaged
data (i.e., an average of their epochs). This would then be considered
an analysis at the "2nd level".

See the `FieldTrip tutorial <ft_cluster_>`_ for a caveat regarding
the possible interpretation of "significant" clusters.

For more information on cluster-based permutation testing in MNE-Python,
see also: :ref:`tut-cluster-one-samp-tfr`
"""
# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Alex Rockhill <aprockhill@mailbox.org>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause

# %%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats

import mne
from mne.stats import spatio_temporal_cluster_test, combine_adjacency
from mne.datasets import sample
from mne.channels import find_ch_adjacency
from mne.viz import plot_compare_evokeds
from mne.time_frequency import tfr_morlet

# %%
# Set parameters
# --------------
data_path = sample.data_path()
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'
event_fname = meg_path / 'sample_audvis_filt-0-40_raw-eve.fif'
event_id = {'Aud/L': 1, 'Aud/R': 2, 'Vis/L': 3, 'Vis/R': 4}
tmin = -0.2
tmax = 0.5

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 30)
events = mne.read_events(event_fname)

# %%
# Read epochs for the channel of interest
# ---------------------------------------

picks = mne.pick_types(raw.info, meg='mag', eog=True)

reject = dict(mag=4e-12, eog=150e-6)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=None, reject=reject, preload=True)

epochs.drop_channels(['EOG 061'])
epochs.equalize_event_counts(event_id)

# Obtain the data as a 3D matrix and transpose it such that
# the dimensions are as expected for the cluster permutation test:
# n_epochs × n_times × n_channels
X = [epochs[event_name].get_data() for event_name in event_id]
X = [np.transpose(x, (0, 2, 1)) for x in X]


# %%
# Find the FieldTrip neighbor definition to setup sensor adjacency
# ----------------------------------------------------------------
adjacency, ch_names = find_ch_adjacency(epochs.info, ch_type='mag')

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
# see also: https://stats.stackexchange.com/a/73993
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
n_observations = len(X[0])
dfn = n_conditions - 1
dfd = n_observations - n_conditions

# Note: we calculate 1 - alpha_cluster_forming to get the critical value
# on the right tail
f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)

# run the cluster based permutation analysis
cluster_stats = spatio_temporal_cluster_test(X, n_permutations=1000,
                                             threshold=f_thresh, tail=tail,
                                             n_jobs=None, buffer_size=None,
                                             adjacency=adjacency)
F_obs, clusters, p_values, _ = cluster_stats

# %%
# .. note:: Note how we only specified an adjacency for sensors! However,
#           because we used :func:`mne.stats.spatio_temporal_cluster_test`,
#           an adjacency for time points was automatically taken into
#           account. That is, at time point N, the time points N - 1 and
#           N + 1 were considered as adjacent (this is also called "lattice
#           adjacency"). This is only possible because we ran the analysis on
#           2D data (times × channels) per observation ... for 3D data per
#           observation (e.g., times × frequencies × channels), we will need
#           to use :func:`mne.stats.combine_adjacency`, as shown further
#           below.
#
# Note also that the same functions work with source estimates.
# The only differences are the origin of the data, the size,
# and the adjacency definition.
# It can be used for single trials or for groups of subjects.
#
# Visualize clusters
# ------------------

# We subselect clusters that we consider significant at an arbitrarily
# picked alpha level: "p_accept".
# NOTE: remember the caveats with respect to "significant" clusters that
# we mentioned in the introduction of this tutorial!
p_accept = 0.01
good_cluster_inds = np.where(p_values < p_accept)[0]

# configure variables for visualization
colors = {"Aud": "crimson", "Vis": 'steelblue'}
linestyles = {"L": '-', "R": '--'}

# organize data for plotting
evokeds = {cond: epochs[cond].average() for cond in event_id}

# loop over clusters
for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for F stat
    f_map = F_obs[time_inds, ...].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = epochs.times[time_inds]

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

    # plot average test statistic and mark significant sensors
    f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epochs.info, tmin=0)
    f_evoked.plot_topomap(times=0, mask=mask, axes=ax_topo, cmap='Reds',
                          vlim=(np.min, np.max), show=False,
                          colorbar=False, mask_params=dict(markersize=10))
    image = ax_topo.images[0]

    # remove the title that would otherwise say "0.000 s"
    ax_topo.set_title("")

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        'Averaged F-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('right', size='300%', pad=1.2)
    title = 'Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += "s (mean)"
    plot_compare_evokeds(evokeds, title=title, picks=ch_inds, axes=ax_signals,
                         colors=colors, linestyles=linestyles, show=False,
                         split_legend=True, truncate_yaxis='auto')

    # plot temporal cluster extent
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='orange', alpha=0.3)

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    plt.show()

# %%
# Permutation statistic for time-frequencies
# ------------------------------------------
#
# Let's do the same thing with the time-frequency decomposition of the data
# (see :ref:`tut-sensors-time-freq` for a tutorial and
# :ref:`ex-tfr-comparison` for a comparison of time-frequency methods) to
# show how cluster permutations can be done on higher-dimensional data.

decim = 4
freqs = np.arange(7, 30, 3)  # define frequencies of interest
n_cycles = freqs / freqs[0]

epochs_power = list()
for condition in [epochs[k] for k in ('Aud/L', 'Vis/L')]:
    this_tfr = tfr_morlet(condition, freqs, n_cycles=n_cycles,
                          decim=decim, average=False, return_itc=False)
    this_tfr.apply_baseline(mode='ratio', baseline=(None, 0))
    epochs_power.append(this_tfr.data)

# transpose again to (epochs, frequencies, times, channels)
X = [np.transpose(x, (0, 2, 3, 1)) for x in epochs_power]

# %%
# Remember the note on the adjacency matrix from above: For 3D data, as here,
# we must use :func:`mne.stats.combine_adjacency` to extend the
# sensor-based adjacency to incorporate the time-frequency plane as well.
#
# Here, the integer inputs are converted into a lattice and
# combined with the sensor adjacency matrix so that data at similar
# times and with similar frequencies and at close sensor locations are
# clustered together.

# our data at each observation is of shape frequencies × times × channels
tfr_adjacency = combine_adjacency(
    len(freqs), len(this_tfr.times), adjacency)

# %%
# Now we can run the cluster permutation test, but first we have to set a
# threshold. This example decimates in time and uses few frequencies so we need
# to increase the threshold from the default value in order to have
# differentiated clusters (i.e., so that our algorithm doesn't just find one
# large cluster). For a more principled method of setting this parameter,
# threshold-free cluster enhancement may be used.
# See :ref:`disc-stats` for a discussion.

# This time we don't calculate a threshold based on the F distribution.
# We might as well select an arbitrary threshold for cluster forming
tfr_threshold = 15.0

# run cluster based permutation analysis
cluster_stats = spatio_temporal_cluster_test(
    X, n_permutations=1000, threshold=tfr_threshold, tail=1, n_jobs=None,
    buffer_size=None, adjacency=tfr_adjacency)

# %%
# Finally, we can plot our results. It is difficult to visualize clusters in
# time-frequency-sensor space; plotting time-frequency spectrograms and
# plotting topomaps display time-frequency and sensor space respectively
# but they are difficult to combine. We will plot topomaps with the clustered
# sensors colored in white adjacent to spectrograms in order to provide a
# visualization of the results. This is a dimensionally limited view, however.
# Each sensor has its own significant time-frequencies, but, in order to
# display a single spectrogram, all the time-frequencies that are significant
# for any sensor in the cluster are plotted as significant. This is a
# difficulty inherent to visualizing high-dimensional data and should be taken
# into consideration when interpreting results.
F_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < p_accept)[0]

for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster information, get unique indices
    freq_inds, time_inds, space_inds = clusters[clu_idx]
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)
    freq_inds = np.unique(freq_inds)

    # get topography for F stat
    f_map = F_obs[freq_inds].mean(axis=0)
    f_map = f_map[time_inds].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = epochs.times[time_inds]

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # plot average test statistic and mark significant sensors
    f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epochs.info, tmin=0)
    f_evoked.plot_topomap(times=0, mask=mask, axes=ax_topo, cmap='Reds',
                          vlim=(np.min, np.max), show=False, colorbar=False,
                          mask_params=dict(markersize=10))
    image = ax_topo.images[0]

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        'Averaged F-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

    # remove the title that would otherwise say "0.000 s"
    ax_topo.set_title("")

    # add new axis for spectrogram
    ax_spec = divider.append_axes('right', size='300%', pad=1.2)
    title = 'Cluster #{0}, {1} spectrogram'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += " (max over channels)"
    F_obs_plot = F_obs[..., ch_inds].max(axis=-1)
    F_obs_plot_sig = np.zeros(F_obs_plot.shape) * np.nan
    F_obs_plot_sig[tuple(np.meshgrid(freq_inds, time_inds))] = \
        F_obs_plot[tuple(np.meshgrid(freq_inds, time_inds))]

    for f_image, cmap in zip([F_obs_plot, F_obs_plot_sig], ['gray', 'autumn']):
        c = ax_spec.imshow(f_image, cmap=cmap, aspect='auto', origin='lower',
                           extent=[epochs.times[0], epochs.times[-1],
                                   freqs[0], freqs[-1]])
    ax_spec.set_xlabel('Time (ms)')
    ax_spec.set_ylabel('Frequency (Hz)')
    ax_spec.set_title(title)

    # add another colorbar
    ax_colorbar2 = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(c, cax=ax_colorbar2)
    ax_colorbar2.set_ylabel('F-stat')

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    plt.show()


# %%
# Exercises
# ----------
#
# - What is the smallest p-value you can obtain, given the finite number of
#   permutations? You can find the answers in the references
#   :footcite:`MarisOostenveld2007,Sassenhagen2019`.
#
# References
# ----------
# .. footbibliography::
#
# .. include:: ../../links.inc
