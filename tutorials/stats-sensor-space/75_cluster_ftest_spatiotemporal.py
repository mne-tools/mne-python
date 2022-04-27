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

See the `FieldTrip website`_ for a caveat regarding
the possible interpretation of "significant" clusters.
"""
# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

# %%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mne
from mne.stats import spatio_temporal_cluster_test, combine_adjacency
from mne.datasets import sample
from mne.channels import find_ch_adjacency
from mne.viz import plot_compare_evokeds
from mne.time_frequency import tfr_morlet

print(__doc__)

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
raw.filter(1, 30, fir_design='firwin')
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

X = [epochs[k].get_data() for k in event_id]  # as 3D matrix
X = [np.transpose(x, (0, 2, 1)) for x in X]  # transpose for clustering


# %%
# Find the FieldTrip neighbor definition to setup sensor adjacency
# ----------------------------------------------------------------
adjacency, ch_names = find_ch_adjacency(epochs.info, ch_type='mag')

print(type(adjacency))  # it's a sparse matrix!

fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(adjacency.toarray(), cmap='gray', origin='lower',
          interpolation='nearest')
ax.set_xlabel('{} Magnetometers'.format(len(ch_names)))
ax.set_ylabel('{} Magnetometers'.format(len(ch_names)))
ax.set_title('Between-sensor adjacency')
fig.tight_layout()

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
# of observing a cluster of that size. For more background read:
# Maris/Oostenveld (2007), "Nonparametric statistical testing of EEG- and
# MEG-data" Journal of Neuroscience Methods, Vol. 164, No. 1., pp. 177-190.
# doi:10.1016/j.jneumeth.2007.03.024


# set cluster threshold
threshold = 50.0  # very high, but the test is quite sensitive on this data
# set family-wise p-value
p_accept = 0.01

cluster_stats = spatio_temporal_cluster_test(X, n_permutations=1000,
                                             threshold=threshold, tail=1,
                                             n_jobs=1, buffer_size=None,
                                             adjacency=adjacency)

F_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < p_accept)[0]

# %%
# Note. The same functions work with source estimate. The only differences
# are the origin of the data, the size, and the adjacency definition.
# It can be used for single trials or for groups of subjects.
#
# Visualize clusters
# ------------------

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
                          vmin=np.min, vmax=np.max, show=False,
                          colorbar=False, mask_params=dict(markersize=10))
    image = ax_topo.images[0]

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

# transpose again to (epochs, frequencies, times, vertices)
X = [np.transpose(x, (0, 2, 3, 1)) for x in epochs_power]

# %%
# Let's extend our adjacency definition to include the time-frequency
# dimensions. Here, the integer inputs are converted into a lattice and
# combined with the sensor adjacency matrix so that data at similar
# times and with similar frequencies and at close sensor locations are
# clustered together.
tfr_adjacency = combine_adjacency(
    len(freqs), len(this_tfr.times), adjacency)

# %%
# Now we can run the cluster permutation test, but first we have to set a
# threshold. This example decimates in time and uses few frequencies so we need
# to increase the threshold from the default value in order to have
# differentiated clusters (i.e. so that our algorithm doesn't just find one
# large cluster). For a more principled method of setting this parameter,
# threshold-free cluster enhancement may be used or the p-value may be set with
#
# .. code-block:: python
#
#     from scipy import stats
#     n_comparisons = len(X)  # L auditory vs L visual stimulus
#     n_conditions = X[0].shape[0]  # 55 epochs per comparison
#     threshold = stats.distributions.f.ppf(
#         1 - p_accept, n_comparisons - 1, n_conditions - 1)
#
# See :ref:`disc-stats` for a discussion.

tfr_threshold = 15.0

# run statistic
cluster_stats = spatio_temporal_cluster_test(
    X, n_permutations=1000, threshold=tfr_threshold, tail=1, n_jobs=1,
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
                          vmin=np.min, vmax=np.max, show=False,
                          colorbar=False, mask_params=dict(markersize=10))
    image = ax_topo.images[0]

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        'Averaged F-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

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


# %%
# Exercises
# ----------
#
# - What is the smallest p-value you can obtain, given the finite number of
#   permutations?
# - use an F distribution to compute the threshold by traditional significance
#   levels. Hint: take a look at :obj:`scipy.stats.f`
#
# .. _fieldtrip website:
#       http://www.fieldtriptoolbox.org/faq/
#       how_not_to_interpret_results_from_a_cluster-based_permutation_test
