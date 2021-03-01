"""
.. _tut-timefreq-twoway-anova:

====================================================================
Mass-univariate twoway repeated measures ANOVA on single trial power
====================================================================

This script shows how to conduct a mass-univariate repeated measures
ANOVA. As the model to be fitted assumes two fully crossed factors,
we will study the interplay between perceptual modality
(auditory VS visual) and the location of stimulus presentation
(left VS right). Here we use single trials as replications
(subjects) while iterating over time slices plus frequency bands
for to fit our mass-univariate model. For the sake of simplicity we
will confine this analysis to one single channel of which we know
that it exposes a strong induced response. We will then visualize
each effect by creating a corresponding mass-univariate effect
image. We conclude with accounting for multiple comparisons by
performing a permutation clustering test using the ANOVA as
clustering function. The results final will be compared to multiple
comparisons using False Discovery Rate correction.
"""
# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.time_frequency import tfr_morlet
from mne.stats import f_threshold_mway_rm, f_mway_rm, fdr_correction
from mne.datasets import sample

print(__doc__)

###############################################################################
# Set parameters
# --------------
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
tmin, tmax = -0.2, 0.5

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

include = []
raw.info['bads'] += ['MEG 2443']  # bads

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                       stim=False, include=include, exclude='bads')

ch_name = 'MEG 1332'

# Load conditions
reject = dict(grad=4000e-13, eog=150e-6)
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=reject)
epochs.pick_channels([ch_name])  # restrict example to one channel

###############################################################################
# We have to make sure all conditions have the same counts, as the ANOVA
# expects a fully balanced data matrix and does not forgive imbalances that
# generously (risk of type-I error).
epochs.equalize_event_counts(event_id)

# Factor to down-sample the temporal dimension of the TFR computed by
# tfr_morlet.
decim = 2
freqs = np.arange(7, 30, 3)  # define frequencies of interest
n_cycles = freqs / freqs[0]
zero_mean = False  # don't correct morlet wavelet to be of mean zero
# To have a true wavelet zero_mean should be True but here for illustration
# purposes it helps to spot the evoked response.

###############################################################################
# Create TFR representations for all conditions
# ---------------------------------------------
epochs_power = list()
for condition in [epochs[k] for k in event_id]:
    this_tfr = tfr_morlet(condition, freqs, n_cycles=n_cycles,
                          decim=decim, average=False, zero_mean=zero_mean,
                          return_itc=False)
    this_tfr.apply_baseline(mode='ratio', baseline=(None, 0))
    this_power = this_tfr.data[:, 0, :, :]  # we only have one channel.
    epochs_power.append(this_power)

###############################################################################
# Setup repeated measures ANOVA
# -----------------------------
#
# We will tell the ANOVA how to interpret the data matrix in terms of factors.
# This is done via the factor levels argument which is a list of the number
# factor levels for each factor.

n_conditions = len(epochs.event_id)
n_replications = epochs.events.shape[0] // n_conditions

factor_levels = [2, 2]  # number of levels in each factor
effects = 'A*B'  # this is the default signature for computing all effects
# Other possible options are 'A' or 'B' for the corresponding main effects
# or 'A:B' for the interaction effect only (this notation is borrowed from the
# R formula language)
n_freqs = len(freqs)
times = 1e3 * epochs.times[::decim]
n_times = len(times)

###############################################################################
# Now we'll assemble the data matrix and swap axes so the trial replications
# are the first dimension and the conditions are the second dimension.
data = np.swapaxes(np.asarray(epochs_power), 1, 0)
# reshape last two dimensions in one mass-univariate observation-vector
data = data.reshape(n_replications, n_conditions, n_freqs * n_times)

# so we have replications * conditions * observations:
print(data.shape)

###############################################################################
# While the iteration scheme used above for assembling the data matrix
# makes sure the first two dimensions are organized as expected (with A =
# modality and B = location):
#
# .. table:: Sample data layout
#
#    ===== ==== ==== ==== ====
#    trial A1B1 A1B2 A2B1 B2B2
#    ===== ==== ==== ==== ====
#    1     1.34 2.53 0.97 1.74
#    ...   ...  ...  ...  ...
#    56    2.45 7.90 3.09 4.76
#    ===== ==== ==== ==== ====
#
# Now we're ready to run our repeated measures ANOVA.
#
# Note. As we treat trials as subjects, the test only accounts for
# time locked responses despite the 'induced' approach.
# For analysis for induced power at the group level averaged TRFs
# are required.

fvals, pvals = f_mway_rm(data, factor_levels, effects=effects)

effect_labels = ['modality', 'location', 'modality by location']

# let's visualize our effects by computing f-images
for effect, sig, effect_label in zip(fvals, pvals, effect_labels):
    plt.figure()
    # show naive F-values in gray
    plt.imshow(effect.reshape(8, 211), cmap=plt.cm.gray, extent=[times[0],
               times[-1], freqs[0], freqs[-1]], aspect='auto',
               origin='lower')
    # create mask for significant Time-frequency locations
    effect[sig >= 0.05] = np.nan
    plt.imshow(effect.reshape(8, 211), cmap='RdBu_r', extent=[times[0],
               times[-1], freqs[0], freqs[-1]], aspect='auto',
               origin='lower')
    plt.colorbar()
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.title(r"Time-locked response for '%s' (%s)" % (effect_label, ch_name))
    plt.show()

###############################################################################
# Account for multiple comparisons using FDR versus permutation clustering test
# -----------------------------------------------------------------------------
#
# First we need to slightly modify the ANOVA function to be suitable for
# the clustering procedure. Also want to set some defaults.
# Let's first override effects to confine the analysis to the interaction
effects = 'A:B'

###############################################################################
# A stat_fun must deal with a variable number of input arguments.
# Inside the clustering function each condition will be passed as flattened
# array, necessitated by the clustering procedure. The ANOVA however expects an
# input array of dimensions: subjects X conditions X observations (optional).
# The following function catches the list input and swaps the first and
# the second dimension and finally calls the ANOVA function.


def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]


# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.001  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_replications, factor_levels, effects,
                               pthresh)
tail = 1  # f-test, so tail > 0
n_permutations = 256  # Save some time (the test won't be too sensitive ...)
T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
    epochs_power, stat_fun=stat_fun, threshold=f_thresh, tail=tail, n_jobs=1,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')

###############################################################################
# Create new stats image with only significant clusters:

good_clusters = np.where(cluster_p_values < .05)[0]
T_obs_plot = T_obs.copy()
T_obs_plot[~clusters[np.squeeze(good_clusters)]] = np.nan

plt.figure()
for f_image, cmap in zip([T_obs, T_obs_plot], [plt.cm.gray, 'RdBu_r']):
    plt.imshow(f_image, cmap=cmap, extent=[times[0], times[-1],
               freqs[0], freqs[-1]], aspect='auto',
               origin='lower')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title("Time-locked response for 'modality by location' (%s)\n"
          " cluster-level corrected (p <= 0.05)" % ch_name)
plt.show()

###############################################################################
# Now using FDR:

mask, _ = fdr_correction(pvals[2])
T_obs_plot2 = T_obs.copy()
T_obs_plot2[~mask.reshape(T_obs_plot.shape)] = np.nan

plt.figure()
for f_image, cmap in zip([T_obs, T_obs_plot2], [plt.cm.gray, 'RdBu_r']):
    if np.isnan(f_image).all():
        continue  # nothing to show
    plt.imshow(f_image, cmap=cmap, extent=[times[0], times[-1],
               freqs[0], freqs[-1]], aspect='auto',
               origin='lower')

plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title("Time-locked response for 'modality by location' (%s)\n"
          " FDR corrected (p <= 0.05)" % ch_name)
plt.show()

###############################################################################
# Both cluster level and FDR correction help get rid of
# potential spots we saw in the naive f-images.
