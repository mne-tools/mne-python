"""
====================================================================
Mass-Univariate Twoway Repeated Measures ANOVA on Single Trial Power
====================================================================

This script shows how to conduct a mass-univariate repeated measures
ANOVA. As the model to be fitted assumes two fully crossed factors,
we will study the interplay between perceptual modality
(auditory VS visual) and the location of stimulus presentation
(left VS right). Here we use single trials as replications
(subjects) while iterating over time slices plus frequency bands
for to fit our mass-univariate model. We will then visualize each
effect by creating a corresponding mass-univariate effect image.
We conclude with accounting for multiple comparisons by using
performing a permutation clustering test using the ANOVA as
clustering function.
"""
# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np

import mne
from mne import fiff
from mne.time_frequency import single_trial_power
from mne.stats.parametric import r_anova_twoway
from mne.datasets import sample

###############################################################################
# Set parameters
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
event_id = 1
tmin = -0.2
tmax = 0.5

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.read_events(event_fname)

include = []
raw.info['bads'] += ['MEG 2443']  # bads

# picks MEG gradiometers
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                        stim=False, include=include, exclude='bads')

ch_name = raw.info['ch_names'][picks[0]]

# Load conditions
reject = dict(grad=4000e-13, eog=150e-6)
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                                picks=picks, baseline=(None, 0),
                                reject=reject)

# make sure all conditions have the same counts, this is crucial for the ANOVA
epochs.equalize_event_counts(event_id, copy=False)
# Time vector
times = 1e3 * epochs.times  # change unit to ms

# Factor to downs-sample the temporal dimension of the PSD computed by
# single_trial_power.
decim = 2
frequencies = np.arange(7, 30, 3)  # define frequencies of interest
Fs = raw.info['sfreq']  # sampling in Hz
n_cycles = 1.5
baseline_mask = times[::decim] < 0

# now create TFR representations for all conditions
epochs_power = []
for condition in [epochs[k].get_data()[:, 97:98, :] for k in event_id]:
    this_power = single_trial_power(condition, Fs=Fs, frequencies=frequencies,
        n_cycles=n_cycles, use_fft=False, decim=decim)
    this_power = this_power[:, 0, :, :]  # we only have one channel.
    # Compute ratio with baseline power (be sure to correct time vector with
    # decimation factor)
    epochs_baseline = np.mean(this_power[:, :, baseline_mask], axis=2)
    this_power /= epochs_baseline[..., np.newaxis]
    epochs_power.append(this_power)

###############################################################################
# Setup repeated measures ANOVA

n_conditions = len(epochs.event_id)
n_replications = epochs.events.shape[0] / n_conditions
factor_levels = [2, 2]  # number of levels in each factor
# assemble data matrix and swap axes so the trial replications
# are the first dimension and the conditions are the second dimension
effects = 'A*B'  # this is the default signature for computing all effects
data = np.swapaxes(np.asarray(epochs_power), 1, 0)
# reshape last two dimensions in one mass-univariate observation-vector
data = data.reshape(n_replications, n_conditions, 8 * 211)

# so we have replications * conditions * observations:
print data.shape

# while the iteration scheme used above for assembling the data matrix
# makes sure the first two dimensions are organized as expected (with A =
# modality and B = location):
#
#           A1B1 A1B2 A2B1 B2B2
# trial 1   1.34 2.53 0.97 1.74
# trial ... .... .... .... ....
# trial 56  2.45 7.90 3.09 4.76
#
# So we're ready to run our repeated measures ANOVA.

fvals, pvals = r_anova_twoway(data, factor_levels, effects=effects)

effect_labels = ['modality', 'location', 'modality by location']
import pylab as pl

# let's visualize our effects by computing f-images
for effect, sig, effect_label in zip(fvals, pvals, effect_labels):
    pl.figure()
    # show naive F-values in gray
    pl.imshow(effect.reshape(8, 211), cmap=pl.cm.gray, extent=[times[0],
        times[-1], frequencies[0], frequencies[-1]], aspect='auto',
        origin='lower')
    # create mask for significant Time-frequency locations
    effect = np.ma.masked_array(effect, [sig > .05])
    pl.imshow(effect.reshape(8, 211), cmap=pl.cm.jet, extent=[times[0],
        times[-1], frequencies[0], frequencies[-1]], aspect='auto',
        origin='lower')
    pl.colorbar()
    pl.xlabel('time (ms)')
    pl.ylabel('Frequency (Hz)')
    pl.title(r"Time-locked response for '%s' (%s)" % (effect_label, ch_name))
    pl.show()

# Note. As we treat trials as subjects, the test only accounts for
# time locked responses despite the 'induced' approach.
# For analysis for induced power at the group level averaged TRFs
# are required.


###############################################################################
# Account for multiple comparisons using a permutation clustering test

# First we need to slightly modify the ANOVA function to be suitable for
# the clustering procedure. Also want to set some defaults.

def stat_fun(*args):  # variable number of arguments required for a stat_fun
    # reshape data as required by r_anova_twoway
    data = np.swapaxes(np.asarray(args), 1, 0).reshape(n_replications, \
        n_conditions, 8 * 211)
    # We will just pick the interaction by passing 'A:B'.
    # (this notations is borrowed from the R formula language)
    return r_anova_twoway(data, factor_levels=[2, 2], effects='A:B',
                return_pvals=False)[0]

threshold = 20.0  # f-values > 20. as clustering threshold to save some time
tail = 1  # f-test, so tail > 0
n_permutations = 256  # Save some time (the test won't be too sensitive ...)
T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
    epochs_power, stat_fun=stat_fun, threshold=threshold, tail=tail, n_jobs=2,
    n_permutations=n_permutations)

# Create new stats image with only significant clusters
good_clusers, _ = np.where(cluster_p_values < .05)
T_obs_plot = np.ma.masked_array(T_obs, np.invert(clusters[good_clusers]))


pl.imshow(T_obs, cmap=pl.cm.gray, extent=[times[0], times[-1],
                                          frequencies[0], frequencies[-1]],
                                  aspect='auto', origin='lower')
pl.imshow(T_obs_plot, cmap=pl.cm.jet, extent=[times[0], times[-1],
                                              frequencies[0], frequencies[-1]],
                                  aspect='auto', origin='lower')

# We see that the cluster level correction helps getting rid of random spots
# we saw in the naive f-images.
pl.xlabel('time (ms)')
pl.ylabel('Frequency (Hz)')
pl.title('Time-locked response for \'modality by location\' (%s)\n'
          ' cluster-level corrected (p <= 0.5)' % ch_name)
pl.show()
