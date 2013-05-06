"""
======================================================================
Repeated measures anova on source data with spatio-temporal clustering
======================================================================

Tests if the differences in evoked responses bewtween stimulation
conditions depend on the stimulus location for a group of subjects 
(simulated here using one subject's data). For this purpuse we will
compute an interaction effect using a repeated measures ANOVA.
The multiple comparisons problem is addressed with a cluster-level
permutation test across space and time. 
"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Denis Engemannn <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

print __doc__

import os.path as op
import numpy as np
from numpy.random import randn
from scipy import stats as stats

import mne
from mne import fiff, spatial_tris_connectivity, compute_morph_matrix,\
    grade_to_tris, SourceEstimate
from mne.stats import spatio_temporal_cluster_test
from mne.stats.parametric import r_anova_twoway
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.datasets import sample
from mne.viz import mne_analyze_colormap

###############################################################################
# Set parameters
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
subjects_dir = data_path + '/subjects'

tmin = -0.2
tmax = 0.3  # Use a lower tmax to reduce multiple comparisons

#   Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.read_events(event_fname)

###############################################################################
# Read epochs for all channels, removing a bad one
raw.info['bads'] += ['MEG 2443']
picks = fiff.pick_types(raw.info, meg=True, eog=True, exclude='bads')
# we'll load all four conditions that make up the 'two ways' of our ANOVA
event_id = dict(l_aud=1, r_aud=2, l_vis=3, r_vis=4)
reject = dict(grad=1000e-13, mag=4000e-15, eog=150e-6)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                     baseline=(None, 0), reject=reject, preload=True)

#    Equalize trial counts to eliminate bias (which would otherwise be
#    introduced by the abs() performed below)
epochs.equalize_event_counts(event_id, copy=False)

###############################################################################
# Transform to source space

fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
inverse_operator = read_inverse_operator(fname_inv)
sample_vertices = [s['vertno'] for s in inverse_operator['src']]

#    Let's average and compute inverse, resampling to speed things up
conditions = []
for cond in event_id:
    evoked = epochs[cond].average()
    evoked.resample(50)
    condition = apply_inverse(evoked, inverse_operator, lambda2, method)
    #    Let's only deal with t > 0, cropping to reduce multiple comparisons
    condition.crop(0, None)
    conditions.append(condition)

tmin = conditions[0].tmin
tstep = conditions[0].tstep

###############################################################################
# Transform to common cortical space

#    Normally you would read in estimates across several subjects and morph
#    them to the same cortical space (e.g. fsaverage). For example purposes,
#    we will simulate this by just having each "subject" have the same
#    response (just noisy in source space) here.

n_vertices_sample, n_times = conditions[0].data.shape
n_subjects = 7
print 'Simulating data for %d subjects.' % n_subjects

#    Let's make sure our results replicate, so set the seed.
np.random.seed(0)
X = randn(n_vertices_sample, n_times, n_subjects, 4) * 10
for ii, condition in enumerate(conditions):
    X[:, :, :, ii] += condition.data[:, :, np.newaxis]

#    It's a good idea to spatially smooth the data, and for visualization
#    purposes, let's morph these to fsaverage, which is a grade 5 source space
#    with vertices 0:10242 for each hemisphere. Usually you'd have to morph
#    each subject's data separately (and you might want to use morph_data
#    instead), but here since all estimates are on 'sample' we can use one
#    morph matix for all the heavy lifting.
fsave_vertices = [np.arange(10242), np.arange(10242)]
morph_mat = compute_morph_matrix('sample', 'fsaverage', sample_vertices,
                                 fsave_vertices, 20, subjects_dir)
n_vertices_fsave = morph_mat.shape[0]

#    We have to change the shape for the dot() to work properly
X = X.reshape(n_vertices_sample, n_times * n_subjects * 4)
print 'Morphing data.'
X = morph_mat.dot(X)  # morph_mat is a sparse matrix
X = X.reshape(n_vertices_fsave, n_times, n_subjects, 4)

#    We need to prepare the gorup matrix fot the ANOVA statistic
X = np.abs(X)  # only magnitude
#    Note that X needs to be a multi-dimensional array of shape
#    samples (subjects) x time x space, so we permute dimensions
X = np.transpose(X, [2, 1, 0, 3])
# finally we split the array into a list a list of conditions
X = [np.squeeze(x) for x in np.split(X, 4, axis=-1)]  

###############################################################################
# Compute statistic

#    To use an algorithm optimized for spatio-temporal clustering, we
#    just pass the spatial connectivity matrix (instead of spatio-temporal)
print 'Computing connectivity.'
connectivity = spatial_tris_connectivity(grade_to_tris(5))


#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
f_threshold = 80  # high threshold to save us some time
n_permutations = 256 # less permutations ... (don't imitate this at home)

def stat_fun(*args):  # variable number of arguments required for a stat_fun
    # reshape data as required by r_anova_twoway
    data = np.swapaxes(np.asarray(args), 1, 0).reshape(n_replications, \
        n_conditions, 8 * 211)
    # We will just pick the interaction by passing 'A:B'.
    # (this notations is borrowed from the R formula language)
    return r_anova_twoway(data, factor_levels=[2, 2], effects='A:B',
                return_pvals=False)[0]

print 'Clustering.'
T_obs, clusters, cluster_p_values, H0 = \
    spatio_temporal_cluster_test(X, connectivity=connectivity, n_jobs=2,
                                 threshold=f_threshold,
                                 n_permutations=n_permutations)
#    Now select the clusters that are sig. at p < 0.05 (note that this value
#    is multiple-comparisons corrected).
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

###############################################################################
# Visualize the clusters

print 'Visualizing clusters.'

#    Now let's build a convenient representation of each cluster, where each
#    cluster becomes a "time point" in the SourceEstimate
data = np.zeros((n_vertices_fsave, n_times))
data_summary = np.zeros((n_vertices_fsave, len(good_cluster_inds) + 1))
for ii, cluster_ind in enumerate(good_cluster_inds):
    data.fill(0)
    v_inds = clusters[cluster_ind][1]
    t_inds = clusters[cluster_ind][0]
    data[v_inds, t_inds] = T_obs[t_inds, v_inds]
    # Store a nice visualization of the cluster by summing across time (in ms)
    data = np.sign(data) * np.logical_not(data == 0) * tstep
    data_summary[:, ii + 1] = 1e3 * np.sum(data, axis=1)

#    Make the first "time point" a sum across all clusters for easy
#    visualization
data_summary[:, 0] = np.sum(data_summary, axis=1)
stc_all_cluster_vis = SourceEstimate(data_summary, fsave_vertices, tmin=0,
                                     tstep=1e-3, subject='fsaverage')

#    Let's actually plot the first "time point" in the SourceEstimate, which
#    shows all the clusters, weighted by duration

subjects_dir = op.join(data_path, 'subjects')
# The brighter the color, the stronger the interaction between 
# stimulus modality and stimulus location
brains = stc_all_cluster_vis.plot('fsaverage', 'inflated', 'both',
                                  subjects_dir=subjects_dir,
                                  time_label='Duration significant (ms)')
for idx, brain in enumerate(brains):
    brain.set_data_time_index(0)
    # The colormap requires brain data to be scaled -fmax -> fmax
    brain.scale_data_colormap(fmin=5, fmid=10, fmax=30, transparent=True)
    brain.show_view('lateral')
    brain.save_image('clusters-%s.png' % ('lh' if idx == 0 else 'rh'))
