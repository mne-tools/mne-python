"""
=================================================================
Permutation t-test on source data with spatio-temporal clustering
=================================================================

This example tests if the evoked response is significantly different between
two conditions across subjects. Here just for demonstration purposes
we simulate data from multiple subjects using one subject's data.
The multiple comparisons problem is addressed with a cluster-level
permutation test across space and time.
"""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
# License: BSD (3-clause)

import os.path as op

import numpy as np
from numpy.random import randn
from scipy import stats as stats

import mne
from mne.epochs import equalize_epoch_counts
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.datasets import sample

print(__doc__)

###############################################################################
# Set parameters
# --------------
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
subjects_dir = data_path + '/subjects'
src_fname = subjects_dir + '/fsaverage/bem/fsaverage-ico-5-src.fif'

tmin = -0.2
tmax = 0.3  # Use a lower tmax to reduce multiple comparisons

#   Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

###############################################################################
# Read epochs for all channels, removing a bad one
# ------------------------------------------------
raw.info['bads'] += ['MEG 2443']
picks = mne.pick_types(raw.info, meg=True, eog=True, exclude='bads')
event_id = 1  # L auditory
reject = dict(grad=1000e-13, mag=4000e-15, eog=150e-6)
epochs1 = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                     baseline=(None, 0), reject=reject, preload=True)

event_id = 3  # L visual
epochs2 = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                     baseline=(None, 0), reject=reject, preload=True)

#    Equalize trial counts to eliminate bias (which would otherwise be
#    introduced by the abs() performed below)
equalize_epoch_counts([epochs1, epochs2])

###############################################################################
# Transform to source space
# -------------------------

fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE, sLORETA, or eLORETA)
inverse_operator = read_inverse_operator(fname_inv)
sample_vertices = [s['vertno'] for s in inverse_operator['src']]

#    Let's average and compute inverse, resampling to speed things up
evoked1 = epochs1.average()
evoked1.resample(50, npad='auto')
condition1 = apply_inverse(evoked1, inverse_operator, lambda2, method)
evoked2 = epochs2.average()
evoked2.resample(50, npad='auto')
condition2 = apply_inverse(evoked2, inverse_operator, lambda2, method)

#    Let's only deal with t > 0, cropping to reduce multiple comparisons
condition1.crop(0, None)
condition2.crop(0, None)
tmin = condition1.tmin
tstep = condition1.tstep * 1000  # convert to milliseconds

###############################################################################
# Transform to common cortical space
# ----------------------------------
#
# Normally you would read in estimates across several subjects and morph
# them to the same cortical space (e.g. fsaverage). For example purposes,
# we will simulate this by just having each "subject" have the same
# response (just noisy in source space) here.
#
# .. note::
#     Note that for 7 subjects with a two-sided statistical test, the minimum
#     significance under a permutation test is only p = 1/(2 ** 6) = 0.015,
#     which is large.
n_vertices_sample, n_times = condition1.data.shape
n_subjects = 7
print('Simulating data for %d subjects.' % n_subjects)

#    Let's make sure our results replicate, so set the seed.
np.random.seed(0)
X = randn(n_vertices_sample, n_times, n_subjects, 2) * 10
X[:, :, :, 0] += condition1.data[:, :, np.newaxis]
X[:, :, :, 1] += condition2.data[:, :, np.newaxis]

###############################################################################
# It's a good idea to spatially smooth the data, and for visualization
# purposes, let's morph these to fsaverage, which is a grade 5 source space
# with vertices 0:10242 for each hemisphere. Usually you'd have to morph
# each subject's data separately (and you might want to use morph_data
# instead), but here since all estimates are on 'sample' we can use one
# morph matrix for all the heavy lifting.

# Read the source space we are morphing to
src = mne.read_source_spaces(src_fname)
fsave_vertices = [s['vertno'] for s in src]
morph_mat = mne.compute_source_morph(
    src=inverse_operator['src'], subject_to='fsaverage',
    spacing=fsave_vertices, subjects_dir=subjects_dir).morph_mat

n_vertices_fsave = morph_mat.shape[0]

#    We have to change the shape for the dot() to work properly
X = X.reshape(n_vertices_sample, n_times * n_subjects * 2)
print('Morphing data.')
X = morph_mat.dot(X)  # morph_mat is a sparse matrix
X = X.reshape(n_vertices_fsave, n_times, n_subjects, 2)

###############################################################################
# Finally, we want to compare the overall activity levels in each condition,
# the diff is taken along the last axis (condition). The negative sign makes
# it so condition1 > condition2 shows up as "red blobs" (instead of blue).
X = np.abs(X)  # only magnitude
X = X[:, :, :, 0] - X[:, :, :, 1]  # make paired contrast


###############################################################################
# Compute statistic
# -----------------
#
# To use an algorithm optimized for spatio-temporal clustering, we
# just pass the spatial adjacency matrix (instead of spatio-temporal)
print('Computing adjacency.')
adjacency = mne.spatial_src_adjacency(src)

#    Note that X needs to be a multi-dimensional array of shape
#    samples (subjects) x time x space, so we permute dimensions
X = np.transpose(X, [2, 1, 0])

#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.001
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu = \
    spatio_temporal_cluster_1samp_test(X, adjacency=adjacency, n_jobs=1,
                                       threshold=t_threshold, buffer_size=None,
                                       verbose=True)
#    Now select the clusters that are sig. at p < 0.05 (note that this value
#    is multiple-comparisons corrected).
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

###############################################################################
# Visualize the clusters
# ----------------------
print('Visualizing clusters.')

#    Now let's build a convenient representation of each cluster, where each
#    cluster becomes a "time point" in the SourceEstimate
stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep,
                                             vertices=fsave_vertices,
                                             subject='fsaverage')

#    Let's actually plot the first "time point" in the SourceEstimate, which
#    shows all the clusters, weighted by duration
subjects_dir = op.join(data_path, 'subjects')
# blue blobs are for condition A < condition B, red for A > B
brain = stc_all_cluster_vis.plot(
    hemi='both', views='lateral', subjects_dir=subjects_dir,
    time_label='temporal extent (ms)', size=(800, 800),
    smoothing_steps=5, clim=dict(kind='value', pos_lims=[0, 1, 40]))
# brain.save_image('clusters.png')
