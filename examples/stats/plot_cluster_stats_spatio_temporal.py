"""
=================================================================
Permutation t-test on source data with spatio-temporal clustering
=================================================================

Tests if the evoked response is significantly different between
conditions across subjects (simulated here using one subject's data).
The multiple comparisons problem is adressed with a cluster-level
permutation test across space and time.

"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
# License: BSD (3-clause)
#
# The techniques follow the methods outlined and used in these papers:
#  - Lee AK, Larson E, Maddox RK. "Mapping cortical dynamics..."
#    J Vis Exp. 2012 Oct 24;(68). doi:pii: 4262. 10.3791/4262.
#  - Larson E, Lee AK. "The cortical dynamics underlying effective
#    switching of auditory..." Neuroimage. 2012 Sep 11;64C:365-370.

print __doc__

import mne
from mne import fiff, spatial_tris_connectivity, compute_morph_matrix,\
    grade_to_tris, equalize_epoch_counts, SourceEstimate, read_surface
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.datasets import sample
from mne.viz import mne_analyze_colormap
import os.path as op
import numpy as np
from numpy.random import randn
from scipy import stats as spstats

###############################################################################
# Set parameters
data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
subjects_dir = op.join(data_path, 'subjects')
tmin = -0.2
tmax = 0.3  # Use a lower tmax to reduce multiple comparisons

#   Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.read_events(event_fname)

###############################################################################
# Read epochs for all channels, removing a bad one
exclude = ['MEG 2443']
picks = fiff.pick_types(raw.info, meg=True, eog=True, exclude=exclude)
event_id = 1  # L auditory
reject = dict(grad=1000e-13, mag=4000e-15, eog=150e-6)
epochs1 = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                     baseline=(None, 0), reject=reject, preload=True)

event_id = 3  # L visual
epochs2 = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                     baseline=(None, 0), reject=reject, preload=True)

#    Equalize trial counts to eliminate bias (which would otherwise be
#    introduced by the abs() performed below)
equalize_epoch_counts(epochs1, epochs2, method='mintime')

###############################################################################
# Transform to source space

fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
inverse_operator = read_inverse_operator(fname_inv)
sample_vertices = [s['vertno'] for s in inverse_operator['src']]

#    Let's average and compute inverse, resampling to speed things up
evoked1 = epochs1.average()
evoked1.resample(50)
condition1 = apply_inverse(evoked1, inverse_operator, lambda2, method)
evoked2 = epochs2.average()
evoked2.resample(50)
condition2 = apply_inverse(evoked2, inverse_operator, lambda2, method)

#    Let's only deal with t > 0, cropping to reduce multiple comparisons
condition1.crop(0, None)
condition2.crop(0, None)
tmin = condition1.tmin
tstep = condition1.tstep

###############################################################################
# Transform to common cortical space

#    Normally you would read in estimates across several subjects and morph
#    them to the same cortical space (e.g. fsaverage). For example purposes,
#    we will simulate this by just having each "subject" have the same
#    response (just noisy in source space) here. Note that for 7 subjects
#    with a two-sided statistical test, the minimum significance under a
#    permutation test is only p = 1/(2 ** 6) = 0.015, which is large.
n_vertices_sample, n_times = condition1.data.shape
n_subjects = 7
print 'Simulating data for %d subjects.' % n_subjects

#    Let's make sure our results replicate, so set the seed.
np.random.seed(0)
X = randn(n_vertices_sample, n_times, n_subjects, 2) * 10
X[:, :, :, 0] += condition1.data[:, :, np.newaxis]
X[:, :, :, 1] += condition2.data[:, :, np.newaxis]

#    It's a good idea to spatially smooth the data, and for visualization
#    purposes, let's morph these to fsaverage, which is a grade 5 source space
#    with vertices 0:10242 for each hemisphere. Usually you'd have to morph
#    each subject's data separately (and you might want to use morph_data
#    instead), but here since all estimates are on 'sample' we can use one
#    morph matix for all the heavy lifting.
fsave_vertices = [np.arange(10242), np.arange(10242)]
morph_mat = compute_morph_matrix('sample', 'fsaverage', sample_vertices,
                                 fsave_vertices, 20, subjects_dir,
                                 array=True)
n_vertices_fsave = morph_mat.shape[0]

#    We have to change the shape for the dot() to work properly
X.shape = (n_vertices_sample, n_times * n_subjects * 2)
print 'Morphing data.'
X = morph_mat.dot(X)
X.shape = (n_vertices_fsave, n_times, n_subjects, 2)

#    Finally, we want to compare the overall activity levels in each condition,
#    the diff is taken along the last axis (condition). The negative sign makes
#    it so condition1 > condition2 shows up as "red blobs" (instead of blue).
X = np.squeeze(-np.diff(np.abs(X)))


###############################################################################
# Compute statistic

#    To use an algorithm optimized for spatio-temporal clustering, we
#    just pass the spatial connectivity matrix (instead of spatio-temporal)
print 'Computing connectivity.'
connectivity = spatial_tris_connectivity(grade_to_tris(5))

#    Note that X needs to be a multi-dimensional array of shape
#    samples (subjects) x time x space, so we permute dimensions
X = np.transpose(X, [2, 1, 0])

#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.001
t_threshold = -spstats.distributions.t.ppf(p_threshold / 2, n_subjects)
print 'Clustering.'
T_obs, clusters, cluster_p_values, H0 = \
    spatio_temporal_cluster_1samp_test(X, connectivity=connectivity, n_jobs=2,
                                       threshold=t_threshold)
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
    # Store a nice visualization of the cluster by summing across time
    data = np.sign(data) * np.logical_not(data == 0) * tstep
    data_summary[:, ii + 1] = np.sum(data, axis=1)

#    Make the first "time point" a sum across all clusters for easy
#    visualization
data_summary[:, 0] = np.sum(data_summary, axis=1)
stc_all_cluster_vis = SourceEstimate(data_summary, fsave_vertices, 0, 1e-3)

#    Let's actually plot the first "time point" in the SourceEstimate, which
#    shows all the clusters, weighted by duration, for the right hemisphere

#from enthought.mayavi import mlab
#mlab.figure(size=(600, 600), bgcolor=(0, 0, 0))
#mlab.triangular_mesh(lh_points[:, 0], lh_points[:, 1], lh_points[:, 2],
#                     lh_faces)
#mlab.triangular_mesh(rh_points[:, 0], rh_points[:, 1], rh_points[:, 2],
#                     rh_faces)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
max_duration = 60.0  # in ms
my_cmap = mne_analyze_colormap([1000 * tstep - 1, 1000 * tstep, max_duration])
surf = read_surface(op.join(data_path, 'subjects', 'fsaverage',
                            'surf', 'rh.white'))
coords = surf[0][fsave_vertices[1]]
vals = stc_all_cluster_vis.rh_data[:, 0] * 1000

fig = plt.figure(facecolor='k')
ax = plt.axes([0, 0, 1, 1], projection='3d', axis_bgcolor='k')
sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=vals,
                vmin=-max_duration, vmax=max_duration,
                cmap=my_cmap, edgecolors='none', s=5)
ax.view_init(0, 0)
plt.axis('off')
cax = plt.axes([0.85, 0.15, 0.025, 0.15], axisbg='k')
cb = plt.colorbar(sc, cax, ticks=[-max_duration, 0, max_duration])
cb.set_label('Duration significant (ms)', color='w')
plt.setp(plt.getp(cb.ax, 'yticklabels'), color='w')
plt.draw()
plt.show()
