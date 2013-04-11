"""
======================================================
Permutation t-test on toy data with spatial clustering
======================================================

Following the illustrative example of Ridgway et al. 2012,
this demonstrates some basic ideas behind both the "hat"
variance adjustment method, as well as threshold-free
cluster enhancement (TFCE) methods in mne-python.

For more information, see:
Ridgway et al. 2012, "The problem of low variance voxels in
statistical parametric mapping; a new hat avoids a 'haircut'",
NeuroImage. 2012 Feb 1;59(3):2131-41.

Smith and Nichols 2009, "Threshold-free cluster enhancement:
adressing problems of smoothing, threshold dependence, and
localisation in cluster inference", NeuroImage 44 (2009) 83-98.
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
# License: BSD (3-clause)

print __doc__

import numpy as np
from scipy import stats, sparse
from functools import partial

from mne.stats import spatio_temporal_cluster_1samp_test, bonferroni_correction
from mne.stats.cluster_level import ttest_1samp_no_p

###############################################################################
# Set parameters
width = 40
n_subjects = 12
signal_mean = 100
signal_sd = 100
noise_sd = 0.1
gaussian_sd = 5
sigma = 1e-3  # sigma for the "hat" method
threshold = -stats.distributions.t.ppf(0.05, n_subjects - 1)
threshold_tfce = dict(start=0, step=0.5)
n_permutations = 1024  # number of clustering permutations

###############################################################################
# Construct simulated data
#    Make the connectivity matrix just next-neighbor spatially
n_src = width * width
connectivity = np.sum(np.array([sparse.eye(n_src, n_src, kk)
                                for kk in [-1, 0, 1]]), axis=0).tocoo()

#    For each "subject", make a smoothed noisy signal with a centered peak
rng = np.random.RandomState(42)
X = noise_sd * rng.randn(n_subjects, width, width)
#    Add a signal at the dead center
X[:, width // 2, width // 2] = signal_mean + rng.randn(n_subjects) * signal_sd
#    Spatially smooth with a 2D Gaussian kernel
size = width // 2 - 1
gaussian = np.exp(-(np.arange(-size, size + 1) ** 2 / float(gaussian_sd ** 2)))
for si in range(X.shape[0]):
    for ri in range(X.shape[1]):
        X[si, ri, :] = np.convolve(X[si, ri, :], gaussian, 'same')
    for ci in range(X.shape[2]):
        X[si, :, ci] = np.convolve(X[si, :, ci], gaussian, 'same')

###############################################################################
# Do some statistics

#     Now let's do a basic t-test:
T_obs = ttest_1samp_no_p(X)

#     To do a Bonferroni correction on these data is simple, and not too bad:
p = stats.distributions.t.sf(T_obs, n_subjects - 1)
p_bon = -np.log10(bonferroni_correction(p)[1])

#     And one with variance correction:
T_obs_hat = ttest_1samp_no_p(X, sigma=sigma)

#    Note that X needs to be a multi-dimensional array of shape
#    samples (subjects) x time x space, so we permute dimensions
X = X.reshape((n_subjects, 1, n_src))

#    Now let's do some clustering using the standard method:
T_obs_cluster, clusters, cluster_p_values, H0 = \
    spatio_temporal_cluster_1samp_test(X, connectivity=connectivity, n_jobs=2,
                                       threshold=threshold, tail=1,
                                       n_permutations=n_permutations)

#    Let's put the cluster data in a readable format
cluster_ps = np.zeros(width * width)
for cl, p in zip(clusters, cluster_p_values):
    cluster_ps[cl[1]] = -np.log10(p)
cluster_ps = cluster_ps.reshape((width, width))
T_obs_cluster = T_obs_cluster.reshape((width, width))

#    Now the threshold-free cluster enhancement method (TFCE):
T_obs_cluster_tfce, clusters, cluster_p_values, H0 = \
    spatio_temporal_cluster_1samp_test(X, connectivity=connectivity, n_jobs=2,
                                       threshold=threshold_tfce, tail=1,
                                       n_permutations=n_permutations)
T_obs_cluster_tfce = T_obs_cluster_tfce.reshape((width, width))
cluster_ps_tfce = -np.log10(cluster_p_values.reshape((width, width)))

#    Now the TFCE with "hat" variance correction:
stat_fun = partial(ttest_1samp_no_p, sigma=sigma)
T_obs_cluster_tfce_hat, clusters, cluster_p_values, H0 = \
    spatio_temporal_cluster_1samp_test(X, connectivity=connectivity, n_jobs=2,
                                       threshold=threshold_tfce, tail=1,
                                       n_permutations=n_permutations,
                                       stat_fun=stat_fun)
T_obs_cluster_tfce_hat = T_obs_cluster_tfce_hat.reshape((width, width))
cluster_ps_tfce_hat = -np.log10(cluster_p_values.reshape((width, width)))

###############################################################################
# Visualize results

import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
pl.ion()
fig = pl.figure(facecolor='w')

x, y = np.mgrid[0:width, 0:width]
kwargs = dict(rstride=1, cstride=1, linewidth=0, cmap='Greens')
ax = fig.add_subplot(2, 4, 1, projection='3d')
ax.plot_surface(x, y, T_obs, **kwargs)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('T statistic')

ax = fig.add_subplot(2, 4, 2, projection='3d')
ax.plot_surface(x, y, T_obs_hat, **kwargs)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('T with "hat"')

ax = fig.add_subplot(2, 4, 3, projection='3d')
ax.plot_surface(x, y, T_obs_cluster_tfce, **kwargs)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('TFCE statistic')

ax = fig.add_subplot(2, 4, 4, projection='3d')
ax.plot_surface(x, y, T_obs_cluster_tfce_hat, **kwargs)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('TFCE w/"hat" stat')

p_lims = [1.3, -np.log10(1.0 / n_permutations)]
ax = fig.add_subplot(2, 4, 5)
pl.imshow(p_bon, cmap='Purples', vmin=p_lims[0], vmax=p_lims[1])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Bonferroni')
ax_1 = ax

ax = fig.add_subplot(2, 4, 6)
pl.imshow(cluster_ps, cmap='Purples', vmin=p_lims[0], vmax=p_lims[1])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Standard clustering')
ax_2 = ax

ax = fig.add_subplot(2, 4, 7)
pl.imshow(cluster_ps_tfce, cmap='Purples', vmin=p_lims[0], vmax=p_lims[1])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Clust. w/TFCE')
ax_3 = ax

ax = fig.add_subplot(2, 4, 8)
pl.imshow(cluster_ps_tfce_hat, cmap='Purples', vmin=p_lims[0], vmax=p_lims[1])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Clust. w/TFCE+"hat"')
ax_4 = ax

pl.tight_layout()
for ax in [ax_1, ax_2, ax_3, ax_4]:
    cbar = pl.colorbar(ax=ax, shrink=0.75, orientation='horizontal',
                       fraction=0.1, pad=0.025)
    cbar.set_label('-log10(p)')
    cbar.set_ticks(p_lims)
    cbar.set_ticklabels(['%0.2f' % p for p in p_lims])
