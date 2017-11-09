"""
.. _tut_stats_cluster_methods:

======================================================
Permutation t-test on toy data with spatial clustering
======================================================

Following the illustrative example of Ridgway et al. 2012 [1]_,
this demonstrates some basic ideas behind both the "hat"
variance adjustment method, as well as threshold-free
cluster enhancement (TFCE) [2]_ methods.

.. note:: This example does quite a bit of processing, so even on a
          fast machine it can take a few minutes to complete.

"""
# Authors: Eric Larson <larson.eric.d@gmail.com>
# License: BSD (3-clause)

from functools import partial

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa, analysis:ignore

from mne.stats import (spatio_temporal_cluster_1samp_test, fdr_correction,
                       bonferroni_correction, ttest_1samp_no_p)

print(__doc__)

###############################################################################
# Set parameters
# --------------
# This toy dataset consists of a 40 x 40 square with a "signal"
# present in the center (at pixel [20, 20]) with white noise
# added and a 5-pixel-SD normal smoothing kernel applied.
width = 40
n_subjects = 10
signal_mean = 100
signal_sd = 100
noise_sd = 0.01
gaussian_sd = 5
sigma = 1e-3  # sigma for the "hat" method
threshold = -stats.distributions.t.ppf(0.05, n_subjects - 1)
n_permutations = 1024  # number of clustering permutations (1024 for exact)

###############################################################################
# Construct simulated data
# ------------------------
n_src = width * width

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
# ------------------
# Let's start with a standard 1-sample t-test.
#
# .. note::
#     X needs to be a multi-dimensional array of shape
#     samples (subjects) x time x space, so we permute dimensions:
X = X.reshape((n_subjects, 1, n_src))
t_uncorrected, p_uncorrected = stats.ttest_1samp(X, 0)

# Hat correction
# ^^^^^^^^^^^^^^
# The "hat" correction can be used to deal with implausibly small variances.
t_uncorrected_hat = ttest_1samp_no_p(X, sigma=sigma)
p_uncorrected_hat = stats.distributions.t.sf(
    np.abs(t_uncorrected_hat), len(X) - 1) * 2

###############################################################################
# Bonferroni correction
# ^^^^^^^^^^^^^^^^^^^^^
# Perhaps the simplest multiple comparison correction, Bonferroni, multiplies
# the p-values by the number of comparisons.
p_bon = bonferroni_correction(p_uncorrected)[1]

###############################################################################
# FDR correction
# ^^^^^^^^^^^^^^
# A less restrictive correction that controls for the false-discovery rate
# (FDR) is the Benjamini-Hochberg procedure:
p_fdr = fdr_correction(p_uncorrected)[1]

###############################################################################
# Clustering
# ^^^^^^^^^^
# We can also use non-parametric clustering to correct for multiple comparisons
# and control the familywise error rate (FWER).
#
# .. note::
#     Using connectivity=None implies grid-like connectivity,
#     which is correct for our grid-like data here.
t_clust, clusters, p_values, H0 = \
    spatio_temporal_cluster_1samp_test(X, n_jobs=1, threshold=threshold,
                                       connectivity=None, tail=1,
                                       n_permutations=n_permutations)

#    Let's put the cluster data in a readable format
p_clust = np.zeros(width * width)
for cl, p in zip(clusters, p_values):
    p_clust[cl[1]] = p

###############################################################################
# "hat" variance correction
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# We can correct for implausibly small variances by using the "hat"
# correction [1]_:
stat_fun = partial(ttest_1samp_no_p, sigma=sigma)
t_hat, clusters, p_values, H0 = \
    spatio_temporal_cluster_1samp_test(X, n_jobs=1, threshold=threshold,
                                       connectivity=None, tail=1,
                                       n_permutations=n_permutations,
                                       stat_fun=stat_fun, buffer_size=None)

#    Let's put the cluster data in a readable format
p_hat = np.zeros(width * width)
for cl, p in zip(clusters, p_values):
    p_hat[cl[1]] = p

###############################################################################
# .. _tfce_example:
#
# Threshold-free cluster enhancemnet (TFCE)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For TFCE, we need to specify how we want to approximate a continuous
# integration across threshold values. This is done using a standard
# `Riemann sum <https://en.wikipedia.org/wiki/Riemann_sum>`_ technique.
# For this, a starting threshold ``'start'`` and a step size ``'step'``
# must be provided in a dict (the smaller the step and lower the starting
# value, the better the approximation, but the longer it takes):

threshold_tfce = dict(start=0, step=0.2)
t_tfce, clusters, p_tfce, H0 = \
    spatio_temporal_cluster_1samp_test(X, n_jobs=1, threshold=threshold_tfce,
                                       connectivity=None,
                                       tail=1, n_permutations=n_permutations)

###############################################################################
# TFCE with "hat" variance correction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can also combine TFCE and the "hat" correction:
t_tfce_hat, clusters, p_tfce_hat, H0 = \
    spatio_temporal_cluster_1samp_test(X, n_jobs=1, threshold=threshold_tfce,
                                       connectivity=None, tail=1,
                                       n_permutations=n_permutations,
                                       stat_fun=stat_fun, buffer_size=None)

###############################################################################
# Visualize results
# -----------------
# The top row shows t statistics, and the bottom shows p values for various
# statistical tests.
#
# Mass-univariate methods (parametric)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The first three columns show mass-univariate statistics, including:
#
# - The 1-sample t test with no correction. Note that it has peaky edges.
# - Bonferroni-corrected p-values.
# - FDR-corrected p-values.
#
# All of these mis-localize the peak due to sharpness in the t statistic
# driven by low-variance pixels toward the edge of the plateau. They also
# over-correct for multiple comparisons because neighboring voxels are
# correlated.
#
# Clustering (non-parametric)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Standard non-parametric, resampling-based clustering identifies the correct
# region. However, the whole area must be declared significant, so no peak
# analysis can be done. Also, the peak is broad. In this method, there are no
# assumptions of Gaussianity (which do hold for this example but do not in
# general).
#
# Clustering with "hat"
# ^^^^^^^^^^^^^^^^^^^^^
# Adding the "hat" technique tightens the estimate of significant activity.
#
# Clustering with TFCE
# ^^^^^^^^^^^^^^^^^^^^
# The TFCE approach allows analyzing each significant point independently,
# but still has a broadened estimate.
#
# Clustering with TFCE and "hat"
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Finally, combining the TFCE and "hat" methods tightens the area declared
# significant (again FWER corrected), and allows for evaluation of each point
# independently instead of as a single, broad cluster.

fig = plt.figure(facecolor='w', figsize=(10, 3))

x, y = np.mgrid[0:width, 0:width]
cmap = 'YlGnBu'
kwargs = dict(rstride=1, cstride=1, linewidth=0, cmap=cmap)

ts = [t_uncorrected, t_uncorrected_hat, None, None,
      t_clust, t_hat, t_tfce, t_tfce_hat]
titles = ['t statistic', '$\hat{\mathrm{t}}$', 'Bonferroni', 'FDR',
          'Clustering', r'$\hat{\mathrm{C}}$',
          r'$\mathrm{C}_{\mathrm{TFCE}}$',
          r'$\hat{\mathrm{C}}_{\mathrm{TFCE}}$']
for ii, (t, title) in enumerate(zip(ts, titles)):
    ax = fig.add_subplot(2, 8, ii + 1, projection='3d')
    if t is None:
        fig.delaxes(ax)
    else:
        ax.plot_surface(x, y, np.reshape(t, (width, width)), **kwargs)
        ax.set(xticks=[], yticks=[], zticks=[])

p_lims = [-np.log10(0.05), -np.log10(0.001)]
ps = [p_uncorrected, p_uncorrected_hat, p_bon, p_fdr,
      p_clust, p_hat, p_tfce, p_tfce_hat]
for ii, (p, title) in enumerate(zip(ps, titles)):
    ax = fig.add_subplot(2, 8, 9 + ii)
    img = ax.imshow(-np.log10(np.reshape(p, (width, width))), cmap=cmap,
                    vmin=p_lims[0], vmax=p_lims[1], interpolation='bilinear')
    ax.set(xticks=[], yticks=[], title=title)
    cbar = plt.colorbar(ax=ax, shrink=0.75, orientation='horizontal',
                        fraction=0.1, pad=0.025, mappable=img)
    cbar.set_label('-log10(p)')
    cbar.set_ticks(p_lims)
    cbar.set_ticklabels(['%0.1f' % p for p in p_lims])

fig.tight_layout()
plt.show()

###############################################################################
# References
# ----------
# .. [1] Ridgway et al. 2012, "The problem of low variance voxels in
#        statistical parametric mapping; a new hat avoids a 'haircut'",
#        NeuroImage. 2012 Feb 1;59(3):2131-41.
#
# .. [2] Smith and Nichols 2009, "Threshold-free cluster enhancement:
#        addressing problems of smoothing, threshold dependence, and
#        localisation in cluster inference", NeuroImage 44 (2009) 83-98.
