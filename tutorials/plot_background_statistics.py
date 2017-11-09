"""
==========
Statistics
==========

Here we will briefly cover multiple statistical concepts in an
introductory manner, and show how to use MNE statistical functions.

.. contents:: Topics
   :local:
   :depth: 2

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
# Hypothesis testing
# ------------------
# `Null hypothesis <https://en.wikipedia.org/wiki/Null_hypothesis>`_
#   In inferential statistics, a general statement or default position that
#   there is no relationship between two measured phenomena, or no association
#   among groups.
#
# We often want to reject a **null hypothesis** with
# at some probability level (e.g., p < 0.05).
#
# To think about what this means, let's follow the illustrative example from
# [1]_ and construct a toy dataset consists of a 40 x 40 square with a "signal"
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
n_src = width * width

# For each "subject", make a smoothed noisy signal with a centered peak
rng = np.random.RandomState(42)
X = noise_sd * rng.randn(n_subjects, width, width)
# Add a signal at the dead center
X[:, width // 2, width // 2] = signal_mean + rng.randn(n_subjects) * signal_sd
# Spatially smooth with a 2D Gaussian kernel
size = width // 2 - 1
gaussian = np.exp(-(np.arange(-size, size + 1) ** 2 / float(gaussian_sd ** 2)))
for si in range(X.shape[0]):
    for ri in range(X.shape[1]):
        X[si, ri, :] = np.convolve(X[si, ri, :], gaussian, 'same')
    for ci in range(X.shape[2]):
        X[si, :, ci] = np.convolve(X[si, :, ci], gaussian, 'same')

###############################################################################
# In this case, a null hypothesis we could test is for each voxel is:
#
#     There is no difference between the mean value and zero ($H_0: \mu = 0$).
#
# The alternative hypothesis, then, is that the voxel has a non-zero mean.
# This is a *two-tailed test* because this means the mean could be less than
# or greater-than zero (whereas a *one-tailed test* would test only one of
# these possibilities, i.e. $H_0: \mu > 0$ or $H_0: \mu < 0$).
#
# Parametric (mass-univariate) tests
# ----------------------------------
# Let's start with a **1-sample t-test**, which is a standard test
# for testing differences in paired sample means. This test is **parametric**,
# as it assumes
# that the underlying sample distribution is Gaussian, and is only valid in
# this case. (This happens to be satisfied by our toy dataset, but
# is not always satisfied for processed M/EEG data.)
#
# In the context of our toy dataset, which has many voxels, applying the
# 1-sample t test is called a *mass-univariate* approach as it treats
# each voxel independently.

# X needs to be a multi-dimensional array of shape
# ``(n_samples, n_time, n_space)`` (where samples here are subjects),
# so we permute dimensions:
X = X.reshape((n_subjects, 1, n_src))
t_uncorrected, p_uncorrected = stats.ttest_1samp(X, 0)

###############################################################################
# The "hat" variance correction
#   Regularizes the variance values used in the t-test calculation [1]_
#   to compensate for ]implausibly small variances.
t_uncorrected_hat = ttest_1samp_no_p(X, sigma=sigma)
p_uncorrected_hat = stats.distributions.t.sf(
    np.abs(t_uncorrected_hat), len(X) - 1) * 2

###############################################################################
# So far, we have done no correction for multiple comparisons.
#
# `Bonferroni<https://en.wikipedia.org/wiki/Bonferroni_correction>`_
#   Perhaps the simplest way to deal with multiple comparisons, it
#   multiplies the p-values by the number of comparisons to control the
#   familywise error rate (FWER).
p_bon = bonferroni_correction(p_uncorrected)[1]

###############################################################################
# `FDR<https://en.wikipedia.org/wiki/False_discovery_rate>`_
#   The **false-discovery rate (FDR) correction**, typically done using the
#   Benjamini/Hochberg procedure, is less restrictive than the Bonferroni
#   procedure for large numbers of comparisons
p_fdr = fdr_correction(p_uncorrected)[1]

###############################################################################
# .. _tfce_example:
#
# Non-parametric (cluster-based) resampling methods
# -------------------------------------------------
# **Non-parametric clustering** can also be used to correct for multiple
# comparisons. To use this, we need to rethink our null hypothesis. Instead
# of thinking about a null hypothesis about means per voxel, we consider a
# null hypothesis about cluster sizes, which could be stated like:
#
#     The distribution of spatial cluster sizes observed in two experimental]
#     conditions are drawn from the same probability distribution.
#
# Here we only have a single condition and we contrast to zero, which can
# be thought of as:
#
#     The distribution of spatial cluster sizes is independent of the sign
#     of the data.
#
# To actually implement this, we use resampling with a maximal statistic.
# Briefly, using exchangability under the null hypothesis, we can permute
# the data multiple times (which here means flip the signs), and for each
# permutation, record the maximum cluster size to bootstrap the null
# distribution of cluster sizes. The p-value for each cluster from
# the veridical data is simply given by the proportion of null distrubtion
# cluster sizes that were smaller.
#
# This reframing to consider *cluster sizes* rather than *individual means*
# has multiple advantages, including (but not limited to):
#
# 1. It controls the familywise error rate (FWER).
# 2. It is non-parametric. Even though our initial test statistic
#    (here a 1-sample t-test) for clustering parametric, the null
#    distribution for the null hypothesis rejection (cluster size
#    distribution is indistinguishable from zero) is obtained by
#    resampling. This means that it makes no assumptions of Gaussianity
#    (which do hold for this example but do not in general).
# 3. It accounts for the correlation structure in the data. The correlation
#    structure -- which in this case is spatial
#    but in general can be multidimensional (e.g., spatio-temporal) --]
#    is accounted for in the correction, because the null
#    distribution will be derived from data that preserve these correlations.
#
# However, if a cluster significantly deviates from the null, no further
# inference on the cluster (e.g., peak location) can be used, as the entire
# cluster is declared signficant.
#
# .. note::
#     Using connectivity=None here implies grid-like connectivity,
#     which is correct for our toy data which are defined on a grid.
t_clust, clusters, p_values, H0 = \
    spatio_temporal_cluster_1samp_test(X, n_jobs=1, threshold=threshold,
                                       connectivity=None, tail=1,
                                       n_permutations=n_permutations)

#    Let's put the cluster data in a readable format
p_clust = np.zeros(width * width)
for cl, p in zip(clusters, p_values):
    p_clust[cl[1]] = p

###############################################################################
# "hat" correction
#   This method can also be used in this context to correct for small
#   variances [1]_:
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
# Threshold-free cluster enhancemnet (TFCE)
#   This eliminates the free parameter of choosing an initial
#   threshold value to include points for clustering by
#   approximating a continuous integration across possible threshold values
#   with a standard
#   `Riemann sum <https://en.wikipedia.org/wiki/Riemann_sum>`_ [2]_.
#   This requires giving a starting threshold ``'start'`` and a step
#   size ``'step'`` in a dict (the smaller the step and lower the starting
#   value, the better the approximation, but the longer it takes):
#
# One significant advantage of TFCE is that, rather than modifying the
# statistical test, it modifies the data. The statistical test is then done
# on a voxel-by-voxel level. This allows for evaluation of each point
# independently for significance rather than only as cluster groups.

threshold_tfce = dict(start=0, step=0.2)
t_tfce, clusters, p_tfce, H0 = \
    spatio_temporal_cluster_1samp_test(X, n_jobs=1, threshold=threshold_tfce,
                                       connectivity=None,
                                       tail=1, n_permutations=n_permutations)

###############################################################################
# We can also combine TFCE and the "hat" correction:
t_tfce_hat, clusters, p_tfce_hat, H0 = \
    spatio_temporal_cluster_1samp_test(X, n_jobs=1, threshold=threshold_tfce,
                                       connectivity=None, tail=1,
                                       n_permutations=n_permutations,
                                       stat_fun=stat_fun, buffer_size=None)

###############################################################################
# Visualize results
# -----------------
# Let's take a look at these statistics. The top row shows each test statistic,
# and the bottom shows p values for various statistical tests.

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
    use_p = -np.log10(np.reshape(np.maximum(p, 1e-5), (width, width)))
    img = ax.imshow(use_p, cmap=cmap, vmin=p_lims[0], vmax=p_lims[1],
                    interpolation='bilinear')
    ax.set(xticks=[], yticks=[], title=title)
    cbar = plt.colorbar(ax=ax, shrink=0.75, orientation='horizontal',
                        fraction=0.1, pad=0.025, mappable=img)
    cbar.set_label('-log10(p)')
    cbar.set_ticks(p_lims)
    cbar.set_ticklabels(['%0.1f' % p for p in p_lims])

fig.tight_layout()
plt.show()

###############################################################################
# Mass-univariate methods (parametric)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The four three columns show mass-univariate statistics, including:
#
# - **uncorrected t test** with no correction has peaky edges.
# - **"hat" variance correction** has reduced peaky
#   edges.
# - **Bonferroni correction** eliminates any significant activity.
# - **FDR correction** is less conservative than Bonferroni.
#
# The non-hat-corrected tests mis-localize the peak due to sharpness in the
# t statistic driven by low-variance pixels toward the edge of the plateau.
# Both Bonferroni and FDR too conservatively correct for multiple comparisons]
# because neighboring voxels are correlated.
#
# Clustering methods (non-parametric)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The next four columns show clustering results:
#
# - **Standard clustering** identifies the correct region. However, the whole
#   area must be declared significant, so no peak analysis can be done.
#   Also, the peak is broad.
# - **Clustering with "hat"** tightens the estimate of significant activity.
# - **Clustering with TFCE** allows analyzing each significant point
#   independently, but still has a broadened estimate.
# - **Clustering with TFCE and "hat"** tightens the area declared
#   significant (again FWER corrected).
#
# Statistical functions in MNE
# ----------------------------
# Parametric methods
# ^^^^^^^^^^^^^^^^^^
#
# - :func:`mne.stats.ttest_1samp_no_p`
#      Compute an optimized one-sample t test, optionally with hat adjustment.
#
#   This is used by default for contrast enhancement in paired cluster tests.
#
# - :func:`mne.stats.f_oneway`
#     An optimized version of the F-test for independent samples.
#     This can be used to compute various F-contrasts. It is used by default
#     for contrast enhancement in non-paired cluster tests.
#
# - :func:`mne.stats.f_mway_rm`
#     Compute a generalized M-way repeated measures ANOVA for balanced designs.
#     This returns F-statistics and p-valus. The associated helper function
#     :func:`mne.stats.f_threshold_mway_rm` can be used to determine the
#     F-threshold at a given significance level and set of degrees of freedom.
#
# - :func:`mne.stats.linear_regression`
#     Compute ordinary least square regressions on multiple targets, e.g.,
#     sensors, time points across trials (samples).
#     For each regressor it returns the beta values, t-staistics, and
#     uncorrected significance values. While it can be used as a test it is
#     particularly useful to compute weighted averages.
#
# .. note:: The :mod:`statsmodels` package offers functions for computing
#           many different statistical contrasts. If the one you need
#           (e.g., the interaction term in an unbalanced ANOVA) is not
#           shown here, check e.g., :func:`statsmodels.stats.anova.anova_lm`.
#
# Non-parametric methods
# ^^^^^^^^^^^^^^^^^^^^^^
# - :func:`mne.stats.permutation_cluster_test`
#     Unpaired contrasts with spatial connectivity.
#
# - :func:`mne.stats.spatio_temporal_cluster_test`
#     Unpaired contrasts with spatio-temporal connectivity.
#
# - :func:`mne.stats.permutation_cluster_1samp_test`
#     Paired contrasts with spatial connectivity.
#
# - :func:`mne.stats.spatio_temporal_cluster_1samp_test`
#     Paired contrasts with spatio-temporal connectivity.
#
# References
# ----------
# .. [1] Ridgway et al. 2012, "The problem of low variance voxels in
#        statistical parametric mapping; a new hat avoids a 'haircut'",
#        NeuroImage. 2012 Feb 1;59(3):2131-41.
#
# .. [2] Smith and Nichols 2009, "Threshold-free cluster enhancement:
#        addressing problems of smoothing, threshold dependence, and
#        localisation in cluster inference", NeuroImage 44 (2009) 83-98.
