"""
=====================
Statistical inference
=====================

Here we will briefly cover multiple concepts of inferential statistics in an
introductory manner, and demonstrate how to use some MNE statistical functions.

.. contents:: Topics
   :local:
   :depth: 3

"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
# License: BSD (3-clause)

from functools import partial

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa, analysis:ignore

from mne.stats import (spatio_temporal_cluster_1samp_test, fdr_correction,
                       bonferroni_correction, ttest_1samp_no_p,
                       permutation_t_test)

print(__doc__)

###############################################################################
# Hypothesis testing
# ------------------
# Null hypothesis
# ^^^^^^^^^^^^^^^
# From `Wikipedia <https://en.wikipedia.org/wiki/Null_hypothesis>`_:
#
#     In inferential statistics, a general statement or default position that
#     there is no relationship between two measured phenomena, or no
#     association among groups.
#
# We typically want to reject a **null hypothesis** with
# at some probability level (e.g., p < 0.05).
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
#     There is no difference between the mean value and zero
#     (:math:`H_0: \mu = 0`).
#
# The alternative hypothesis, then, is that the voxel has a non-zero mean.
# This is a *two-tailed test* because this means the mean could be less than
# or greater-than zero (whereas a *one-tailed test* would test only one of
# these possibilities, i.e. :math:`H_0: \mu \geq 0` or
# :math:`H_0: \mu \leq 0`).
#
# Parametric tests
# ^^^^^^^^^^^^^^^^
# Let's start with a **1-sample t-test**, which is a standard test
# for differences in paired sample means. This test is **parametric**,
# as it assumes that the underlying sample distribution is Gaussian, and is
# only valid in this case. (This happens to be satisfied by our toy dataset,
# but is not always satisfied for neuroimaging data.)
#
# In the context of our toy dataset, which has many voxels, applying the
# 1-sample t test is called a *mass-univariate* approach as it treats
# each voxel independently.

# X needs to be a multi-dimensional array of shape (n_samples, n_time, n_space)
# (where samples here are subjects) so we permute dimensions:
t_uncorrected, p_uncorrected = stats.ttest_1samp(X, 0, axis=0)

###############################################################################
# "hat" variance adjustment
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# The "hat" technique regularizes the variance values used in the t-test
# calculation [1]_ to compensate for implausibly small variances.
t_uncorrected_hat = ttest_1samp_no_p(X, sigma=sigma)
p_uncorrected_hat = stats.distributions.t.sf(
    np.abs(t_uncorrected_hat), len(X) - 1) * 2

###############################################################################
# Non-parametric tests
# ^^^^^^^^^^^^^^^^^^^^
# Instead of assuming an underlying Gaussian distribution, we could instead
# use a **non-parametric permutation** method. Under the null hypothesis,
# we have the princple of **exchangability**, which means that, if the null
# is true, we should be able to exchange conditions and not change the
# distribution of the test statistic.
#
# In the case of a paired t test against 0 (or between two conditions where
# you have already subtracted them), exchangeability means that we can flip
# the signs of our data. Therefor we can bootstrap the **null distribution**
# values by taking random subsets of samples (subjects), flipping the sign
# of their data, and recording the resulting statistic value. The value from
# the statistic evaluateid on the veridical data can then be compared to this
# distrubiton, and the p value is simply the proportion of null distrtibution
# values that were smaller.

# Here we have to do a bit of gymnastics to get our function to do
# a permutation test without correcting for multiple comparisons:

X.shape = (n_subjects, n_src)
t_uncorrected_perm = np.zeros(width * width)
p_uncorrected_perm = np.zeros(width * width)
for ii in range(n_src):
    t_uncorrected_perm[ii], p_uncorrected_perm[ii] = \
        permutation_t_test(X[:, [ii]])[:2]

###############################################################################
# Multiple comparisons
# --------------------
# So far, we have done no correction for multiple comparisons. This is
# potentially problematic for these data because there are
# :math:`40 \times 40 = 1600` tests being performed. If we just use
# a threshold ``p < 0.05`` for all of our tests, we would expect many
# voxels to be declared significant even if there were no true effect.
# In other words, we would make many **Type I errors** (adapted from
# `Wikipedia <https://en.wikipedia.org/wiki/Type_I_and_type_II_errors>`_):
#
# .. rst-class:: skinnytable
#
#   +----------+--------+------------------+------------------+
#   |                   |          Null hypothesis            |
#   |                   +------------------+------------------+
#   |                   |       True       |       False      |
#   +==========+========+==================+==================+
#   |          |        | Type I error     | Correct          |
#   |          | Yes    |   False positive |   True positive  |
#   + Reject   +--------+------------------+------------------+
#   |          |        | Correct          | Type II error    |
#   |          | No     |   True Negative  |   False negative |
#   +----------+--------+------------------+------------------+
#
# To combat this problem, multiple methods exist. Typically these
# provide control over either the:
#
# 1. `Familywise error rate (FWER) <fwer>`_
#      The probability of making one or more Type I error:
#      :math:`p(N_{\mathrm{Type\ I}} >= 1)`.
# 2. `False discovery rate (FDR) <fdr>`_
#      The expected proportion of rejected null hypotheses that are false:
#      :math:`N_{\mathrm{Type\ I}} / N_{\mathrm{reject}}`.
#
# We cover some techniques that control FWER and FDR below.
#
# Bonferroni correction
# ^^^^^^^^^^^^^^^^^^^^^
# Perhaps the simplest way to deal with multiple comparisons,
# `Bonferroni <https://en.wikipedia.org/wiki/Bonferroni_correction>`_
# correction multiplies the p-values by the number of comparisons to
# control the FWER.

p_bon = bonferroni_correction(p_uncorrected)[1]

###############################################################################
# False discovery rate (FDR) correction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Typically FDR is performed with the Benjamini/Hochberg procedure, which
# is less restrictive than Bonferroni correction for large numbers of
# comparisons (fewer Type II errors) but provides less strict control of
# errors (more Type I errors).

p_fdr = fdr_correction(p_uncorrected)[1]

###############################################################################
# Non-parametric permutation test with a maximal statistic
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# **Non-parametric permutation tests** can also be used to correct for multiple
# comparisons. In its simplest form, we again do permutations using
# exchangeability under the null hypothesis, but this time we take the
# *maximum statistic across all tests* in each permutation to form the
# null distribution. The p-value for each vertex from the veridical data
# then given by the proportion of null distrubtion values
# that were smaller than.
#
# This method has two important features:
#
# 1. It controls FWER.
# 2. It is non-parametric. Even though our initial test statistic
#    (here a 1-sample t-test) for clustering parametric, the null
#    distribution for the null hypothesis rejection (cluster size
#    distribution is indistinguishable from zero) is obtained by
#    permutations. This means that it makes no assumptions of Gaussianity
#    (which do hold for this example but do not in general).

t_perm, p_perm = permutation_t_test(X)[:2]

###############################################################################
# Clustering
# ^^^^^^^^^^
# Each of the aforementioned multiple comparisons corrections have the
# disadvantage of not fully incorporating the correlation structure of the
# data. However, by defining the connectivity/adjacency/neighbor structure
# in our data, we can use **clustering** to compensate.
#
# To use this, we need to rethink our null hypothesis. Instead
# of thinking about a null hypothesis about means per voxel, we consider a
# null hypothesis about sizes of clusters in our data, which could be stated
# like:
#
#     The distribution of spatial cluster sizes observed in two experimental
#     conditions are drawn from the same probability distribution.
#
# Here we only have a single condition and we contrast to zero, which can
# be thought of as:
#
#     The distribution of spatial cluster sizes is independent of the sign
#     of the data.
#
# In this case, we again do a permutations with a maximal statistic, but, under
# each permutation, we:
#
# 1. Threshold the computed statistic with some **initial**
#    ``threshold`` value.
# 2. Cluster points that exceed this threshold (with the same sign)
#    based on adjacency.
# 3. Record the *size* of each cluster (measured, e.g., by a simple vertex
#    count or the sum of voxel t values within the cluster).
#
# After doing these permutations, the cluster sizes in our veridical data
# are compared to this null distribution. The p value associated with each
# cluster is again given by the proportion of smaller null distribution
# values. This can then be subjected to a standard p-value threshold
# (e.g., ``p < 0.05``) to reject the null hypothesis (i.e., find an effect
# of interest).
#
# This reframing to consider *cluster sizes* rather than *individual means*
# maintains the advantages of the standard non-parametric permutation
# test -- namely controlling FWER and making no assumptions of parametric
# data distribution.
# Cricitally, though, it also accounts for the correlation structure in the
# data -- which in this toy case is spatial but
# in general can be multidimensional (e.g., spatio-temporal) -- because the
# null distribution will be derived from data in a way that preserves these
# correlations.
#
# However, there is a drawback. If a cluster significantly deviates from
# the null, no further inference on the cluster (e.g., peak location) can be
# made, as the entire cluster as a whole is used to reject the null.

# We need to define our connectivity/neighbor/adjacency matrix, which we know
# is just a grid. Using ``connectivity=None`` is highly optimized for this
# case.
X.shape = (n_subjects, width, width)
t_clust, clusters, p_values, H0 = \
    spatio_temporal_cluster_1samp_test(X, n_jobs=1, threshold=threshold,
                                       connectivity=None, tail=1,
                                       n_permutations=n_permutations)
# Put the cluster data in a viewable format
p_clust = np.ones((width, width))
for cl, p in zip(clusters, p_values):
    p_clust[cl[0], cl[1]] = p

###############################################################################
# "hat" variance adjustment
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# This method can also be used in this context to correct for small
# variances [1]_:
stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
t_hat, clusters, p_values, H0 = \
    spatio_temporal_cluster_1samp_test(X, n_jobs=1, threshold=threshold,
                                       connectivity=None, tail=1,
                                       n_permutations=n_permutations,
                                       stat_fun=stat_fun_hat, buffer_size=None)
p_hat = np.ones((width, width))
for cl, p in zip(clusters, p_values):
    p_hat[cl[0], cl[1]] = p

###############################################################################
# .. _tfce_example:
#
# Threshold-free cluster enhancement (TFCE)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TFCE eliminates the free parameter initial ``threshold`` value that
# determines which points are included in clustering by approximating
# a continuous integration across possible threshold values with a standard
# `Riemann sum <https://en.wikipedia.org/wiki/Riemann_sum>`_ [2]_.
# This requires giving a starting threshold ``'start'`` and a step
# size ``'step'``, which in MNE is supplied as a dict.
# The smaller the ``'step'`` and lower closer to 0 the ``'start'`` value,
# the better the approximation, but the longer it takes).
#
# A significant advantage of TFCE is that, rather than modifying the
# statistical test, it modifies the data itself (while still controlling
# for multiple comparisons). The statistical test is then done at the level
# of voxels rather than clusters. This allows for evaluation of each point
# independently for significance rather than only as cluster groups.

threshold_tfce = dict(start=0, step=0.2)
t_tfce, _, p_tfce, H0 = \
    spatio_temporal_cluster_1samp_test(X, n_jobs=1, threshold=threshold_tfce,
                                       connectivity=None,
                                       tail=1, n_permutations=n_permutations)

###############################################################################
# We can also combine TFCE and the "hat" correction:
t_tfce_hat, _, p_tfce_hat, H0 = \
    spatio_temporal_cluster_1samp_test(X, n_jobs=1, threshold=threshold_tfce,
                                       connectivity=None, tail=1,
                                       n_permutations=n_permutations,
                                       stat_fun=stat_fun_hat, buffer_size=None)

###############################################################################
# Visualize and compare methods
# -----------------------------
# Let's take a look at these statistics. The top row shows each test statistic,
# and the bottom shows p values for various statistical tests, with the ones
# with proper control over FWER or FDR with bold titles.

fig = plt.figure(facecolor='w', figsize=(12, 3))

x, y = np.mgrid[0:width, 0:width]
cmap = 'YlGnBu'
kwargs = dict(rstride=1, cstride=1, linewidth=0, cmap=cmap)

mccs = [False, False, False,
        True, True, True,
        True, True, True, True]
ts = [t_uncorrected, t_uncorrected_hat, t_uncorrected_perm,
      None, None, t_perm,
      t_clust, t_hat, t_tfce, t_tfce_hat]
titles = ['t statistic', '$\mathrm{t_{hat}}$', 'Permutation',
          'Bonferroni', 'FDR', '$\mathbf{Perm_{max}}$',
          'Clustering', r'$\mathbf{C_{hat}}$',
          r'$\mathbf{C_{TFCE}}$',
          r'$\mathbf{C_{hat,TFCE}}$']
for ii, t in enumerate(ts):
    ax = fig.add_subplot(2, 10, ii + 1, projection='3d')
    if t is None:
        fig.delaxes(ax)
    else:
        ax.plot_surface(x, y, np.reshape(t, (width, width)), **kwargs)
        ax.set(xticks=[], yticks=[], zticks=[])

p_lims = [-np.log10(0.05), -np.log10(0.001)]
ps = [p_uncorrected, p_uncorrected_hat, p_uncorrected_perm,
      p_bon, p_fdr, p_perm,
      p_clust, p_hat, p_tfce, p_tfce_hat]
for ii, (p, title, mcc) in enumerate(zip(ps, titles, mccs)):
    ax = fig.add_subplot(2, 10, 11 + ii)
    use_p = -np.log10(np.reshape(np.maximum(p, 1e-5), (width, width)))
    img = ax.imshow(use_p, cmap=cmap, vmin=p_lims[0], vmax=p_lims[1],
                    interpolation='bilinear')
    ax.set(xticks=[], yticks=[], title=title)
    if mcc:
        ax.title.set_weight('bold')
    cbar = plt.colorbar(ax=ax, shrink=0.75, orientation='horizontal',
                        fraction=0.1, pad=0.025, mappable=img)
    cbar.set_label('-log10(p)')
    cbar.set_ticks(p_lims)
    cbar.set_ticklabels(['%0.1f' % p for p in p_lims])

fig.tight_layout(pad=0, w_pad=0.1, h_pad=0.1)
plt.show()

###############################################################################
# The first three columns show the parametric and non-parametric statistics
# that are not corrected for multiple comparisons:
#
# - **t test** has peaky edges.
# - **"hat" variance correction** of the t test has reduced peaky edges,
#   correcting for sharpness in the statistic driven by low-variance voxels.
# - **non-parametric permutation test** is very similar to the t test, as
#   the data are parametric.
#
# The next three columns show multiple comparison corrections of the
# mass univariate tests (parametric and non-parametric). These
# too conservatively correct for multiple comparisons because neighboring
# voxels in our data are correlated:
#
# - **Bonferroni correction** eliminates any significant activity.
# - **FDR correction** is less conservative than Bonferroni.
# - **Permutation test with a maximal statistic** also eliminates any
#   significant activity.
#
# The final four columns show the non-parametric, cluster-based permuatation
# tests with a maximal statistic:
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
# MNE provides several convenience parametric testing functions that can be
# used in conjunction with the non-parametric clustering methods. However,
# the set of functions we provide is not meant to be exhaustive.
#
# If the univariate statistical contrast of interest to you is not listed
# here (e.g., interaction term in an unbalanced ANOVA), consider checking
# out the :mod:`statsmodels` package. It offers many functions for computing
# statistical contrasts, e.g., :func:`statsmodels.stats.anova.anova_lm`.
# To use these functions in clustering:
#
# 1. Determine which statistical test you would use in a univariate context
#    (e.g., to compute your contrast of interest if there only a single
#    output, like reaction times).
# 2. Wrap the call to that function within a function that takes an input of
#    the same shape that is expected by your clustering function,
#    and returns an array of the same shape without the "samples" dimension
#    (e.g., :func:`mne.stats.permutation_cluster_1samp_test` takes an array
#    of shape ``(n_samples, p, q)`` and returns an array of shape ``(p, q)``).
# 3. Pass this wrapped function to the ``stat_fun`` argument to the clustering
#    function.
# 4. Set an appropriate ``threshold`` value (float or dict) based on the
#    values your statistical contrast function returns.
#
# Parametric methods provided by MNE
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# - :func:`mne.stats.ttest_1samp_no_p`
#     Paired t test, optionally with hat adjustment.
#     This is used by default for contrast enhancement in paired cluster tests.
#
# - :func:`mne.stats.f_oneway`
#     One-way ANOVA for independent samples.
#     This can be used to compute various F-contrasts. It is used by default
#     for contrast enhancement in non-paired cluster tests.
#
# - :func:`mne.stats.f_mway_rm`
#     M-way ANOVA for repeated measures and balanced designs.
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
# Non-parametric methods
# ^^^^^^^^^^^^^^^^^^^^^^
#
# - :func:`mne.stats.permutation_cluster_test`
#     Unpaired contrasts with spatial connectivity.
#
# - :func:`mne.stats.spatio_temporal_cluster_test`
#     Unpaired contrasts with spatio-temporal connectivity.
#
# - :func:`mne.stats.permutation_t_test`
#     Paired contrast with no connectivity.
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
#
# .. _fwer: https://en.wikipedia.org/wiki/Family-wise_error_rate
# .. _fdr: https://en.wikipedia.org/wiki/False_discovery_rate
