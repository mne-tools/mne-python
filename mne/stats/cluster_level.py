#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Thorsten Kranz <thorstenkranz@gmail.com>
#         Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import numpy as np
from scipy import ndimage
from scipy.stats import percentileofscore, ttest_1samp

from .parametric import f_oneway


def _find_clusters(x, threshold, tail=0):
    """For a given 1d-array (test statistic), find all clusters which
    are above/below a certain threshold. Returns a list of 2-tuples.

    Parameters
    ----------
    x: 1D array
        Data
    threshold: float
        Where to threshold the statistic
    tail : -1 | 0 | 1
        Type of comparison

    Returns
    -------
    clusters: list of slices or list of arrays (boolean masks)
        We use slices for 1D signals and mask to multidimensional
        arrays.

    sums: array
        Sum of x values in clusters
    """
    if not tail in [-1, 0, 1]:
        raise ValueError('invalid tail parameter')

    x = np.asanyarray(x)

    if tail == -1:
        x_in = x < threshold
    elif tail == 1:
        x_in = x > threshold
    else:
        x_in = np.abs(x) > threshold

    labels, n_labels = ndimage.label(x_in)

    if x.ndim == 1:
        clusters = ndimage.find_objects(labels, n_labels)
        sums = ndimage.measurements.sum(x, labels, index=range(1, n_labels+1))
    else:
        clusters = list()
        sums = np.empty(n_labels)
        for l in range(1, n_labels+1):
            c = labels == l
            clusters.append(c)
            sums[l-1] = np.sum(x[c])

    return clusters, sums


def _pval_from_histogram(T, H0, tail):
    """Get p-values from stats values given an H0 distribution

    For each stat compute a p-value as percentile of its statistics
    within all statistics in surrogate data
    """
    if not tail in [-1, 0, 1]:
        raise ValueError('invalid tail parameter')

    pval = np.array([percentileofscore(H0, t) for t in T])

    # from pct to fraction
    if tail == -1: # up tail
        pval =  pval / 100.0
    elif tail == 1: # low tail
        pval = (100.0 - pval) / 100.0
    elif tail == 0: # both tails
        pval = 100.0 - pval
        pval += np.array([percentileofscore(H0, -t) for t in T])

    return pval


def permutation_cluster_test(X, stat_fun=f_oneway, threshold=1.67,
                             n_permutations=1000, tail=0):
    """Cluster-level statistical permutation test

    For a list of 2d-arrays of data, e.g. power values, calculate some
    statistics for each timepoint (dim 1) over groups.  Do a cluster
    analysis with permutation test for calculating corrected p-values.
    Randomized data are generated with random partitions of the data.

    Parameters
    ----------
    X : list
        List of 2d-arrays containing the data, dim 1: timepoints, dim 2:
        elements of groups
    stat_fun : callable
        function called to calculate statistics, must accept 1d-arrays as
        arguments (default: scipy.stats.f_oneway)
    threshold : float
        The threshold for the statistic.
    n_permutations : int
        The number of permutations to compute.
    tail : -1 or 0 or 1 (default = 0)
        If tail is 1, the statistic is thresholded above threshold.
        If tail is -1, the statistic is thresholded below threshold.
        If tail is 0, the statistic is thresholded on both sides of
        the distribution.

    Returns
    -------
    T_obs : array of shape [n_tests]
        T-statistic observerd for all variables
    clusters: list of tuples
        Each tuple is a pair of indices (begin/end of cluster)
    cluster_pv: array
        P-value for each cluster
    H0 : array of shape [n_permutations]
        Max cluster level stats observed under permutation.

    Notes
    -----
    Reference:
    Cluster permutation algorithm as described in
    Maris/Oostenveld (2007),
    "Nonparametric statistical testing of EEG- and MEG-data"
    Journal of Neuroscience Methods, Vol. 164, No. 1., pp. 177-190.
    doi:10.1016/j.jneumeth.2007.03.024
    """
    X_full = np.concatenate(X, axis=0)
    n_samples_per_condition = [x.shape[0] for x in X]

    # Step 1: Calculate Anova (or other stat_fun) for original data
    # -------------------------------------------------------------
    T_obs = stat_fun(*X)

    clusters, cluster_stats = _find_clusters(T_obs, threshold, tail)

    # make list of indices for random data split
    splits_idx = np.append([0], np.cumsum(n_samples_per_condition))
    slices = [slice(splits_idx[k], splits_idx[k+1])
                                                for k in range(len(X))]

    # Step 2: If we have some clusters, repeat process on permuted data
    # -------------------------------------------------------------------
    if len(clusters) > 0:
        H0 = np.zeros(n_permutations) # histogram
        for i_s in range(n_permutations):
            np.random.shuffle(X_full)
            X_shuffle_list = [X_full[s] for s in slices]
            T_obs_surr = stat_fun(*X_shuffle_list)
            _, perm_clusters_sums = _find_clusters(T_obs_surr, threshold, tail)

            if len(perm_clusters_sums) > 0:
                H0[i_s] = np.max(perm_clusters_sums)
            else:
                H0[i_s] = 0

        cluster_pv = _pval_from_histogram(cluster_stats, H0, tail)
        return T_obs, clusters, cluster_pv, H0
    else:
        return T_obs, np.array([]), np.array([]), np.array([])


permutation_cluster_test.__test__ = False


def permutation_cluster_t_test(X, threshold=1.67, n_permutations=1000, tail=0):
    """Non-parametric cluster-level 1 sample T-test

    From a array of observations, e.g. signal amplitudes or power spectrum
    estimates etc., calculate if the observed mean significantly deviates
    from 0. The procedure uses a cluster analysis with permutation test
    for calculating corrected p-values. Randomized data are generated with
    random sign flips.

    Parameters
    ----------
    X: array
        Array where the first dimension corresponds to the
        samples (observations). X[k] can be a 1D or 2D array (time series
        or TF image) associated to the kth observation.
    threshold: float
        The threshold for the statistic.
    n_permutations: int
        The number of permutations to compute.
    tail : -1 or 0 or 1 (default = 0)
        If tail is 1, the statistic is thresholded above threshold.
        If tail is -1, the statistic is thresholded below threshold.
        If tail is 0, the statistic is thresholded on both sides of
        the distribution.

    Returns
    -------
    T_obs : array of shape [n_tests]
        T-statistic observerd for all variables
    clusters: list of tuples
        Each tuple is a pair of indices (begin/end of cluster)
    cluster_pv: array
        P-value for each cluster
    H0 : array of shape [n_permutations]
        Max cluster level stats observed under permutation.

    Notes
    -----
    Reference:
    Cluster permutation algorithm as described in
    Maris/Oostenveld (2007),
    "Nonparametric statistical testing of EEG- and MEG-data"
    Journal of Neuroscience Methods, Vol. 164, No. 1., pp. 177-190.
    doi:10.1016/j.jneumeth.2007.03.024
    """
    X_copy = X.copy()
    n_samples = X.shape[0]
    shape_ones = tuple([1] * X[0].ndim)
    # Step 1: Calculate T-stat for original data
    # -------------------------------------------------------------
    T_obs, _ = ttest_1samp(X, 0)

    clusters, cluster_stats = _find_clusters(T_obs, threshold, tail)

    # Step 2: If we have some clusters, repeat process on permuted data
    # -------------------------------------------------------------------
    if len(clusters) > 0:
        H0 = np.empty(n_permutations) # histogram
        for i_s in range(n_permutations):
            # new surrogate data with random sign flip
            signs = np.sign(0.5 - np.random.rand(n_samples, *shape_ones))
            X_copy *= signs

            # Recompute statistic on randomized data
            T_obs_surr, _ = ttest_1samp(X_copy, 0)
            _, perm_clusters_sums = _find_clusters(T_obs_surr, threshold, tail)

            if len(perm_clusters_sums) > 0:
                idx_max = np.argmax(np.abs(perm_clusters_sums))
                H0[i_s] = perm_clusters_sums[idx_max] # get max with sign info
            else:
                H0[i_s] = 0

        cluster_pv = _pval_from_histogram(cluster_stats, H0, tail)

        return T_obs, clusters, cluster_pv, H0
    else:
        return T_obs, np.array([]), np.array([]), np.array([])


permutation_cluster_t_test.__test__ = False

# if __name__ == "__main__":
#     noiselevel = 30
#     np.random.seed(0)
# 
#     # 1D
#     normfactor = np.hanning(20).sum()
#     condition1 = np.random.randn(50, 300) * noiselevel
#     for i in range(50):
#         condition1[i] = np.convolve(condition1[i], np.hanning(20),
#                                       mode="same") / normfactor
#     condition2 = np.random.randn(43, 300) * noiselevel
#     for i in range(43):
#         condition2[i] = np.convolve(condition2[i], np.hanning(20),
#                                       mode="same") / normfactor
#     pseudoekp = 5 * np.hanning(150)[None,:]
#     condition1[:, 100:250] += pseudoekp
#     condition2[:, 100:250] -= pseudoekp
# 
#     # Make it 2D
#     condition1 = np.tile(condition1[:,100:275,None], (1, 1, 15))
#     condition2 = np.tile(condition2[:,100:275,None], (1, 1, 15))
#     shape1 = condition1[..., :3].shape
#     shape2 = condition2[..., :3].shape
#     condition1[..., :3] = np.random.randn(*shape1) * noiselevel
#     condition2[..., :3] = np.random.randn(*shape2) * noiselevel
#     condition1[..., -3:] = np.random.randn(*shape1) * noiselevel
#     condition2[..., -3:] = np.random.randn(*shape2) * noiselevel
# 
#     # X, threshold, tail = condition1, 1.67, 1
#     # X, threshold, tail = -condition1, -1.67, -1
#     # # X, threshold, tail = condition1, 1.67, 0
#     # fs, clusters, cluster_p_values, histogram = permutation_cluster_t_test(
#     #                                     condition1, n_permutations=500, tail=tail,
#     #                                     threshold=threshold)
# 
#     # import pylab as pl
#     # pl.close('all')
#     # pl.hist(histogram)
#     # pl.show()
# 
#     fs, clusters, cluster_p_values, histogram = permutation_cluster_test(
#                                 [condition1, condition2], n_permutations=1000)
#     
#     # Plotting for a better understanding
#     import pylab as pl
#     pl.close('all')
#     
#     if condition1.ndim == 2:
#         pl.subplot(211)
#         pl.plot(condition1.mean(axis=0), label="Condition 1")
#         pl.plot(condition2.mean(axis=0), label="Condition 2")
#         pl.ylabel("signal [a.u.]")
#         pl.subplot(212)
#         for i_c, c in enumerate(clusters):
#             c = c[0]
#             if cluster_p_values[i_c] <= 0.05:
#                 h = pl.axvspan(c.start, c.stop-1, color='r', alpha=0.3)
#             else:
#                 pl.axvspan(c.start, c.stop-1, color=(0.3, 0.3, 0.3), alpha=0.3)
#         hf = pl.plot(fs, 'g')
#         pl.legend((h, ), ('cluster p-value < 0.05', ))
#         pl.xlabel("time (ms)")
#         pl.ylabel("f-values")
#     else:
#         fs_plot = np.nan * np.ones_like(fs)
#         for c, p_val in zip(clusters, cluster_p_values):
#             if p_val <= 0.05:
#                 fs_plot[c] = fs[c]
#     
#         pl.imshow(fs.T, cmap=pl.cm.gray)
#         pl.imshow(fs_plot.T, cmap=pl.cm.jet)
#         # pl.imshow(fs.T, cmap=pl.cm.gray, alpha=0.6)
#         # pl.imshow(fs_plot.T, cmap=pl.cm.jet, alpha=0.6)
#         pl.xlabel('time')
#         pl.ylabel('Freq')
#         pl.colorbar()
#     
#     pl.show()
