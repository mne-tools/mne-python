#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Thorsten Kranz <thorstenkranz@gmail.com>
#         Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import numpy as np
from scipy import stats, ndimage
from scipy.stats import percentileofscore
from scikits.learn.feature_selection import univariate_selection

def f_oneway(*args):
    """Call scipy.stats.f_oneway, but return only f-value"""
    return univariate_selection.f_oneway(*args)[0]
    # return stats.f_oneway(*args)[0]

# def best_component(x, threshold, tail=0):
#     if tail == -1:
#         x_in = x < threshold
#     elif tail == 1:
#         x_in = x > threshold
#     else:
#         x_in = np.abs(x) > threshold
#     labels, n_labels = ndimage.label(x_in)
#     return np.max(ndimage.measurements.sum(x, labels, index=range(1, n_labels+1)))

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
    clusters: list of tuples
        Each tuple is a pair of indices (begin/end of cluster)

    Example
    -------
    >>> _find_clusters([1, 2, 3, 1], 1.9, tail=1)
    [(1, 3)]
    >>> _find_clusters([2, 2, 3, 1], 1.9, tail=1)
    [(0, 3)]
    >>> _find_clusters([1, 2, 3, 2], 1.9, tail=1)
    [(1, 4)]
    >>> _find_clusters([1, -2, 3, 1], 1.9, tail=0)
    [(1, 3)]
    >>> _find_clusters([1, -2, -3, 1], -1.9, tail=-1)
    [(1, 3)]
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
    return ndimage.find_objects(labels, n_labels)


def permutation_1d_cluster_test(X, stat_fun=f_oneway, threshold=1.67,
                             n_permutations=1000, tail=0):
    """Cluster-level statistical permutation test

    For a list of 2d-arrays of data, e.g. power values, calculate some
    statistics for each timepoint (dim 1) over groups.  Do a cluster
    analysis with permutation test for calculating corrected p-values

    Parameters
    ----------
    X: list
        List of 2d-arrays containing the data, dim 1: timepoints, dim 2:
        elements of groups
    stat_fun : callable
        function called to calculate statistics, must accept 1d-arrays as
        arguments (default: scipy.stats.f_oneway)
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
    n_samples_total = X_full.shape[0]
    n_samples_per_condition = [x.shape[0] for x in X]

    # Step 1: Calculate Anova (or other stat_fun) for original data
    # -------------------------------------------------------------
    T_obs = stat_fun(*X)

    clusters = _find_clusters(T_obs, threshold, tail)

    # make list of indices for random data split
    splits_idx = np.append([0], np.cumsum(n_samples_per_condition))
    slices = [slice(splits_idx[k], splits_idx[k+1])
                                                for k in range(len(X))]

    # Step 2: If we have some clusters, repeat process on permutated data
    # -------------------------------------------------------------------
    if len(clusters) > 0:
        cluster_stats = [np.sum(T_obs[c]) for c in clusters]
        cluster_pv = np.ones(len(clusters), dtype=np.float)
        H0 = np.zeros(n_permutations) # histogram
        for i_s in range(n_permutations):
            np.random.shuffle(X_full)
            X_shuffle_list = [X_full[s] for s in slices]
            T_obs_surr = stat_fun(*X_shuffle_list)
            clusters_perm = _find_clusters(T_obs_surr, threshold, tail)

            if len(clusters_perm) > 0:
                cluster_stats_perm = [np.sum(T_obs_surr[c])
                                              for c in clusters_perm]
                H0[i_s] = max(cluster_stats_perm)
            else:
                H0[i_s] = 0

        # for each cluster in original data, calculate p-value as percentile
        # of its cluster statistics within all cluster statistics in surrogate
        # data
        cluster_pv[:] = [percentileofscore(H0, cluster_stats[i_cl])
                                             for i_cl in range(len(clusters))]
        cluster_pv[:] = (100.0 - cluster_pv[:]) / 100.0 # from pct to fraction
        return T_obs, clusters, cluster_pv, H0
    else:
        return T_obs, np.array([]), np.array([]), np.array([])


permutation_1d_cluster_test.__test__ = False

if __name__ == "__main__":
    noiselevel = 30

    normfactor = np.hanning(20).sum()

    condition1 = np.random.randn(50, 500) * noiselevel
    for i in range(50):
        condition1[i] = np.convolve(condition1[i], np.hanning(20),
                                      mode="same") / normfactor

    condition2 = np.random.randn(43, 500) * noiselevel
    for i in range(43):
        condition2[i] = np.convolve(condition2[i], np.hanning(20),
                                      mode="same") / normfactor

    pseudoekp = 5 * np.hanning(150)[None,:]
    condition1[:, 100:250] += pseudoekp
    condition2[:, 100:250] -= pseudoekp

    fs, cluster_times, cluster_p_values, histogram = permutation_1d_cluster_test(
                                [condition1, condition2], n_permutations=1000)

    # # Plotting for a better understanding
    # import pylab as pl
    # pl.close('all')
    # pl.subplot(211)
    # pl.plot(condition1.mean(axis=0), label="Condition 1")
    # pl.plot(condition2.mean(axis=0), label="Condition 2")
    # pl.ylabel("signal [a.u.]")
    # pl.subplot(212)
    # for i_c, c in enumerate(cluster_times):
    #     c = c[0]
    #     if cluster_p_values[i_c] <= 0.05:
    #         h = pl.axvspan(c.start, c.stop-1, color='r', alpha=0.3)
    #     else:
    #         pl.axvspan(c.start, c.stop-1, color=(0.3, 0.3, 0.3), alpha=0.3)
    # hf = pl.plot(fs, 'g')
    # pl.legend((h, ), ('cluster p-value < 0.05', ))
    # pl.xlabel("time (ms)")
    # pl.ylabel("f-values")
    # pl.show()

