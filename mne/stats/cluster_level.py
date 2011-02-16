#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Thorsten Kranz <thorstenkranz@gmail.com>
#         Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import numpy as np
from scipy import stats
from scipy.stats import percentileofscore


def f_oneway(*args):
    """Call scipy.stats.f_oneway, but return only f-value"""
    return stats.f_oneway(*args)[0]


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

    x = np.concatenate([np.array([threshold]), x, np.array([threshold])])
    if tail == -1:
        x_in = (x < threshold).astype(np.int)
    elif tail == 1:
        x_in = (x > threshold).astype(np.int)
    else:
        x_in = (np.abs(x) > threshold).astype(np.int)

    x_switch = np.diff(x_in)
    in_points = np.where(x_switch > 0)[0]
    out_points = np.where(x_switch < 0)[0]
    clusters = zip(in_points, out_points)
    return clusters


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

    splits_idx = np.cumsum(n_samples_per_condition)[:-1]
    # Step 2: If we have some clusters, repeat process on permutated data
    # -------------------------------------------------------------------
    if len(clusters) > 0:
        cluster_stats = [np.sum(T_obs[c[0]:c[1]]) for c in clusters]
        cluster_pv = np.ones(len(clusters), dtype=np.float)
        H0 = np.zeros(n_permutations) # histogram
        for i_s in range(n_permutations):
            # make list of indices for random data split
            indices_lists = np.split(np.random.permutation(n_samples_total),
                                     splits_idx)

            X_shuffle_list = [X_full[indices] for indices in indices_lists]
            T_obs_surr = stat_fun(*X_shuffle_list)
            clusters_perm = _find_clusters(T_obs_surr, threshold, tail)

            if len(clusters_perm) > 0:
                cluster_stats_perm = [np.sum(T_obs_surr[c[0]:c[1]])
                                      for c in clusters_perm]
                H0[i_s] = max(cluster_stats_perm)
            else:
                H0[i_s] = 0

        # for each cluster in original data, calculate p-value as percentile
        # of its cluster statistics within all cluster statistics in surrogate
        # data
        cluster_pv[:] = [percentileofscore(H0,
                                           cluster_stats[i_cl])
                          for i_cl in range(len(clusters))]
        cluster_pv[:] = (100.0 - cluster_pv[:]) / 100.0 # from pct to fraction
        return T_obs, clusters, cluster_pv, H0
    else:
        return T_obs, np.array([]), np.array([]), np.array([])


permutation_1d_cluster_test.__test__ = False
