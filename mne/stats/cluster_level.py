#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Thorsten Kranz <thorstenkranz@gmail.com>
#         Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import numpy as np
from scipy import stats, sparse, ndimage

from .parametric import f_oneway
from ..parallel import parallel_func


def _get_components(x_in, connectivity):
    """get connected components from a mask and a connectivity matrix"""
    try:
        from sklearn.utils._csgraph import cs_graph_components
    except:
        from scikits.learn.utils._csgraph import cs_graph_components

    mask = np.logical_and(x_in[connectivity.row], x_in[connectivity.col])
    data = connectivity.data[mask]
    row = connectivity.row[mask]
    col = connectivity.col[mask]
    shape = connectivity.shape
    idx = np.where(x_in)[0]
    row = np.concatenate((row, idx))
    col = np.concatenate((col, idx))
    data = np.concatenate((data, np.ones(len(idx), dtype=data.dtype)))
    connectivity = sparse.coo_matrix((data, (row, col)), shape=shape)
    _, components = cs_graph_components(connectivity)
    # print "-- number of components : %d" % np.unique(components).size
    return components


def _find_clusters(x, threshold, tail=0, connectivity=None):
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
    connectivity : sparse matrix in COO format
        Defines connectivity between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Defaut is None, i.e, no connectivity.

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

    if connectivity is None:
        labels, n_labels = ndimage.label(x_in)

        if x.ndim == 1:
            clusters = ndimage.find_objects(labels, n_labels)
            if len(clusters) == 0:
                sums = []
            else:
                sums = ndimage.measurements.sum(x, labels,
                                                index=range(1, n_labels + 1))
        else:
            clusters = list()
            sums = np.empty(n_labels)
            for l in range(1, n_labels + 1):
                c = labels == l
                clusters.append(c)
                sums[l - 1] = np.sum(x[c])
    else:
        if x.ndim > 1:
            raise Exception("Data should be 1D when using a connectivity "
                            "to define clusters.")
        if np.sum(x_in) == 0:
            return [], np.empty(0)
        components = _get_components(x_in, connectivity)
        labels = np.unique(components)
        clusters = list()
        sums = list()
        for l in labels:
            c = (components == l)
            if np.any(x_in[c]):
                clusters.append(c)
                sums.append(np.sum(x[c]))
        sums = np.array(sums)
    return clusters, np.atleast_1d(sums)


def _pval_from_histogram(T, H0, tail):
    """Get p-values from stats values given an H0 distribution

    For each stat compute a p-value as percentile of its statistics
    within all statistics in surrogate data
    """
    if not tail in [-1, 0, 1]:
        raise ValueError('invalid tail parameter')

    # from pct to fraction
    if tail == -1:  # up tail
        pval = np.array([np.sum(H0 <= t) for t in T])
    elif tail == 1:  # low tail
        pval = np.array([np.sum(H0 >= t) for t in T])
    elif tail == 0:  # both tails
        pval = np.array([np.sum(H0 >= abs(t)) for t in T])
        pval += np.array([np.sum(H0 <= -abs(t)) for t in T])

    pval = (pval + 1.0) / (H0.size + 1.0)  # the init data is one resampling
    return pval


def _one_permutation(X_full, slices, stat_fun, tail, threshold, connectivity,
                     rng):
    rng.shuffle(X_full)
    X_shuffle_list = [X_full[s] for s in slices]
    T_obs_surr = stat_fun(*X_shuffle_list)
    _, perm_clusters_sums = _find_clusters(T_obs_surr, threshold, tail,
                                           connectivity)

    if len(perm_clusters_sums) > 0:
        return np.max(perm_clusters_sums)
    else:
        return 0


def permutation_cluster_test(X, stat_fun=f_oneway, threshold=1.67,
                             n_permutations=1000, tail=0,
                             connectivity=None, n_jobs=1,
                             verbose=5, seed=None):
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
    connectivity : sparse matrix.
        Defines connectivity between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Defaut is None, i.e, no connectivity.
    verbose : int
        If > 0, print some text during computation.
    n_jobs : int
        Number of permutations to run in parallel (requires joblib package.)
    seed : int or None
        Seed the random number generator for results reproducibility.

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

    if connectivity is not None:
        connectivity = connectivity.tocoo()

    # Step 1: Calculate Anova (or other stat_fun) for original data
    # -------------------------------------------------------------
    T_obs = stat_fun(*X)

    clusters, cluster_stats = _find_clusters(T_obs, threshold, tail,
                                             connectivity)

    # make list of indices for random data split
    splits_idx = np.append([0], np.cumsum(n_samples_per_condition))
    slices = [slice(splits_idx[k], splits_idx[k + 1])
                                                for k in range(len(X))]

    parallel, my_one_permutation, _ = parallel_func(_one_permutation, n_jobs,
                                                 verbose)

    # Step 2: If we have some clusters, repeat process on permuted data
    # -------------------------------------------------------------------
    if len(clusters) > 0:
        if seed is None:
            seeds = [None] * n_permutations
        else:
            seeds = seed + np.arange(n_permutations)
        H0 = parallel(my_one_permutation(X_full, slices, stat_fun, tail,
                            threshold, connectivity, np.random.RandomState(s))
                                for s in seeds)
        H0 = np.array(H0)
        cluster_pv = _pval_from_histogram(cluster_stats, H0, tail)
        return T_obs, clusters, cluster_pv, H0
    else:
        return T_obs, np.array([]), np.array([]), np.array([])


permutation_cluster_test.__test__ = False


def ttest_1samp(X):
    """Returns T-values
    """
    T, _ = stats.ttest_1samp(X, 0)
    return T


def _one_1samp_permutation(n_samples, shape_ones, X_copy, threshold, tail,
                           connectivity, stat_fun, rng):
    # new surrogate data with random sign flip
    signs = np.sign(0.5 - rng.rand(n_samples, *shape_ones))
    X_copy *= signs

    # Recompute statistic on randomized data
    T_obs_surr = stat_fun(X_copy)
    _, perm_clusters_sums = _find_clusters(T_obs_surr, threshold, tail,
                                           connectivity)

    if len(perm_clusters_sums) > 0:
        idx_max = np.argmax(np.abs(perm_clusters_sums))
        return perm_clusters_sums[idx_max]  # get max with sign info
    else:
        return 0.0


def permutation_cluster_1samp_test(X, threshold=1.67, n_permutations=1000,
                                   tail=0, stat_fun=ttest_1samp,
                                   connectivity=None, n_jobs=1,
                                   verbose=5, seed=None):
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
    connectivity : sparse matrix.
        Defines connectivity between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Defaut is None, i.e, no connectivity.
    verbose : int
        If > 0, print some text during computation.
    n_jobs : int
        Number of permutations to run in parallel (requires joblib package.)
    seed : int or None
        Seed the random number generator for results reproducibility.


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

    if connectivity is not None:
        connectivity = connectivity.tocoo()

    # Step 1: Calculate T-stat for original data
    # -------------------------------------------------------------
    T_obs = stat_fun(X)

    clusters, cluster_stats = _find_clusters(T_obs, threshold, tail,
                                             connectivity)

    parallel, my_one_1samp_permutation, _ = parallel_func(
                                _one_1samp_permutation, n_jobs, verbose)

    # Step 2: If we have some clusters, repeat process on permuted data
    # -------------------------------------------------------------------
    if len(clusters) > 0:
        if seed is None:
            seeds = [None] * n_permutations
        else:
            seeds = seed + np.arange(n_permutations)
        H0 = parallel(my_one_1samp_permutation(n_samples, shape_ones, X_copy,
                                    threshold, tail, connectivity, stat_fun,
                                    np.random.RandomState(s))
                                    for s in seeds)
        H0 = np.array(H0)
        cluster_pv = _pval_from_histogram(cluster_stats, H0, tail)

        return T_obs, clusters, cluster_pv, H0
    else:
        return T_obs, np.array([]), np.array([]), np.array([])


permutation_cluster_1samp_test.__test__ = False
