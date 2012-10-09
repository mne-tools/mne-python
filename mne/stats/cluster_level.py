#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Thorsten Kranz <thorstenkranz@gmail.com>
#         Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#         Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import numpy as np
from scipy import stats, sparse, ndimage

from .parametric import f_oneway
from ..parallel import parallel_func


def _get_clusters_st(x_in, neighbors, max_tstep=1, use_box=False):
    """Directly calculate connectivity based on knowledge that time points are
    only connected to adjacent neighbors for data organized as time x space.

    This algorithm time increases linearly with the number of time points,
    compared to with the square for the standard (graph) algorithm.

    Note that it's possible an even faster algorithm could create clusters for
    each time point (using the standard graph method), then combine these
    clusters across time points in some reasonable way. This could then be used
    to extend to time x space x frequency datasets, for example. This has not
    been implemented yet."""
    n_tot = x_in.size
    n_vertices = len(neighbors)
    n_times = n_tot / float(n_vertices)
    if not n_times == int(n_times):
        raise ValueError('x_in.size must be multiple of connectivity.shape[0]')
    n_times = int(n_times)
    orig_nos = np.where(x_in)[0]
    t = orig_nos / n_vertices
    s = orig_nos % n_vertices

    tborder = np.zeros((n_times + 1, 1), dtype=int)
    for ii in range(n_times):
        temp = np.where(np.less_equal(t, ii))[0]
        if temp.size > 0:
            tborder[ii + 1] = temp[-1] + 1
        else:
            tborder[ii + 1] = tborder[ii]

    r = np.ones(t.shape, dtype=bool)
    clusters = list()
    next_ind = np.array([0])
    if s.size > 0:
        while next_ind.size > 0:
            # put first point in a cluster, adjust remaining
            t_inds = np.array([next_ind[0]])
            r[next_ind[0]] = False
            icount = 1  # count of nodes in the current cluster
            # look for significant values at the next time point,
            # same sensor, not placed yet, and add those
            while icount <= t_inds.size:
                ind = t_inds[icount - 1]
                bud1 = np.arange(tborder[max(t[ind] - max_tstep, 0)],
                                 tborder[min(t[ind] + max_tstep + 1, n_times)])
                if use_box:
                    # look at previous and next time points (using max_tstep)
                    # for all neighboring vertices
                    bud1 = bud1[r[bud1]]
                    if bud1.size > 0:
                        bud1 = bud1[np.in1d(s[bud1], neighbors[s[ind]],
                                            assume_unique=True)]
                        t_inds = np.concatenate((t_inds, bud1))
                        r[bud1] = False
                else:
                    sel1 = bud1[r[bud1]]
                    sel1 = sel1[np.equal(s[ind], s[sel1])]
                    # look at current time point across other vertices
                    bud1 = np.arange(tborder[t[ind]], tborder[t[ind] + 1])
                    bud1 = bud1[r[bud1]]
                    if bud1.size > 0:
                        bud1 = bud1[np.in1d(s[bud1], neighbors[s[ind]],
                                            assume_unique=True)]
                        buddies = np.concatenate((sel1, bud1))
                    else:
                        buddies = sel1
                    t_inds = np.concatenate((t_inds, buddies))
                    r[buddies] = False
                icount += 1
            next_ind = np.where(r)[0]
            clust = np.zeros((n_tot), dtype=bool)
            clust[orig_nos[t_inds]] = True
            clusters.append(clust)

    return clusters


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


def _find_clusters(x, threshold, tail=0, connectivity=None, by_sign=True,
                   max_tstep=1):
    """For a given 1d-array (test statistic), find all clusters which
    are above/below a certain threshold. Returns a list of 2-tuples.

    Parameters
    ----------
    x: 1D array
        Data
    threshold: float
        Where to threshold the statistic. Should be negative for tail == -1,
        and positive for tail == 0 or 1.
    tail : -1 | 0 | 1
        Type of comparison
    connectivity : sparse matrix in COO format, None, or list
        Defines connectivity between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        If connectivity is a list, it is assumed that each entry stores the
        indices of the spatial neighbors in a spatio-temporal dataset x.
        Defaut is None, i.e, no connectivity.
    by_sign : bool
        When doing a two-tailed test (tail == 0), if True only points with
        the same sign will be clustered together. This value is ignored for
        one-tailed tests.
    max_tstep : int
        If connectivity is a list, this defines the maximal number of time
        steps permitted for elements to be considered temporal neighbors.

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
        clusters, sums = _find_clusters_1dir(x, x_in, connectivity, max_tstep)
    elif tail == 1:
        x_in = x > threshold
        clusters, sums = _find_clusters_1dir(x, x_in, connectivity, max_tstep)
    else:
        if not by_sign:
            x_in = np.abs(x) > threshold
            clusters, sums = _find_clusters_1dir(x, x_in, connectivity,
                                                 max_tstep)
        else:
            x_in = x > threshold
            clusters_pos, sums_pos = _find_clusters_1dir(x, x_in, connectivity,
                                                         max_tstep)
            x_in = x < -threshold
            clusters_neg, sums_neg = _find_clusters_1dir(x, x_in, connectivity,
                                                         max_tstep)
            clusters = clusters_pos + clusters_neg
            sums = np.concatenate((sums_pos, sums_neg))

    return clusters, sums


def _find_clusters_1dir(x, x_in, connectivity, max_tstep):
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

        if isinstance(connectivity, sparse.spmatrix):
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
        elif isinstance(connectivity, list):  # use temporal adjacency
            clusters = _get_clusters_st(x_in, connectivity, max_tstep)
            sums = np.array([np.sum(x[c]) for c in clusters])
        else:
            raise ValueError('Connectivity must be a sparse matrix or list')

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
    else:  # both tails
        pval = np.array([np.sum(abs(H0) >= abs(t)) for t in T])

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
        # XXX need to add code to make it a full perm test when possible
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


def ttest_1samp_no_p(X):
    """t-test with no p-value calculation
    Returns T-values

    Notes
    -----
    One can use the conversion:
        threshold = -stats.distributions.t.ppf(p_thresh, n_samples)
    to converting a desired p-value threshold to t-value threshold

    that for two-tailed tests, p_thresh should be divided by 2"""
    return np.mean(X, axis=0) \
        / np.sqrt(np.var(X, axis=0, ddof=1) / X.shape[0])


def _one_1samp_permutation(n_samples, shape_ones, X_copy, threshold, tail,
                           connectivity, stat_fun, max_tstep, rng):
    if isinstance(rng, np.random.mtrand.RandomState):
        # new surrogate data with random sign flip
        signs = np.sign(0.5 - rng.rand(n_samples, *shape_ones))
    elif isinstance(rng, np.ndarray):
        # new surrogate data with specified sign flip
        if not rng.size == n_samples:
            raise ValueError('rng string must be n_samples long')
        signs = 2 * rng[:, None].astype(int) - 1
        if not np.all(np.equal(np.abs(signs), 1)):
            raise ValueError('signs from rng must be +/- 1')
    else:
        raise ValueError('rng must be a RandomState or str')
    X_copy *= signs

    # Recompute statistic on randomized data
    T_obs_surr = stat_fun(X_copy)
    _, perm_clusters_sums = _find_clusters(x=T_obs_surr, threshold=threshold,
                                           tail=tail, max_tstep=max_tstep,
                                           connectivity=connectivity)

    if len(perm_clusters_sums) > 0:
        idx_max = np.argmax(np.abs(perm_clusters_sums))
        return perm_clusters_sums[idx_max]  # get max with sign info
    else:
        return 0.0


def permutation_cluster_1samp_test(X, threshold=1.67, n_permutations=1024,
                                   tail=0, stat_fun=ttest_1samp_no_p,
                                   connectivity=None, verbose=5, n_jobs=1,
                                   seed=None, max_tstep=1):
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
    stat_fun : function
        Function used to compute the statistical map
    connectivity : sparse matrix or None
        Defines connectivity between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        This matrix must be square with dimension (n_vertices * n_times) or
        (n_vertices). Defaut is None, i.e, no connectivity. Use square
        n_vertices matrix for datasets with a large temporal extent to save on
        memory and computation time.
    verbose : int
        If > 0, print some text during computation.
    n_jobs : int
        Number of permutations to run in parallel (requires joblib package.)
    seed : int or None
        Seed the random number generator for results reproducibility.
        Note that if n_permutations >= 2^(n_samples) [or (2^(n_samples-1)) for
        two-tailed tests], this value will be ignored since an exact test
        (full permutation test) will be performed.
    max_tstep : int
        When connectivity is a n_vertices x n_vertices matrix, specify the
        maximum number of time steps between vertices to be considered
        neighbors. This is not used for full or None connectivity matrices.


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
        if connectivity.shape[0] == X.shape[1]:  # use global algorithm
            connectivity = connectivity.tocoo()
        else:  # use temporal adjacency algorithm
            n_times = X.shape[1] / float(connectivity.shape[0])
            if not round(n_times) == n_times:
                raise ValueError('connectivity must be of the correct size')
            # we claim to only use upper triangular part... not true here
            connectivity = (connectivity + connectivity.transpose()).tocsr()
            connectivity = [connectivity.indices[connectivity.indptr[i]:
                            connectivity.indptr[i+1]] for i in
                            range(len(connectivity.indptr)-1)]

    # Step 1: Calculate T-stat for original data
    # -------------------------------------------------------------
    T_obs = stat_fun(X)

    clusters, cluster_stats = _find_clusters(T_obs, threshold, tail,
                                             connectivity, max_tstep=max_tstep)

    parallel, my_one_1samp_permutation, _ = parallel_func(
                                _one_1samp_permutation, n_jobs, verbose)

    # Step 2: If we have some clusters, repeat process on permuted data
    # -------------------------------------------------------------------
    if len(clusters) > 0:
        # check to see if we can do an exact test
        # note for a two-tailed test, we can exploit symmetry to just do half
        max_perms = 2 ** (n_samples - (tail == 0))
        if max_perms <= n_permutations:
            # omit first perm b/c accounted for in _pval_from_histogram,
            # convert to binary array representation
            seeds = [np.fromiter(np.binary_repr(s, n_samples), dtype=int)
                     for s in range(1, max_perms)]
        else:
            if seed is None:
                seeds = [None] * n_permutations
            else:
                seeds = seed + np.arange(n_permutations)
            seeds = [np.random.RandomState(s) for s in seeds]

        H0 = parallel(my_one_1samp_permutation(n_samples, shape_ones, X_copy,
                                               threshold, tail, connectivity,
                                               stat_fun, max_tstep,
                                               s) for s in seeds)
        H0 = np.array(H0)
        cluster_pv = _pval_from_histogram(cluster_stats, H0, tail)

        return T_obs, clusters, cluster_pv, H0
    else:
        return T_obs, np.array([]), np.array([]), np.array([])


permutation_cluster_1samp_test.__test__ = False


def spatio_temporal_cluster_test(X, threshold=None, n_permutations=1024,
                                 tail=0, stat_fun=ttest_1samp_no_p,
                                 connectivity=None, verbose=5, n_jobs=1,
                                 seed=None, max_tstep=1, partitions=None):
    """Non-parametric cluster-level 1 sample T-test for spatio-temporal data

    This function provides a convenient wrapper for data organized in the form
    observations x space x time) to use permutation_cluster_1samp_test.

    Parameters
    ----------
    X: array
        Array of shape observations x time x vertices.
    threshold: float, or None
        If threshold is None, it will choose a t-threshold equivalent to
        p < 0.05 for the given number of (within-subject) observations.
    n_permutations: int
        See permutation_cluster_1samp_test.
    tail : -1 or 0 or 1 (default = 0)
        See permutation_cluster_1samp_test.
    stat_fun : function
        See permutation_cluster_1samp_test.
    connectivity : sparse matrix or None
        See permutation_cluster_1samp_test.
    verbose : int
        See permutation_cluster_1samp_test.
    n_jobs : int
        See permutation_cluster_1samp_test.
    seed : int or None
        See permutation_cluster_1samp_test.
    max_tstep : int
        See permutation_cluster_1samp_test.

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
    n_samples, n_times, n_vertices = X.shape

    if stat_fun is None:
        stat_fun = ttest_1samp_no_p

    if threshold is None:
        p_thresh = 0.05 / (1 + (tail == 0))
        threshold = -stats.distributions.t.ppf(p_thresh, n_samples)
        if np.sign(tail) < 0:
            threshold = -threshold

    # make it contiguous
    X = np.ascontiguousarray(X.reshape(n_samples, -1))

    # do the heavy lifting
    out = permutation_cluster_1samp_test(X, threshold=threshold,
              stat_fun=stat_fun, tail=tail, n_permutations=n_permutations,
              connectivity=connectivity, n_jobs=n_jobs, seed=seed,
              max_tstep=max_tstep, verbose=verbose)
    return out