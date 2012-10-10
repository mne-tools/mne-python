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
    n_vertices = len(neighbors)
    n_tot = x_in.size
    n_times, junk = divmod(n_tot, n_vertices)
    if not junk == 0:
        raise ValueError('x_in.size must be multiple of connectivity.shape[0]')
    v = np.where(x_in)[0]
    t, s = divmod(v, n_vertices)

    t_border = np.zeros((n_times + 1, 1), dtype=int)
    for ii in range(n_times):
        temp = np.where(np.less_equal(t, ii))[0]
        if temp.size > 0:
            t_border[ii + 1] = temp[-1] + 1
        else:
            t_border[ii + 1] = t_border[ii]

    r = np.ones(t.shape, dtype=bool)
    clusters = list()
    next_ind = 0
    inds = np.arange(t_border[0], t_border[n_times])
    if s.size > 0:
        while next_ind is not None:
            # put first point in a cluster, adjust remaining
            t_inds = [next_ind]
            r[next_ind] = False
            icount = 1  # count of nodes in the current cluster
            # look for significant values at the next time point,
            # same sensor, not placed yet, and add those
            while icount <= len(t_inds):
                ind = t_inds[icount - 1]
                buddies = inds[t_border[max(t[ind] - max_tstep, 0)]: \
                               t_border[min(t[ind] + max_tstep + 1, n_times)]]
                if use_box:
                    # look at previous and next time points (using max_tstep)
                    # for all neighboring vertices
                    buddies = buddies[r[buddies]]
                    buddies = buddies[np.in1d(s[buddies], neighbors[s[ind]],
                                              assume_unique=True)]
                else:
                    selves = buddies[r[buddies]]
                    selves = selves[s[ind] == s[selves]]
                    # look at current time point across other vertices
                    buddies = inds[t_border[t[ind]]:t_border[t[ind] + 1]]
                    buddies = buddies[r[buddies]]
                    buddies = buddies[np.in1d(s[buddies], neighbors[s[ind]],
                                              assume_unique=True)]
                    buddies = np.concatenate((selves, buddies))
                t_inds += buddies.tolist()
                r[buddies] = False
                icount += 1
            # this is equivalent to np.where(r)[0] for these purposes, but it's
            # a little bit faster. Unfortunately there's no way to tell numpy
            # just to find the first instance (to save checking every one):
            next_ind = np.argmax(r)
            if next_ind == 0:
                next_ind = None
            clusters.append(v[t_inds])

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
    labels = np.unique(components)
    clusters = list()
    for l in labels:
        c = np.where(components == l)[0]
        if np.any(x_in[c]):
            clusters.append(c)
    # print "-- number of components : %d" % np.unique(components).size
    return clusters


def _find_clusters(x, threshold, tail=0, connectivity=None, by_sign=True,
                   max_tstep=1, include=None, partitions=None):
    """For a given 1d-array (test statistic), find all clusters which
    are above/below a certain threshold. Returns a list of 2-tuples.

    Parameters
    ----------
    x : 1D array
        Data
    threshold : float
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
    include : 1D bool array or None
        Mask to apply to the data of points to cluster. If None, all points
        are used.
    partitions : array of int or None
        An array (same size as X) of integers indicating which points belong
        to each partition.

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

    clusters = list()
    sums = list()
    if tail == 0:
        if not by_sign:
            if include is None:
                x_in = np.abs(x) > threshold
            else:
                x_in = np.logical_and(np.abs(x) > threshold, include)

            out = _find_clusters_1dir_parts(x, x_in, connectivity, max_tstep,
                                            partitions)
            clusters += out[0]
            sums.append(out[1])
        else:
            if include is None:
                x_in = x > threshold
            else:
                x_in = np.logical_and(x > threshold, include)

            out = _find_clusters_1dir_parts(x, x_in, connectivity, max_tstep,
                                            partitions)
            clusters += out[0]
            sums.append(out[1])

            if include is None:
                x_in = x < -threshold
            else:
                x_in = np.logical_and(x < -threshold, include)

            out = _find_clusters_1dir_parts(x, x_in, connectivity, max_tstep,
                                            partitions)
            clusters += out[0]
            sums.append(out[1])
    else:
        if tail == -1:
            if include is None:
                x_in = x < threshold
            else:
                x_in = np.logical_and(x < threshold, include)
        else:  # tail == 1
            if include is None:
                x_in = x > threshold
            else:
                x_in = np.logical_and(x > threshold, include)

        out = _find_clusters_1dir_parts(x, x_in, connectivity, max_tstep,
                                        partitions)
        clusters += out[0]
        sums.append(out[1])

    sums = np.concatenate(sums)
    return clusters, sums


def _find_clusters_1dir_parts(x, x_in, connectivity, max_tstep, partitions):
    """Deal with partitions, and pass the work to _find_clusters_1dir
    """
    if partitions is None:
        clusters, sums = _find_clusters_1dir(x, x_in, connectivity, max_tstep)
    else:
        # cluster each partition separately
        clusters = list()
        sums = list()
        for p in range(np.max(partitions) + 1):
            x_i = np.logical_and(x_in, partitions == p)
            out = _find_clusters_1dir(x, x_i, connectivity, max_tstep)
            clusters += out[0]
            sums.append(out[1])
        sums = np.concatenate(sums)
    return clusters, sums


def _find_clusters_1dir(x, x_in, connectivity, max_tstep):
    """Actually call the clustering algorithm"""
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
            clusters = _get_components(x_in, connectivity)
        elif isinstance(connectivity, list):  # use temporal adjacency
            clusters = _get_clusters_st(x_in, connectivity, max_tstep)
        else:
            raise ValueError('Connectivity must be a sparse matrix or list')
        sums = np.array([np.sum(x[c]) for c in clusters])

    return clusters, np.atleast_1d(sums)


def _clusters_to_bool(components, n_tot):
    """Convert to the old format of clusters, which were bool arrays"""
    for ci, c in enumerate(components):
        components[ci] = np.zeros((n_tot), dtype=bool)
        components[ci][c] = True
    return components


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
    # convert clusters to old format
    if connectivity is not None:
        clusters = _clusters_to_bool(clusters, X.shape[1])

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

        threshold = -scipy.stats.distributions.t.ppf(p_thresh, n_samples)

    to convert a desired p-value threshold to t-value threshold. Don't forget
    that for two-tailed tests, p_thresh in the above should be divided by 2
    """
    return np.mean(X, axis=0) \
        / np.sqrt(np.var(X, axis=0, ddof=1) / X.shape[0])


def _one_1samp_permutation(n_samples, shape_ones, X_copy, threshold, tail,
                           connectivity, stat_fun, max_tstep, include,
                           partitions, rng):
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
                                           connectivity=connectivity,
                                           partitions=partitions,
                                           include=include)

    if len(perm_clusters_sums) > 0:
        idx_max = np.argmax(np.abs(perm_clusters_sums))
        return perm_clusters_sums[idx_max]  # get max with sign info
    else:
        return 0.0


def permutation_cluster_1samp_test(X, threshold=1.67, n_permutations=1024,
                                   tail=0, stat_fun=ttest_1samp_no_p,
                                   connectivity=None, verbose=5, n_jobs=1,
                                   seed=None, max_tstep=1, partitions=None,
                                   exclude=None, step_down_p=0):
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
    partitions : array of int or None
        An array (same size as X) of integers indicating which points belong
        to each partition. If data can be broken up into disjoint sets
        (e.g., hemipsheres), this can speed up computation.
    exclude : boolean array or None
        Mask to apply to the data to exclude certain points from clustering
        (e.g., medial wall vertices). Should be the same shape as X. If None,
        no points are excluded.
    step_down_p : float
        To perform a step-down-in-jumps test, pass a p-value for clusters to
        exclude from each successive iteration. Default is zero, perform no
        step-down test (since no clusters will be smaller than this value).
        Setting this to a reasonable value, e.g. 0.05, can increase sensitivity
        but costs computation time.

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
                            connectivity.indptr[i + 1]] for i in
                            range(len(connectivity.indptr) - 1)]

    # do some checks
    if (partitions is not None) and not partitions.size == X.shape[1]:
        raise ValueError('partitions must be the same shape as X[1]')
    if (exclude is not None) and not exclude.size == X.shape[1]:
        raise ValueError('exclude must be the same shape as X[1]')

    # Step 1: Calculate T-stat for original data
    # -------------------------------------------------------------
    T_obs = stat_fun(X)
    if exclude is not None:
        include = np.logical_not(exclude)
    else:
        include = None

    clusters, cluster_stats = _find_clusters(T_obs, threshold, tail,
                                             connectivity, max_tstep=max_tstep,
                                             include=include,
                                             partitions=partitions)
    # convert clusters to old format
    if connectivity is not None:
        clusters = _clusters_to_bool(clusters, X.shape[1])

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

        # Step 3: repeat permutations for stetp-down-in-jumps procedure
        smallest_p = -1
        clusters_kept = 0
        step_down_include = None  # start out including all points
        step_down_iteration = 0
        while smallest_p < step_down_p:
            # actually do the clustering for each partition
            if include is not None:
                if step_down_include is not None:
                    this_include = np.logical_and(include, step_down_include)
                else:
                    this_include = include
            else:
                this_include = step_down_include
            H0 = parallel(my_one_1samp_permutation(n_samples, shape_ones,
                                                   X_copy, threshold, tail,
                                                   connectivity, stat_fun,
                                                   max_tstep, this_include,
                                                   partitions, s)
                                                   for s in seeds)
            H0 = np.array(H0)
            cluster_pv = _pval_from_histogram(cluster_stats, H0, tail)

            # sort them by significance; for backward compat, don't sort the
            # clusters themselves
            inds = np.argsort(cluster_pv)
            ord_pv = cluster_pv[inds]
            smallest_p = ord_pv[clusters_kept]
            step_down_include = np.ones(X.shape[1], dtype=bool)
            under = np.where(cluster_pv < step_down_p)[0]
            for ci in under:
                step_down_include[clusters[ci]] = False
            step_down_iteration += 1
            if verbose > 0 and step_down_p > 0:
                extra_text = 'additional ' if step_down_iteration > 1 else ''
                new_count = under.size - clusters_kept
                plural = '' if new_count == 1 else 's'
                print 'Step-down-in-jumps iteration %i found %i %scluster%s'\
                    % (step_down_iteration, new_count, extra_text, plural)
            clusters_kept += under.size

        return T_obs, clusters, cluster_pv, H0
    else:
        return T_obs, np.array([]), np.array([]), np.array([])


permutation_cluster_1samp_test.__test__ = False


def spatio_temporal_cluster_test(X, threshold=None, n_permutations=1024,
                                 tail=0, stat_fun=ttest_1samp_no_p,
                                 connectivity=None, verbose=5, n_jobs=1,
                                 seed=None, max_tstep=1,
                                 spatial_partitions=None,
                                 spatial_exclude=None, step_down_p=0):
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
    partitions : list of int or None
        See permutation_cluster_1samp_test.
    spatial_partitions : list of int or None
        List of spatial indices that divide disjoint sets (e.g., hemispheres).
        For fsaverage (2 hemispheres @ 10242 vertices), this would be [10242].
    spatial_exclude : list of int or None
        List of spatial indices to exclude from clustering.
    step_down_p : float
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

    # convert spatial_exclude before passing on if necessary
    if spatial_exclude is not None:
        exclude = st_mask_from_s_inds(n_times, n_vertices,
                                              spatial_exclude, True)
    else:
        exclude = None

    # convert spatial partitions before passing on if necessary
    if spatial_partitions is not None:
        partitions = np.zeros((1, n_vertices), dtype=int)
        partitions[0, spatial_partitions] = 1
        partitions = partitions.cumsum() * np.ones((n_times, 1), dtype=int)
        partitions = partitions.ravel()
    else:
        partitions = None

    # make it contiguous
    X = np.ascontiguousarray(X.reshape(n_samples, -1))

    # do the heavy lifting
    out = permutation_cluster_1samp_test(X, threshold=threshold,
              stat_fun=stat_fun, tail=tail, n_permutations=n_permutations,
              connectivity=connectivity, n_jobs=n_jobs, seed=seed,
              max_tstep=max_tstep, verbose=verbose, partitions=partitions,
              exclude=exclude, step_down_p=step_down_p)
    return out


spatio_temporal_cluster_test.__test__ = False


def st_mask_from_s_inds(n_times, n_vertices, vertices, set_as=True):
    """This function returns a boolean mask vector to apply to a spatio-
    temporal connectivity matrix (n_times * n_vertices square) to include (or
    exclude) certain spatial coordinates. This is useful for excluding certain
    regions from analysis (e.g., medial wall vertices).

    Parameters
    ----------
    n_times : int
        Number of time points
    n_vertices : int
        Number of spatial points
    vertices : list or array of int
        Vertex numbers to set
    set_as : bool
        If True, all points except "vertices" are set to False (inclusion).
        If False, all points except "vertices" are set to True (exclusion).

    Returns
    -------
    mask : array of bool
        A (n_times * n_vertices) array of boolean values for masking
    """
    mask = np.zeros((n_times, n_vertices), dtype=bool)
    mask[:, vertices] = True
    mask = mask.ravel()
    if set_as is False:
        mask = np.logical_not(mask)
    return mask