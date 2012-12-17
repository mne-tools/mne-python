#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Thorsten Kranz <thorstenkranz@gmail.com>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import numpy as np
from scipy import stats, sparse, ndimage

import logging
logger = logging.getLogger('mne')

from .parametric import f_oneway
from ..parallel import parallel_func
from ..utils import split_list
from ..fixes import in1d, unravel_index
from .. import verbose


def _get_clusters_spatial(x_in, neighbors):
    """Helper function to form spatial clusters using neighbor lists

    This is equivalent to _get_components with n_times = 1, with a properly
    reconfigured connectivity matrix (formed as "neighbors" list)
    """
    if not x_in.size == len(neighbors):
        raise ValueError('x_in.size must be the same as len(neighbors)')
    s = np.where(x_in)[0]
    r = np.ones(s.shape, dtype=bool)
    clusters = list()
    next_ind = 0 if s.size > 0 else None
    while next_ind is not None:
        # put first point in a cluster, adjust remaining
        t_inds = [next_ind]
        r[next_ind] = False
        icount = 1  # count of nodes in the current cluster
        while icount <= len(t_inds):
            ind = t_inds[icount - 1]
            # look across other vertices
            buddies = np.where(r)[0]
            buddies = buddies[in1d(s[buddies], neighbors[s[ind]],
                                      assume_unique=True)]
            t_inds += buddies.tolist()
            r[buddies] = False
            icount += 1
        # this is equivalent to np.where(r)[0] for these purposes, but it's
        # a little bit faster. Unfortunately there's no way to tell numpy
        # just to find the first instance (to save checking every one):
        next_ind = np.argmax(r)
        if next_ind == 0:
            next_ind = None
        clusters.append(s[t_inds])
    return clusters


def _reassign(check, clusters, base, num):
    """Helper function to reassign cluster numbers"""
    # reconfigure check matrix
    check[check == num] = base
    # concatenate new values into clusters array
    clusters[base - 1] = np.concatenate((clusters[base - 1],
                                         clusters[num - 1]))
    clusters[num - 1] = np.array([], dtype=int)


def _get_clusters_st_1step(x_in, neighbors):
    """Directly calculate connectivity based on knowledge that time points are
    only connected to adjacent neighbors for data organized as time x space.

    This algorithm time increases linearly with the number of time points,
    compared to with the square for the standard (graph) algorithm.

    This algorithm creates clusters for each time point using a method more
    efficient than the standard graph method (but otherwise equivalent), then
    combines these clusters across time points in a reasonable way."""
    n_src = len(neighbors)
    n_times = int(x_in.shape[0] / n_src)
    orig_shape = x_in.shape
    x_in.shape = (n_times, n_src)

    # start cluster numbering at 1 for diffing convenience
    enum_offset = 1
    clusters = list()
    check = np.zeros(x_in.shape, dtype=int)
    for ii, xa in enumerate(x_in):
        c = _get_clusters_spatial(xa, neighbors)
        for ci, cl in enumerate(c):
            check[ii, cl] = ci + enum_offset
        enum_offset += len(c)
        # give them the correct offsets
        c = [cl + ii * n_src for cl in c]
        clusters += c

    # now that each cluster has been assigned a unique number, combine them
    diffs = np.logical_and(np.diff(check, axis=0) > 0, check[:-1] > 0)
    # go through each time point
    for check1, check2, d in zip(check[:-1], check[1:], diffs):
        # go through each one that needs reassignment
        n = check2[d]
        for num in np.unique(n):
            prevs = check1[d][num == n]
            base = np.min(prevs)
            for pr in np.unique(prevs[prevs != base]):
                _reassign(check1, clusters, base, pr)
            # reassign values
            _reassign(check2, clusters, base, num)
    # clean up clusters
    clusters = [cl for cl in clusters if len(cl) > 0]
    x_in.shape = orig_shape
    return clusters


def _get_clusters_st_multistep(x_in, neighbors, max_step=1):
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
                selves = inds[t_border[max(t[ind] - max_step, 0)]:
                              t_border[min(t[ind] + max_step + 1, n_times)]]
                selves = selves[r[selves]]
                selves = selves[s[ind] == s[selves]]

                # look at current time point across other vertices
                buddies = inds[t_border[t[ind]]:t_border[t[ind] + 1]]
                buddies = buddies[r[buddies]]
                buddies = buddies[in1d(s[buddies], neighbors[s[ind]],
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


def _get_clusters_st(x_in, neighbors, max_step=1):
    """Helper function to choose the most efficient version"""
    if max_step == 1:
        return _get_clusters_st_1step(x_in, neighbors)
    else:
        return _get_clusters_st_multistep(x_in, neighbors, max_step)


def _get_components(x_in, connectivity, return_list=True):
    """get connected components from a mask and a connectivity matrix"""
    try:
        from sklearn.utils._csgraph import cs_graph_components
    except:
        try:
            from scikits.learn.utils._csgraph import cs_graph_components
        except:
            # in theory we might be able to shoehorn this into using
            # _get_clusters_spatial if we transform connectivity into
            # a neighbor list, and it might end up being faster anyway,
            # but for now:
            raise ValueError('scikits-learn must be installed')

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
    if return_list:
        labels = np.unique(components)
        clusters = list()
        for l in labels:
            c = np.where(components == l)[0]
            if np.any(x_in[c]):
                clusters.append(c)
        # logger.info("-- number of components : %d"
        #             % np.unique(components).size)
        return clusters
    else:
        return components


def _find_clusters(x, threshold, tail=0, connectivity=None, by_sign=True,
                   max_step=1, include=None, partitions=None, t_power=1):
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
        Defaut is None, i.e, a regular lattice connectivity.
    by_sign : bool
        When doing a two-tailed test (tail == 0), if True only points with
        the same sign will be clustered together. This value is ignored for
        one-tailed tests.
    max_step : int
        If connectivity is a list, this defines the maximal number of steps
        between vertices along the second dimension (typically time) to be
        considered connected.
    include : 1D bool array or None
        Mask to apply to the data of points to cluster. If None, all points
        are used.
    partitions : array of int or None
        An array (same size as X) of integers indicating which points belong
        to each partition.
    t_power : float
        Power to raise the statistical values (usually t-values) by before
        summing (sign will be retained). Note that t_power == 0 will give a
        count of nodes in each cluster, t_power == 1 will weight each node by
        its statistical score.

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

            out = _find_clusters_1dir_parts(x, x_in, connectivity, max_step,
                                            partitions, t_power)
            clusters += out[0]
            sums.append(out[1])
        else:
            if include is None:
                x_in = x > threshold
            else:
                x_in = np.logical_and(x > threshold, include)

            out = _find_clusters_1dir_parts(x, x_in, connectivity, max_step,
                                            partitions, t_power)
            clusters += out[0]
            sums.append(out[1])

            if include is None:
                x_in = x < -threshold
            else:
                x_in = np.logical_and(x < -threshold, include)

            out = _find_clusters_1dir_parts(x, x_in, connectivity, max_step,
                                            partitions, t_power)
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

        out = _find_clusters_1dir_parts(x, x_in, connectivity, max_step,
                                        partitions, t_power)
        clusters += out[0]
        sums.append(out[1])

    sums = np.concatenate(sums)
    return clusters, sums


def _find_clusters_1dir_parts(x, x_in, connectivity, max_step, partitions,
                              t_power):
    """Deal with partitions, and pass the work to _find_clusters_1dir
    """
    if partitions is None:
        clusters, sums = _find_clusters_1dir(x, x_in, connectivity, max_step,
                                             t_power)
    else:
        # cluster each partition separately
        clusters = list()
        sums = list()
        for p in range(np.max(partitions) + 1):
            x_i = np.logical_and(x_in, partitions == p)
            out = _find_clusters_1dir(x, x_i, connectivity, max_step, t_power)
            clusters += out[0]
            sums.append(out[1])
        sums = np.concatenate(sums)
    return clusters, sums


def _find_clusters_1dir(x, x_in, connectivity, max_step, t_power):
    """Actually call the clustering algorithm"""
    if connectivity is None:
        labels, n_labels = ndimage.label(x_in)

        if x.ndim == 1:
            clusters = ndimage.find_objects(labels, n_labels)
            if len(clusters) == 0:
                sums = []
            else:
                if t_power == 1:
                    sums = ndimage.measurements.sum(x, labels,
                                                  index=range(1, n_labels + 1))
                else:
                    sums = ndimage.measurements.sum(np.sign(x) *
                                                  np.abs(x) ** t_power, labels,
                                                  index=range(1, n_labels + 1))
        else:
            clusters = list()
            sums = np.empty(n_labels)
            for l in range(1, n_labels + 1):
                c = labels == l
                clusters.append(c)
                if t_power == 1:
                    sums[l - 1] = np.sum(x[c])
                else:
                    sums[l - 1] = np.sum(np.sign(x[c]) *
                                         np.abs(x[c]) ** t_power)
    else:
        if x.ndim > 1:
            raise Exception("Data should be 1D when using a connectivity "
                            "to define clusters.")
        if np.sum(x_in) == 0:
            return [], np.empty(0)

        if isinstance(connectivity, sparse.spmatrix):
            clusters = _get_components(x_in, connectivity)
        elif isinstance(connectivity, list):  # use temporal adjacency
            clusters = _get_clusters_st(x_in, connectivity, max_step)
        else:
            raise ValueError('Connectivity must be a sparse matrix or list')
        if t_power == 1:
            sums = np.array([np.sum(x[c]) for c in clusters])
        else:
            sums = np.array([np.sum(np.sign(x[c]) * np.abs(x[c]) ** t_power)
                            for c in clusters])

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


def _do_permutations(X_full, slices, stat_fun, tail, threshold, connectivity,
                     seeds, sample_shape):

    n_samp = X_full.shape[0]

    # allocate space for output
    max_cluster_sums = np.empty(len(seeds), dtype=np.double)

    for seed_idx, seed in enumerate(seeds):
        # shuffle sample indices
        rng = np.random.RandomState(seed)
        idx_shuffled = np.arange(n_samp)
        rng.shuffle(idx_shuffled)
        idx_shuffle_list = [idx_shuffled[s] for s in slices]

        # shuffle all data at once
        X_shuffle_list = [X_full[idx, :] for idx in idx_shuffle_list]
        T_obs_surr = stat_fun(*X_shuffle_list)

        # The stat should have the same shape as the samples for no conn.
        if connectivity is None:
            T_obs_surr.shape = sample_shape

        _, perm_clusters_sums = _find_clusters(T_obs_surr, threshold, tail,
                                               connectivity)
        if len(perm_clusters_sums) > 0:
            max_cluster_sums[seed_idx] = np.max(perm_clusters_sums)
        else:
            max_cluster_sums[seed_idx] = 0

    return max_cluster_sums


@verbose
def permutation_cluster_test(X, stat_fun=f_oneway, threshold=1.67,
                             n_permutations=1000, tail=0,
                             connectivity=None, n_jobs=1,
                             verbose=None, seed=None, out_type='mask'):
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
        Defaut is None, i.e, a regular lattice connectivity.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    n_jobs : int
        Number of permutations to run in parallel (requires joblib package.)
    seed : int or None
        Seed the random number generator for results reproducibility.
    out_type : str
        For arrays with connectivity, this sets the output format for clusters.
        If 'mask', it will pass back a list of boolean mask arrays.
        If 'indices', it will pass back a list of lists, where each list is the
        set of vertices in a given cluster. Note that the latter may use far
        less memory for large datasets.

    Returns
    -------
    T_obs : array of shape [n_tests]
        T-statistic observerd for all variables
    clusters : list
        List type defined by out_type above.
    cluster_pv : array
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

    if not out_type in ['mask', 'indices']:
        raise ValueError('out_type must be either \'mask\' or \'indices\'')

    # flatten the last dimensions if data is high dimensional
    sample_shape = X[0].shape[1:]
    if X[0].ndim > 2:
        X = [np.reshape(x, (x.shape[0], -1)) for x in X]

    X_full = np.concatenate(X, axis=0)
    n_samples_per_condition = [x.shape[0] for x in X]

    if connectivity is not None:
        connectivity = connectivity.tocoo()

    # Step 1: Calculate Anova (or other stat_fun) for original data
    # -------------------------------------------------------------
    T_obs = stat_fun(*X)
    logger.info('stat_fun(H1): min=%f max=%f'
                % (np.min(T_obs), np.max(T_obs)))

    # The stat should have the same shape as the samples for no conn.
    if connectivity is None:
        T_obs.shape = sample_shape

    clusters, cluster_stats = _find_clusters(T_obs, threshold, tail,
                                             connectivity)
    logger.info('Found %d clusters' % len(clusters))

    # convert clusters to old format
    if connectivity is not None and out_type == 'mask':
        clusters = _clusters_to_bool(clusters, X[0].shape[1])

    # The clusters and stat should have the same shape as the samples
    clusters = _reshape_clusters(clusters, sample_shape)
    T_obs.shape = sample_shape

    # make list of indices for random data split
    splits_idx = np.append([0], np.cumsum(n_samples_per_condition))
    slices = [slice(splits_idx[k], splits_idx[k + 1])
                                                for k in range(len(X))]

    parallel, my_do_permutations, _ = parallel_func(_do_permutations, n_jobs)

    # Step 2: If we have some clusters, repeat process on permuted data
    # -------------------------------------------------------------------
    if len(clusters) > 0:
        # XXX need to add code to make it a full perm test when possible
        if seed is None:
            seeds = [None] * n_permutations
        else:
            seeds = list(seed + np.arange(n_permutations))
        H0 = parallel(my_do_permutations(X_full, slices, stat_fun, tail,
                            threshold, connectivity, s, sample_shape)
                            for s in split_list(seeds, n_jobs))
        H0 = np.concatenate(H0)
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

        threshold = -scipy.stats.distributions.t.ppf(p_thresh, n_samples - 1)

    to convert a desired p-value threshold to t-value threshold. Don't forget
    that for two-tailed tests, p_thresh in the above should be divided by 2
    """
    return np.mean(X, axis=0) \
        / np.sqrt(np.var(X, axis=0, ddof=1) / X.shape[0])


def _do_1samp_permutations(X, threshold, tail, connectivity, stat_fun,
                           max_step, include, partitions, t_power, seeds,
                           sample_shape):
    n_samp = X.shape[0]

    # allocate space for output
    max_cluster_sums = np.empty(len(seeds), dtype=np.double)

    for seed_idx, rng in enumerate(seeds):
        if isinstance(rng, np.random.mtrand.RandomState):
            # new surrogate data with random sign flip
            signs = np.sign(0.5 - rng.rand(n_samp))
            signs = signs[:, np.newaxis]
        elif isinstance(rng, np.ndarray):
            # new surrogate data with specified sign flip
            if not rng.size == n_samp:
                raise ValueError('rng string must be n_samples long')
            signs = 2 * rng[:, None].astype(int) - 1
            if not np.all(np.equal(np.abs(signs), 1)):
                raise ValueError('signs from rng must be +/- 1')
        else:
            raise ValueError('rng must be a RandomState or str')
        X *= signs

        # Recompute statistic on randomized data
        T_obs_surr = stat_fun(X)

        # Set X back to previous state (trade memory efficiency for CPU use)
        X *= signs

        # The stat should have the same shape as the samples for no conn.
        if connectivity is None:
            T_obs_surr.shape = sample_shape

        # Find cluster on randomized stats
        _, perm_clusters_sums = _find_clusters(T_obs_surr, threshold=threshold,
                                               tail=tail, max_step=max_step,
                                               connectivity=connectivity,
                                               partitions=partitions,
                                               include=include,
                                               t_power=t_power)
        if len(perm_clusters_sums) > 0:
            # get max with sign info
            idx_max = np.argmax(np.abs(perm_clusters_sums))
            max_cluster_sums[seed_idx] = perm_clusters_sums[idx_max]
        else:
            max_cluster_sums[seed_idx] = 0

    return max_cluster_sums


@verbose
def permutation_cluster_1samp_test(X, threshold=1.67, n_permutations=1024,
                                   tail=0, stat_fun=ttest_1samp_no_p,
                                   connectivity=None, verbose=None, n_jobs=1,
                                   seed=None, max_step=1, exclude=None,
                                   step_down_p=0, t_power=1, out_type='mask',
                                   check_disjoint=False):
    """Non-parametric cluster-level 1 sample T-test

    From a array of observations, e.g. signal amplitudes or power spectrum
    estimates etc., calculate if the observed mean significantly deviates
    from 0. The procedure uses a cluster analysis with permutation test
    for calculating corrected p-values. Randomized data are generated with
    random sign flips.

    Parameters
    ----------
    X: array, shape=(n_samples, n_variables)
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
        (n_vertices). Defaut is None, i.e, a regular lattice connectivity.
        Use square n_vertices matrix for datasets with a large temporal
        extent to save on memory and computation time.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    n_jobs : int
        Number of permutations to run in parallel (requires joblib package.)
    seed : int or None
        Seed the random number generator for results reproducibility.
        Note that if n_permutations >= 2^(n_samples) [or (2^(n_samples-1)) for
        two-tailed tests], this value will be ignored since an exact test
        (full permutation test) will be performed.
    max_step : int
        When connectivity is a n_vertices x n_vertices matrix, specify the
        maximum number of steps between vertices along the second dimension
        (typically time) to be considered connected. This is not used for full
        or None connectivity matrices.
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
    t_power : float
        Power to raise the statistical values (usually t-values) by before
        summing (sign will be retained). Note that t_power == 0 will give a
        count of nodes in each cluster, t_power == 1 will weight each node by
        its statistical score.
    out_type : str
        For arrays with connectivity, this sets the output format for clusters.
        If 'mask', it will pass back a list of boolean mask arrays.
        If 'indices', it will pass back a list of lists, where each list is the
        set of vertices in a given cluster. Note that the latter may use far
        less memory for large datasets.
    check_disjoint : bool
        If True, the connectivity matrix (or list) will be examined to
        determine of it can be separated into disjoint sets. In some cases
        (usually with connectivity as a list and many "time" points), this
        can lead to faster clustering, but results should be identical.

    Returns
    -------
    T_obs : array of shape [n_tests]
        T-statistic observerd for all variables
    clusters : list
        List type defined by out_type above.
    cluster_pv : array
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

    if not out_type in ['mask', 'indices']:
        raise ValueError('out_type must be either \'mask\' or \'indices\'')

    if X.ndim == 1:
        X = X[:, np.newaxis]
    n_samples = X.shape[0]
    n_times = X.shape[1]

    # flatten the last dimensions if data is high dimensional
    sample_shape = X.shape[1:]
    if X.ndim > 2:
        X = np.reshape(X, (X.shape[0], -1))

    if connectivity is not None:
        if connectivity.shape[0] == X.shape[1]:  # use global algorithm
            connectivity = connectivity.tocoo()
            n_times = None
        else:  # use temporal adjacency algorithm
            if not round(X.shape[1] / float(connectivity.shape[0])) == n_times:
                raise ValueError('connectivity must be of the correct size')
            # we claim to only use upper triangular part... not true here
            connectivity = (connectivity + connectivity.transpose()).tocsr()
            connectivity = [connectivity.indices[connectivity.indptr[i]:
                            connectivity.indptr[i + 1]] for i in
                            range(len(connectivity.indptr) - 1)]

    if (exclude is not None) and not exclude.size == X.shape[1]:
        raise ValueError('exclude must be the same shape as X[1]')

    # Step 1: Calculate T-stat for original data
    # -------------------------------------------------------------
    T_obs = stat_fun(X)

    # The stat should have the same shape as the samples for no conn.
    if connectivity is None:
        T_obs.shape = sample_shape

    if exclude is not None:
        include = np.logical_not(exclude)
    else:
        include = None

    # determine if connectivity itself can be separated into disjoint sets
    if check_disjoint is True and connectivity is not None:
        partitions = _get_partitions_from_connectivity(connectivity, n_times)
    else:
        partitions = None

    clusters, cluster_stats = _find_clusters(T_obs, threshold, tail,
                                             connectivity, max_step=max_step,
                                             include=include,
                                             partitions=partitions,
                                             t_power=t_power)

    # convert clusters to old format
    if connectivity is not None and out_type == 'mask':
        clusters = _clusters_to_bool(clusters, X.shape[1])

    # The clusters and stat should have the same shape as the samples
    clusters = _reshape_clusters(clusters, sample_shape)
    T_obs.shape = sample_shape

    parallel, my_do_1samp_permutations, _ = parallel_func(
                               _do_1samp_permutations, n_jobs)

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
                seeds = list(seed + np.arange(n_permutations))
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
            H0 = parallel(my_do_1samp_permutations(X, threshold, tail,
                          connectivity, stat_fun, max_step, this_include,
                          partitions, t_power, s, sample_shape)
                          for s in split_list(seeds, n_jobs))
            H0 = np.concatenate(H0)
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
            if step_down_p > 0:
                extra_text = 'additional ' if step_down_iteration > 1 else ''
                new_count = under.size - clusters_kept
                plural = '' if new_count == 1 else 's'
                logger.info('Step-down-in-jumps iteration'
                            '%i found %i %scluster%s'
                            % (step_down_iteration, new_count,
                               extra_text, plural))
            clusters_kept += under.size

        return T_obs, clusters, cluster_pv, H0
    else:
        return T_obs, np.array([]), np.array([]), np.array([])


permutation_cluster_1samp_test.__test__ = False


@verbose
def spatio_temporal_cluster_1samp_test(X, threshold=None,
        n_permutations=1024, tail=0, stat_fun=ttest_1samp_no_p,
        connectivity=None, verbose=None, n_jobs=1, seed=None, max_step=1,
        spatial_exclude=None, step_down_p=0, t_power=1, out_type='indices',
        check_disjoint=False):
    """Non-parametric cluster-level 1 sample T-test for spatio-temporal data

    This function provides a convenient wrapper for data organized in the form
    (observations x space x time) to use permutation_cluster_1samp_test.

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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    n_jobs : int
        See permutation_cluster_1samp_test.
    seed : int or None
        See permutation_cluster_1samp_test.
    max_step : int
        See permutation_cluster_1samp_test.
    spatial_exclude : list of int or None
        List of spatial indices to exclude from clustering.
    step_down_p : float
        See permutation_cluster_1samp_test.
    t_power : float
        See permutation_cluster_1samp_test.
    out_type : str
        See permutation_cluster_1samp_test.
    check_disjoint : bool
        See permutation_cluster_1samp_test.

    Returns
    -------
    T_obs : array of shape [n_tests]
        T-statistic observerd for all variables
    clusters : list
        List type defined by out_type above.
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
        threshold = -stats.distributions.t.ppf(p_thresh, n_samples - 1)
        if np.sign(tail) < 0:
            threshold = -threshold

    # convert spatial_exclude before passing on if necessary
    if spatial_exclude is not None:
        exclude = _st_mask_from_s_inds(n_times, n_vertices,
                                       spatial_exclude, True)
    else:
        exclude = None

    # do the heavy lifting
    out = permutation_cluster_1samp_test(X, threshold=threshold,
              stat_fun=stat_fun, tail=tail, n_permutations=n_permutations,
              connectivity=connectivity, n_jobs=n_jobs, seed=seed,
              max_step=max_step, exclude=exclude, step_down_p=step_down_p,
              t_power=t_power, out_type=out_type,
              check_disjoint=check_disjoint)
    return out


spatio_temporal_cluster_1samp_test.__test__ = False


def _st_mask_from_s_inds(n_times, n_vertices, vertices, set_as=True):
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


@verbose
def _get_partitions_from_connectivity(connectivity, n_times, verbose=None):
    """Use indices to specify disjoint subsets (e.g., hemispheres) based on
    connectivity"""
    if isinstance(connectivity, list):
        test = np.ones(len(connectivity))
        test_conn = np.zeros((len(connectivity), len(connectivity)),
                             dtype='bool')
        for vi in range(len(connectivity)):
            test_conn[connectivity[vi], vi] = True
        test_conn = sparse.coo_matrix(test_conn, dtype='float')
    else:
        test = np.ones(connectivity.shape[0])
        test_conn = connectivity

    part_clusts, _ = _find_clusters(test, 0, 1, test_conn)
    if len(part_clusts) > 1:
        logger.info('%i disjoint connectivity sets found'
                    % len(part_clusts))
        partitions = np.zeros(len(test), dtype='int')
        for ii, pc in enumerate(part_clusts):
            partitions[pc] = ii
        if isinstance(connectivity, list):
            partitions = np.tile(partitions, n_times)
    else:
        logger.info('No disjoint connectivity sets found')
        partitions = None

    return partitions


def _reshape_clusters(clusters, sample_shape):
    """Reshape cluster masks or indices to be of the correct shape"""
    # format of the bool mask and indices are ndarrays
    if len(clusters) > 0 and isinstance(clusters[0], np.ndarray):
        if clusters[0].dtype == bool:  # format of mask
            clusters = [c.reshape(sample_shape) for c in clusters]
        else:  # format of indices
            clusters = [unravel_index(c, sample_shape) for c in clusters]
    return clusters
