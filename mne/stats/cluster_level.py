#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Thorsten Kranz <thorstenkranz@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: Simplified BSD

import logging

import numpy as np
from scipy import sparse

from .parametric import f_oneway
from ..parallel import parallel_func, check_n_jobs
from ..utils import split_list, logger, verbose, ProgressBar, warn, _pl
from ..source_estimate import SourceEstimate


def _get_clusters_spatial(s, neighbors):
    """Form spatial clusters using neighbor lists.

    This is equivalent to _get_components with n_times = 1, with a properly
    reconfigured connectivity matrix (formed as "neighbors" list)
    """
    # s is a vector of spatial indices that are significant, like:
    #     s = np.where(x_in)[0]
    # for x_in representing a single time-instant
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
            buddies = buddies[np.in1d(s[buddies], neighbors[s[ind]],
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
    """Reassign cluster numbers."""
    # reconfigure check matrix
    check[check == num] = base
    # concatenate new values into clusters array
    clusters[base - 1] = np.concatenate((clusters[base - 1],
                                         clusters[num - 1]))
    clusters[num - 1] = np.array([], dtype=int)


def _get_clusters_st_1step(keepers, neighbors):
    """Directly calculate connectivity.

    This uses knowledge that time points are
    only connected to adjacent neighbors for data organized as time x space.

    This algorithm time increases linearly with the number of time points,
    compared to with the square for the standard (graph) algorithm.

    This algorithm creates clusters for each time point using a method more
    efficient than the standard graph method (but otherwise equivalent), then
    combines these clusters across time points in a reasonable way.
    """
    n_src = len(neighbors)
    n_times = len(keepers)
    # start cluster numbering at 1 for diffing convenience
    enum_offset = 1
    check = np.zeros((n_times, n_src), dtype=int)
    clusters = list()
    for ii, k in enumerate(keepers):
        c = _get_clusters_spatial(k, neighbors)
        for ci, cl in enumerate(c):
            check[ii, cl] = ci + enum_offset
        enum_offset += len(c)
        # give them the correct offsets
        c = [cl + ii * n_src for cl in c]
        clusters += c

    # now that each cluster has been assigned a unique number, combine them
    # by going through each time point
    for check1, check2, k in zip(check[:-1], check[1:], keepers[:-1]):
        # go through each one that needs reassignment
        inds = k[check2[k] - check1[k] > 0]
        check1_d = check1[inds]
        n = check2[inds]
        nexts = np.unique(n)
        for num in nexts:
            prevs = check1_d[n == num]
            base = np.min(prevs)
            for pr in np.unique(prevs[prevs != base]):
                _reassign(check1, clusters, base, pr)
            # reassign values
            _reassign(check2, clusters, base, num)
    # clean up clusters
    clusters = [cl for cl in clusters if len(cl) > 0]
    return clusters


def _get_clusters_st_multistep(keepers, neighbors, max_step=1):
    """Directly calculate connectivity.

    This uses knowledge that time points are
    only connected to adjacent neighbors for data organized as time x space.

    This algorithm time increases linearly with the number of time points,
    compared to with the square for the standard (graph) algorithm.
    """
    n_src = len(neighbors)
    n_times = len(keepers)
    t_border = list()
    t_border.append(0)
    for ki, k in enumerate(keepers):
        keepers[ki] = k + ki * n_src
        t_border.append(t_border[ki] + len(k))
    t_border = np.array(t_border)
    keepers = np.concatenate(keepers)
    v = keepers
    t, s = divmod(v, n_src)

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


def _get_clusters_st(x_in, neighbors, max_step=1):
    """Choose the most efficient version."""
    n_src = len(neighbors)
    n_times = x_in.size // n_src
    cl_goods = np.where(x_in)[0]
    if len(cl_goods) > 0:
        keepers = [np.array([], dtype=int)] * n_times
        row, col = np.unravel_index(cl_goods, (n_times, n_src))
        if isinstance(row, int):
            row = [row]
            col = [col]
            lims = [0]
        else:
            order = np.argsort(row)
            row = row[order]
            col = col[order]
            lims = [0] + (np.where(np.diff(row) > 0)[0] +
                          1).tolist() + [len(row)]

        for start, end in zip(lims[:-1], lims[1:]):
            keepers[row[start]] = np.sort(col[start:end])
        if max_step == 1:
            return _get_clusters_st_1step(keepers, neighbors)
        else:
            return _get_clusters_st_multistep(keepers, neighbors,
                                              max_step)
    else:
        return []


def _get_components(x_in, connectivity, return_list=True):
    """Get connected components from a mask and a connectivity matrix."""
    try:
        from sklearn.utils._csgraph import cs_graph_components
    except ImportError:
        try:
            from scikits.learn.utils._csgraph import cs_graph_components
        except ImportError:
            try:
                from sklearn.utils.sparsetools import connected_components
                cs_graph_components = connected_components
            except ImportError:
                # in theory we might be able to shoehorn this into using
                # _get_clusters_spatial if we transform connectivity into
                # a neighbor list, and it might end up being faster anyway,
                # but for now:
                raise ImportError('scikit-learn must be installed')

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
        start = np.min(components)
        stop = np.max(components)
        comp_list = [list() for i in range(start, stop + 1, 1)]
        mask = np.zeros(len(comp_list), dtype=bool)
        for ii, comp in enumerate(components):
            comp_list[comp].append(ii)
            mask[comp] += x_in[ii]
        clusters = [np.array(k) for k, m in zip(comp_list, mask) if m]
        return clusters
    else:
        return components


def _find_clusters(x, threshold, tail=0, connectivity=None, max_step=1,
                   include=None, partitions=None, t_power=1, show_info=False):
    """Find all clusters which are above/below a certain threshold.

    When doing a two-tailed test (tail == 0), only points with the same
    sign will be clustered together.

    Parameters
    ----------
    x : 1D array
        Data
    threshold : float | dict
        Where to threshold the statistic. Should be negative for tail == -1,
        and positive for tail == 0 or 1. Can also be an dict for
        threshold-free cluster enhancement.
    tail : -1 | 0 | 1
        Type of comparison
    connectivity : sparse matrix in COO format, None, or list
        Defines connectivity between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        If connectivity is a list, it is assumed that each entry stores the
        indices of the spatial neighbors in a spatio-temporal dataset x.
        Default is None, i.e, a regular lattice connectivity.
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
    show_info : bool
        If True, display information about thresholds used (for TFCE). Should
        only be done for the standard permutation.

    Returns
    -------
    clusters : list of slices or list of arrays (boolean masks)
        We use slices for 1D signals and mask to multidimensional
        arrays.
    sums: array
        Sum of x values in clusters.
    """
    from scipy import ndimage
    if tail not in [-1, 0, 1]:
        raise ValueError('invalid tail parameter')

    x = np.asanyarray(x)

    if not np.isscalar(threshold):
        if not isinstance(threshold, dict):
            raise TypeError('threshold must be a number, or a dict for '
                            'threshold-free cluster enhancement')
        if not all(key in threshold for key in ['start', 'step']):
            raise KeyError('threshold, if dict, must have at least '
                           '"start" and "step"')
        tfce = True
        if tail == -1:
            if threshold['start'] > 0:
                raise ValueError('threshold["start"] must be <= 0 for '
                                 'tail == -1')
            if threshold['step'] >= 0:
                raise ValueError('threshold["step"] must be < 0 for '
                                 'tail == -1')
            stop = np.min(x)
        elif tail == 1:
            stop = np.max(x)
        else:  # tail == 0
            stop = np.max(np.abs(x))
        thresholds = np.arange(threshold['start'], stop,
                               threshold['step'], float)
        h_power = threshold.get('h_power', 2)
        e_power = threshold.get('e_power', 0.5)
        if show_info is True:
            if len(thresholds) == 0:
                warn('threshold["start"] (%s) is more extreme than data '
                     'statistics with most extreme value %s'
                     % (threshold['start'], stop))
            else:
                logger.info('Using %d thresholds from %0.2f to %0.2f for TFCE '
                            'computation (h_power=%0.2f, e_power=%0.2f)'
                            % (len(thresholds), thresholds[0], thresholds[-1],
                               h_power, e_power))
        scores = np.zeros(x.size)
    else:
        thresholds = [threshold]
        tfce = False

    # include all points by default
    if include is None:
        include = np.ones(x.shape, dtype=bool)

    if not np.all(np.diff(thresholds) > 0):
        raise RuntimeError('Threshold misconfiguration, must be monotonically'
                           ' increasing')

    # set these here just in case thresholds == []
    clusters = list()
    sums = np.empty(0)
    for ti, thresh in enumerate(thresholds):
        # these need to be reset on each run
        clusters = list()
        sums = np.empty(0)
        if tail == 0:
            x_ins = [np.logical_and(x > thresh, include),
                     np.logical_and(x < -thresh, include)]
        elif tail == -1:
            x_ins = [np.logical_and(x < thresh, include)]
        else:  # tail == 1
            x_ins = [np.logical_and(x > thresh, include)]
        # loop over tails
        for x_in in x_ins:
            if np.any(x_in):
                out = _find_clusters_1dir_parts(x, x_in, connectivity,
                                                max_step, partitions, t_power,
                                                ndimage)
                clusters += out[0]
                sums = np.concatenate((sums, out[1]))
        if tfce is True:
            # the score of each point is the sum of the h^H * e^E for each
            # supporting section "rectangle" h x e.
            if ti == 0:
                h = abs(thresh)
            else:
                h = abs(thresh - thresholds[ti - 1])
            h = h ** h_power
            for c in clusters:
                # triage based on cluster storage type
                if isinstance(c, slice):
                    len_c = c.stop - c.start
                elif isinstance(c, tuple):
                    len_c = len(c)
                elif c.dtype == bool:
                    len_c = np.sum(c)
                else:
                    len_c = len(c)
                scores[c] += h * (len_c ** e_power)
    if tfce is True:
        # each point gets treated independently
        clusters = np.arange(x.size)
        if connectivity is None:
            if x.ndim == 1:
                # slices
                clusters = [slice(c, c + 1) for c in clusters]
            else:
                # boolean masks (raveled)
                clusters = [(clusters == ii).ravel()
                            for ii in range(len(clusters))]
        else:
            clusters = [np.array([c]) for c in clusters]
        sums = scores
    return clusters, sums


def _find_clusters_1dir_parts(x, x_in, connectivity, max_step, partitions,
                              t_power, ndimage):
    """Deal with partitions, and pass the work to _find_clusters_1dir."""
    if partitions is None:
        clusters, sums = _find_clusters_1dir(x, x_in, connectivity, max_step,
                                             t_power, ndimage)
    else:
        # cluster each partition separately
        clusters = list()
        sums = list()
        for p in range(np.max(partitions) + 1):
            x_i = np.logical_and(x_in, partitions == p)
            out = _find_clusters_1dir(x, x_i, connectivity, max_step, t_power,
                                      ndimage)
            clusters += out[0]
            sums.append(out[1])
        sums = np.concatenate(sums)
    return clusters, sums


def _find_clusters_1dir(x, x_in, connectivity, max_step, t_power, ndimage):
    """Actually call the clustering algorithm."""
    if connectivity is None:
        labels, n_labels = ndimage.label(x_in)

        if x.ndim == 1:
            # slices
            clusters = ndimage.find_objects(labels, n_labels)
            if len(clusters) == 0:
                sums = list()
            else:
                index = list(range(1, n_labels + 1))
                if t_power == 1:
                    sums = ndimage.measurements.sum(x, labels, index=index)
                else:
                    sums = ndimage.measurements.sum(np.sign(x) *
                                                    np.abs(x) ** t_power,
                                                    labels, index=index)
        else:
            # boolean masks (raveled)
            clusters = list()
            sums = np.empty(n_labels)
            for l in range(1, n_labels + 1):
                c = labels == l
                clusters.append(c.ravel())
                if t_power == 1:
                    sums[l - 1] = np.sum(x[c])
                else:
                    sums[l - 1] = np.sum(np.sign(x[c]) *
                                         np.abs(x[c]) ** t_power)
    else:
        if x.ndim > 1:
            raise Exception("Data should be 1D when using a connectivity "
                            "to define clusters.")
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


def _cluster_indices_to_mask(components, n_tot):
    """Convert to the old format of clusters, which were bool arrays."""
    for ci, c in enumerate(components):
        components[ci] = np.zeros((n_tot), dtype=bool)
        components[ci][c] = True
    return components


def _cluster_mask_to_indices(components):
    """Convert to the old format of clusters, which were bool arrays."""
    for ci, c in enumerate(components):
        if not isinstance(c, slice):
            components[ci] = np.where(c)[0]
    return components


def _pval_from_histogram(T, H0, tail):
    """Get p-values from stats values given an H0 distribution.

    For each stat compute a p-value as percentile of its statistics
    within all statistics in surrogate data
    """
    if tail not in [-1, 0, 1]:
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


def _setup_connectivity(connectivity, n_vertices, n_times):
    if connectivity.shape[0] == n_vertices:  # use global algorithm
        connectivity = connectivity.tocoo()
        n_times = None
    else:  # use temporal adjacency algorithm
        if not round(n_vertices / float(connectivity.shape[0])) == n_times:
            raise ValueError('connectivity must be of the correct size')
        # we claim to only use upper triangular part... not true here
        connectivity = (connectivity + connectivity.transpose()).tocsr()
        connectivity = [connectivity.indices[connectivity.indptr[i]:
                        connectivity.indptr[i + 1]] for i in
                        range(len(connectivity.indptr) - 1)]
    return connectivity


def _do_permutations(X_full, slices, threshold, tail, connectivity, stat_fun,
                     max_step, include, partitions, t_power, seeds,
                     sample_shape, buffer_size, progress_bar):
    n_samp, n_vars = X_full.shape

    if buffer_size is not None and n_vars <= buffer_size:
        buffer_size = None  # don't use buffer for few variables

    # allocate space for output
    max_cluster_sums = np.empty(len(seeds), dtype=np.double)

    if buffer_size is not None:
        # allocate buffer, so we don't need to allocate memory during loop
        X_buffer = [np.empty((len(X_full[s]), buffer_size), dtype=X_full.dtype)
                    for s in slices]

    for seed_idx, seed in enumerate(seeds):
        if progress_bar is not None:
            if (not (seed_idx + 1) % 32) or (seed_idx == 0):
                progress_bar.update(seed_idx + 1)

        # shuffle sample indices
        rng = np.random.RandomState(seed)
        idx_shuffled = np.arange(n_samp)
        rng.shuffle(idx_shuffled)
        idx_shuffle_list = [idx_shuffled[s] for s in slices]

        if buffer_size is None:
            # shuffle all data at once
            X_shuffle_list = [X_full[idx, :] for idx in idx_shuffle_list]
            T_obs_surr = stat_fun(*X_shuffle_list)
        else:
            # only shuffle a small data buffer, so we need less memory
            T_obs_surr = np.empty(n_vars, dtype=X_full.dtype)

            for pos in range(0, n_vars, buffer_size):
                # number of variables for this loop
                n_var_loop = min(pos + buffer_size, n_vars) - pos

                # fill buffer
                for i, idx in enumerate(idx_shuffle_list):
                    X_buffer[i][:, :n_var_loop] =\
                        X_full[idx, pos: pos + n_var_loop]

                # apply stat_fun and store result
                tmp = stat_fun(*X_buffer)
                T_obs_surr[pos: pos + n_var_loop] = tmp[:n_var_loop]

        # The stat should have the same shape as the samples for no conn.
        if connectivity is None:
            T_obs_surr.shape = sample_shape

        # Find cluster on randomized stats
        out = _find_clusters(T_obs_surr, threshold=threshold, tail=tail,
                             max_step=max_step, connectivity=connectivity,
                             partitions=partitions, include=include,
                             t_power=t_power)
        perm_clusters_sums = out[1]

        if len(perm_clusters_sums) > 0:
            max_cluster_sums[seed_idx] = np.max(perm_clusters_sums)
        else:
            max_cluster_sums[seed_idx] = 0

    return max_cluster_sums


def _do_1samp_permutations(X, slices, threshold, tail, connectivity, stat_fun,
                           max_step, include, partitions, t_power, seeds,
                           sample_shape, buffer_size, progress_bar):
    n_samp, n_vars = X.shape
    assert slices is None  # should be None for the 1 sample case

    if buffer_size is not None and n_vars <= buffer_size:
        buffer_size = None  # don't use buffer for few variables

    # allocate space for output
    max_cluster_sums = np.empty(len(seeds), dtype=np.double)

    if buffer_size is not None:
        # allocate a buffer so we don't need to allocate memory in loop
        X_flip_buffer = np.empty((n_samp, buffer_size), dtype=X.dtype)

    for seed_idx, seed in enumerate(seeds):
        if progress_bar is not None:
            if not (seed_idx + 1) % 32 or seed_idx == 0:
                progress_bar.update(seed_idx + 1)

        if isinstance(seed, np.ndarray):
            # new surrogate data with specified sign flip
            if not seed.size == n_samp:
                raise ValueError('rng string must be n_samples long')
            signs = 2 * seed[:, None].astype(int) - 1
            if not np.all(np.equal(np.abs(signs), 1)):
                raise ValueError('signs from rng must be +/- 1')
        else:
            rng = np.random.RandomState(seed)
            # new surrogate data with random sign flip
            signs = np.sign(0.5 - rng.rand(n_samp))
            signs = signs[:, np.newaxis]

        if buffer_size is None:
            # be careful about non-writable memmap (GH#1507)
            if X.flags.writeable:
                X *= signs
                # Recompute statistic on randomized data
                T_obs_surr = stat_fun(X)
                # Set X back to previous state (trade memory eff. for CPU use)
                X *= signs
            else:
                T_obs_surr = stat_fun(X * signs)
        else:
            # only sign-flip a small data buffer, so we need less memory
            T_obs_surr = np.empty(n_vars, dtype=X.dtype)

            for pos in range(0, n_vars, buffer_size):
                # number of variables for this loop
                n_var_loop = min(pos + buffer_size, n_vars) - pos

                X_flip_buffer[:, :n_var_loop] =\
                    signs * X[:, pos: pos + n_var_loop]

                # apply stat_fun and store result
                tmp = stat_fun(X_flip_buffer)
                T_obs_surr[pos: pos + n_var_loop] = tmp[:n_var_loop]

        # The stat should have the same shape as the samples for no conn.
        if connectivity is None:
            T_obs_surr.shape = sample_shape

        # Find cluster on randomized stats
        out = _find_clusters(T_obs_surr, threshold=threshold, tail=tail,
                             max_step=max_step, connectivity=connectivity,
                             partitions=partitions, include=include,
                             t_power=t_power)
        perm_clusters_sums = out[1]
        if len(perm_clusters_sums) > 0:
            # get max with sign info
            idx_max = np.argmax(np.abs(perm_clusters_sums))
            max_cluster_sums[seed_idx] = perm_clusters_sums[idx_max]
        else:
            max_cluster_sums[seed_idx] = 0

    return max_cluster_sums


@verbose
def _permutation_cluster_test(X, threshold, n_permutations, tail, stat_fun,
                              connectivity, verbose, n_jobs, seed, max_step,
                              exclude, step_down_p, t_power, out_type,
                              check_disjoint, buffer_size):
    n_jobs = check_n_jobs(n_jobs)
    """Aux Function.

    Note. X is required to be a list. Depending on the length of X
    either a 1 sample t-test or an f-test / more sample permutation scheme
    is elicited.
    """
    if out_type not in ['mask', 'indices']:
        raise ValueError('out_type must be either \'mask\' or \'indices\'')
    if not isinstance(threshold, dict) and (tail < 0 and threshold > 0 or
                                            tail > 0 and threshold < 0 or
                                            tail == 0 and threshold < 0):
        raise ValueError('incompatible tail and threshold signs, got %s and %s'
                         % (tail, threshold))

    # check dimensions for each group in X (a list at this stage).
    X = [x[:, np.newaxis] if x.ndim == 1 else x for x in X]
    n_samples = X[0].shape[0]
    n_times = X[0].shape[1]

    sample_shape = X[0].shape[1:]
    for x in X:
        if x.shape[1:] != sample_shape:
            raise ValueError('All samples mush have the same size')

    # flatten the last dimensions in case the data is high dimensional
    X = [np.reshape(x, (x.shape[0], -1)) for x in X]
    n_tests = X[0].shape[1]

    if connectivity is not None:
        connectivity = _setup_connectivity(connectivity, n_tests, n_times)

    if (exclude is not None) and not exclude.size == n_tests:
        raise ValueError('exclude must be the same shape as X[0]')

    # Step 1: Calculate T-stat for original data
    # -------------------------------------------------------------
    T_obs = stat_fun(*X)
    logger.info('stat_fun(H1): min=%f max=%f' % (np.min(T_obs), np.max(T_obs)))

    # test if stat_fun treats variables independently
    if buffer_size is not None:
        T_obs_buffer = np.zeros_like(T_obs)
        for pos in range(0, n_tests, buffer_size):
            T_obs_buffer[pos: pos + buffer_size] =\
                stat_fun(*[x[:, pos: pos + buffer_size] for x in X])

        if not np.alltrue(T_obs == T_obs_buffer):
            warn('Provided stat_fun does not treat variables independently. '
                 'Setting buffer_size to None.')
            buffer_size = None

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
    logger.info('Running initial clustering')
    out = _find_clusters(T_obs, threshold, tail, connectivity,
                         max_step=max_step, include=include,
                         partitions=partitions, t_power=t_power,
                         show_info=True)
    clusters, cluster_stats = out
    # For TFCE, return the "adjusted" statistic instead of raw scores
    if isinstance(threshold, dict):
        T_obs = cluster_stats.copy()

    logger.info('Found %d clusters' % len(clusters))

    # convert clusters to old format
    if connectivity is not None:
        # our algorithms output lists of indices by default
        if out_type == 'mask':
            clusters = _cluster_indices_to_mask(clusters, n_tests)
    else:
        # ndimage outputs slices or boolean masks by default
        if out_type == 'indices':
            clusters = _cluster_mask_to_indices(clusters)

    # The stat should have the same shape as the samples
    T_obs.shape = sample_shape

    if len(X) == 1:  # 1 sample test
        do_perm_func = _do_1samp_permutations
        X_full = X[0]
        slices = None
    else:
        do_perm_func = _do_permutations
        X_full = np.concatenate(X, axis=0)
        n_samples_per_condition = [x.shape[0] for x in X]
        splits_idx = np.append([0], np.cumsum(n_samples_per_condition))
        slices = [slice(splits_idx[k], splits_idx[k + 1])
                  for k in range(len(X))]
    parallel, my_do_perm_func, _ = parallel_func(do_perm_func, n_jobs)

    # Step 2: If we have some clusters, repeat process on permuted data
    # -------------------------------------------------------------------

    def get_progress_bar(seeds):
        # make sure the progress bar adds to up 100% across n jobs
        return (ProgressBar(len(seeds), spinner=True) if
                logger.level <= logging.INFO else None)

    if len(clusters) > 0:
        # check to see if we can do an exact test
        # note for a two-tailed test, we can exploit symmetry to just do half
        seeds = None
        if len(X) == 1:
            max_perms = 2 ** (n_samples - (tail == 0))
            if max_perms <= n_permutations:
                # omit first perm b/c accounted for in _pval_from_histogram,
                # convert to binary array representation
                seeds = [np.fromiter(np.binary_repr(s, n_samples), dtype=int)
                         for s in range(1, max_perms)]

        if seeds is None:
            if seed is None:
                seeds = [None] * n_permutations
            else:
                seeds = list(seed + np.arange(n_permutations))

        # Step 3: repeat permutations for step-down-in-jumps procedure
        n_removed = 1  # number of new clusters added
        total_removed = 0
        step_down_include = None  # start out including all points
        n_step_downs = 0

        while n_removed > 0:
            # actually do the clustering for each partition
            if include is not None:
                if step_down_include is not None:
                    this_include = np.logical_and(include, step_down_include)
                else:
                    this_include = include
            else:
                this_include = step_down_include
            logger.info('Permuting ...')
            H0 = parallel(my_do_perm_func(X_full, slices, threshold, tail,
                          connectivity, stat_fun, max_step, this_include,
                          partitions, t_power, s, sample_shape, buffer_size,
                          get_progress_bar(s))
                          for s in split_list(seeds, n_jobs))
            H0 = np.concatenate(H0)
            logger.info('Computing cluster p-values')
            cluster_pv = _pval_from_histogram(cluster_stats, H0, tail)

            # figure out how many new ones will be removed for step-down
            to_remove = np.where(cluster_pv < step_down_p)[0]
            n_removed = to_remove.size - total_removed
            total_removed = to_remove.size
            step_down_include = np.ones(n_tests, dtype=bool)
            for ti in to_remove:
                step_down_include[clusters[ti]] = False
            if connectivity is None:
                step_down_include.shape = sample_shape
            n_step_downs += 1
            if step_down_p > 0:
                a_text = 'additional ' if n_step_downs > 1 else ''
                logger.info('Step-down-in-jumps iteration #%i found %i %s'
                            'cluster%s to exclude from subsequent iterations'
                            % (n_step_downs, n_removed, a_text,
                               _pl(n_removed)))
        logger.info('Done.')
        # The clusters should have the same shape as the samples
        clusters = _reshape_clusters(clusters, sample_shape)
        return T_obs, clusters, cluster_pv, H0
    else:
        return T_obs, np.array([]), np.array([]), np.array([])


def ttest_1samp_no_p(X, sigma=0, method='relative'):
    """Perform t-test with variance adjustment and no p-value calculation.

    Parameters
    ----------
    X : array
        Array to return t-values for.
    sigma : float
        The variance estate will be given by "var + sigma * max(var)" or
        "var + sigma", depending on "method". By default this is 0 (no
        adjustment). See Notes for details.
    method : str
        If 'relative', the minimum variance estimate will be sigma * max(var),
        if 'absolute' the minimum variance estimate will be sigma.

    Returns
    -------
    t : array
        t-values, potentially adjusted using the hat method.

    Notes
    -----
    One can use the conversion:

        threshold = -scipy.stats.distributions.t.ppf(p_thresh, n_samples - 1)

    to convert a desired p-value threshold to t-value threshold. Don't forget
    that for two-tailed tests, p_thresh in the above should be divided by 2.

    To use the "hat" adjustment method, a value of sigma=1e-3 may be a
    reasonable choice. See Ridgway et al. 2012 "The problem of low variance
    voxels in statistical parametric mapping; a new hat avoids a 'haircut'",
    NeuroImage. 2012 Feb 1;59(3):2131-41.
    """
    if method not in ['absolute', 'relative']:
        raise ValueError('method must be "absolute" or "relative", not %s'
                         % method)
    var = np.var(X, axis=0, ddof=1)
    if sigma > 0:
        limit = sigma * np.max(var) if method == 'relative' else sigma
        var += limit
    return np.mean(X, axis=0) / np.sqrt(var / X.shape[0])


@verbose
def permutation_cluster_test(X, threshold=None, n_permutations=1024,
                             tail=0, stat_fun=f_oneway,
                             connectivity=None, verbose=None, n_jobs=1,
                             seed=None, max_step=1, exclude=None,
                             step_down_p=0, t_power=1, out_type='mask',
                             check_disjoint=False, buffer_size=1000):
    """Cluster-level statistical permutation test.

    For a list of nd-arrays of data, e.g. 2d for time series or 3d for
    time-frequency power values, calculate some statistics corrected for
    multiple comparisons using permutations and cluster level correction.
    Each element of the list X contains the data for one group of
    observations. Randomized data are generated with random partitions
    of the data.

    Parameters
    ----------
    X : list
        List of nd-arrays containing the data. Each element of X contains
        the samples for one group. First dimension of each element is the
        number of samples/observations in this group. The other dimensions
        are for the size of the observations. For example if X = [X1, X2]
        with X1.shape = (20, 50, 4) and X2.shape = (17, 50, 4) one has
        2 groups with respectively 20 and 17 observations in each.
        Each data point is of shape (50, 4).
    threshold : float | dict | None
        If threshold is None, it will choose a t-threshold equivalent to
        p < 0.05 for the given number of (within-subject) observations.
        If a dict is used, then threshold-free cluster enhancement (TFCE)
        will be used.
    n_permutations : int
        The number of permutations to compute.
    tail : -1 or 0 or 1 (default = 0)
        If tail is 1, the statistic is thresholded above threshold.
        If tail is -1, the statistic is thresholded below threshold.
        If tail is 0, the statistic is thresholded on both sides of
        the distribution.
    stat_fun : callable
        function called to calculate statistics, must accept 1d-arrays as
        arguments (default: scipy.stats.f_oneway).
    connectivity : sparse matrix.
        Defines connectivity between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is None, i.e, a regular lattice connectivity.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    n_jobs : int
        Number of permutations to run in parallel (requires joblib package).
    seed : int or None
        Seed the random number generator for results reproducibility.
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
        Power to raise the statistical values (usually f-values) by before
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
    buffer_size: int or None
        The statistics will be computed for blocks of variables of size
        "buffer_size" at a time. This is option significantly reduces the
        memory requirements when n_jobs > 1 and memory sharing between
        processes is enabled (see set_cache_dir()), as X will be shared
        between processes and each process only needs to allocate space
        for a small block of variables.

    Returns
    -------
    T_obs : array of shape [n_tests]
        T-statistic observed for all variables.
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
    from scipy import stats
    ppf = stats.f.ppf
    if threshold is None:
        p_thresh = 0.05 / (1 + (tail == 0))
        n_samples_per_group = [len(x) for x in X]
        threshold = ppf(1. - p_thresh, *n_samples_per_group)
        if np.sign(tail) < 0:
            threshold = -threshold

    return _permutation_cluster_test(X=X, threshold=threshold,
                                     n_permutations=n_permutations,
                                     tail=tail, stat_fun=stat_fun,
                                     connectivity=connectivity,
                                     verbose=verbose,
                                     n_jobs=n_jobs, seed=seed,
                                     max_step=max_step,
                                     exclude=exclude, step_down_p=step_down_p,
                                     t_power=t_power, out_type=out_type,
                                     check_disjoint=check_disjoint,
                                     buffer_size=buffer_size)


permutation_cluster_test.__test__ = False


@verbose
def permutation_cluster_1samp_test(X, threshold=None, n_permutations=1024,
                                   tail=0, stat_fun=ttest_1samp_no_p,
                                   connectivity=None, verbose=None, n_jobs=1,
                                   seed=None, max_step=1, exclude=None,
                                   step_down_p=0, t_power=1, out_type='mask',
                                   check_disjoint=False, buffer_size=1000):
    """Non-parametric cluster-level 1 sample T-test.

    From a array of observations, e.g. signal amplitudes or power spectrum
    estimates etc., calculate if the observed mean significantly deviates
    from 0. The procedure uses a cluster analysis with permutation test
    for calculating corrected p-values. Randomized data are generated with
    random sign flips.

    Parameters
    ----------
    X : array, shape=(n_samples, p, q) or (n_samples, p)
        Array where the first dimension corresponds to the
        samples (observations). X[k] can be a 1D or 2D array (time series
        or TF image) associated to the kth observation.
    threshold : float | dict | None
        If threshold is None, it will choose a t-threshold equivalent to
        p < 0.05 for the given number of (within-subject) observations.
        If a dict is used, then threshold-free cluster enhancement (TFCE)
        will be used.
    n_permutations : int
        The number of permutations to compute.
    tail : -1 or 0 or 1 (default = 0)
        If tail is 1, the statistic is thresholded above threshold.
        If tail is -1, the statistic is thresholded below threshold.
        If tail is 0, the statistic is thresholded on both sides of
        the distribution.
    stat_fun : function
        Function used to compute the statistical map.
    connectivity : sparse matrix or None
        Defines connectivity between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        This matrix must be square with dimension (n_vertices * n_times) or
        (n_vertices). Default is None, i.e, a regular lattice connectivity.
        Use square n_vertices matrix for datasets with a large temporal
        extent to save on memory and computation time.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    n_jobs : int
        Number of permutations to run in parallel (requires joblib package).
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
    buffer_size: int or None
        The statistics will be computed for blocks of variables of size
        "buffer_size" at a time. This is option significantly reduces the
        memory requirements when n_jobs > 1 and memory sharing between
        processes is enabled (see set_cache_dir()), as X will be shared
        between processes and each process only needs to allocate space
        for a small block of variables.

    Returns
    -------
    T_obs : array of shape [n_tests]
        T-statistic observed for all variables
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
    from scipy import stats
    ppf = stats.t.ppf
    if threshold is None:
        p_thresh = 0.05 / (1 + (tail == 0))
        n_samples = len(X)
        threshold = -ppf(p_thresh, n_samples - 1)
        if np.sign(tail) < 0:
            threshold = -threshold

    X = [X]  # for one sample only one data array
    return _permutation_cluster_test(X=X,
                                     threshold=threshold,
                                     n_permutations=n_permutations,
                                     tail=tail, stat_fun=stat_fun,
                                     connectivity=connectivity,
                                     verbose=verbose,
                                     n_jobs=n_jobs, seed=seed,
                                     max_step=max_step,
                                     exclude=exclude, step_down_p=step_down_p,
                                     t_power=t_power, out_type=out_type,
                                     check_disjoint=check_disjoint,
                                     buffer_size=buffer_size)


permutation_cluster_1samp_test.__test__ = False


@verbose
def spatio_temporal_cluster_1samp_test(X, threshold=None,
                                       n_permutations=1024, tail=0,
                                       stat_fun=ttest_1samp_no_p,
                                       connectivity=None, verbose=None,
                                       n_jobs=1, seed=None, max_step=1,
                                       spatial_exclude=None, step_down_p=0,
                                       t_power=1, out_type='indices',
                                       check_disjoint=False, buffer_size=1000):
    """Non-parametric cluster-level 1 sample T-test for spatio-temporal data.

    This function provides a convenient wrapper for data organized in the form
    (observations x time x space) to use permutation_cluster_1samp_test.

    Parameters
    ----------
    X : array
        Array of shape observations x time x vertices.
    threshold : float | dict | None
        If threshold is None, it will choose a t-threshold equivalent to
        p < 0.05 for the given number of (within-subject) observations.
        If a dict is used, then threshold-free cluster enhancement (TFCE)
        will be used.
    n_permutations : int
        The number of permutations to compute.
    tail : -1 or 0 or 1 (default = 0)
        If tail is 1, the statistic is thresholded above threshold.
        If tail is -1, the statistic is thresholded below threshold.
        If tail is 0, the statistic is thresholded on both sides of
        the distribution.
    stat_fun : function
        Function used to compute the statistical map.
    connectivity : sparse matrix or None
        Defines connectivity between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        This matrix must be square with dimension (n_vertices * n_times) or
        (n_vertices). Default is None, i.e, a regular lattice connectivity.
        Use square n_vertices matrix for datasets with a large temporal
        extent to save on memory and computation time.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    n_jobs : int
        Number of permutations to run in parallel (requires joblib package).
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
    spatial_exclude : list of int or None
        List of spatial indices to exclude from clustering.
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
    buffer_size: int or None
        The statistics will be computed for blocks of variables of size
        "buffer_size" at a time. This is option significantly reduces the
        memory requirements when n_jobs > 1 and memory sharing between
        processes is enabled (see set_cache_dir()), as X will be shared
        between processes and each process only needs to allocate space
        for a small block of variables.

    Returns
    -------
    T_obs : array of shape [n_tests]
        T-statistic observed for all variables.
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

    TFCE originally described in Smith/Nichols (2009),
    "Threshold-free cluster enhancement: Addressing problems of
    smoothing, threshold dependence, and localisation in cluster
    inference", NeuroImage 44 (2009) 83-98.
    """
    n_samples, n_times, n_vertices = X.shape

    # convert spatial_exclude before passing on if necessary
    if spatial_exclude is not None:
        exclude = _st_mask_from_s_inds(n_times, n_vertices,
                                       spatial_exclude, True)
    else:
        exclude = None

    # do the heavy lifting
    out = permutation_cluster_1samp_test(X, threshold=threshold,
                                         stat_fun=stat_fun, tail=tail,
                                         n_permutations=n_permutations,
                                         connectivity=connectivity,
                                         n_jobs=n_jobs, seed=seed,
                                         max_step=max_step, exclude=exclude,
                                         step_down_p=step_down_p,
                                         t_power=t_power, out_type=out_type,
                                         check_disjoint=check_disjoint,
                                         buffer_size=buffer_size)
    return out


spatio_temporal_cluster_1samp_test.__test__ = False


@verbose
def spatio_temporal_cluster_test(X, threshold=1.67, n_permutations=1024,
                                 tail=0, stat_fun=f_oneway,
                                 connectivity=None, verbose=None, n_jobs=1,
                                 seed=None, max_step=1, spatial_exclude=None,
                                 step_down_p=0, t_power=1, out_type='indices',
                                 check_disjoint=False, buffer_size=1000):
    """Non-parametric cluster-level test for spatio-temporal data.

    This function provides a convenient wrapper for data organized in the form
    (observations x time x space) to use permutation_cluster_test.

    Parameters
    ----------
    X: list of arrays
        Array of shape (observations, time, vertices) in each group.
    threshold: float
        The threshold for the statistic.
    n_permutations: int
        See permutation_cluster_test.
    tail : -1 or 0 or 1 (default = 0)
        See permutation_cluster_test.
    stat_fun : function
        function called to calculate statistics, must accept 1d-arrays as
        arguments (default: scipy.stats.f_oneway)
    connectivity : sparse matrix or None
        Defines connectivity between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is None, i.e, a regular lattice connectivity.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    n_jobs : int
        Number of permutations to run in parallel (requires joblib package).
    seed : int or None
        Seed the random number generator for results reproducibility.
    max_step : int
        When connectivity is a n_vertices x n_vertices matrix, specify the
        maximum number of steps between vertices along the second dimension
        (typically time) to be considered connected. This is not used for full
        or None connectivity matrices.
    spatial_exclude : list of int or None
        List of spatial indices to exclude from clustering.
    step_down_p : float
        To perform a step-down-in-jumps test, pass a p-value for clusters to
        exclude from each successive iteration. Default is zero, perform no
        step-down test (since no clusters will be smaller than this value).
        Setting this to a reasonable value, e.g. 0.05, can increase sensitivity
        but costs computation time.
    t_power : float
        Power to raise the statistical values (usually f-values) by before
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
    buffer_size: int or None
        The statistics will be computed for blocks of variables of size
        "buffer_size" at a time. This is option significantly reduces the
        memory requirements when n_jobs > 1 and memory sharing between
        processes is enabled (see set_cache_dir()), as X will be shared
        between processes and each process only needs to allocate space
        for a small block of variables.

    Returns
    -------
    T_obs : array of shape [n_tests]
        T-statistic observed for all variables
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
    n_samples, n_times, n_vertices = X[0].shape

    # convert spatial_exclude before passing on if necessary
    if spatial_exclude is not None:
        exclude = _st_mask_from_s_inds(n_times, n_vertices,
                                       spatial_exclude, True)
    else:
        exclude = None

    # do the heavy lifting
    out = permutation_cluster_test(X, threshold=threshold,
                                   stat_fun=stat_fun, tail=tail,
                                   n_permutations=n_permutations,
                                   connectivity=connectivity, n_jobs=n_jobs,
                                   seed=seed, max_step=max_step,
                                   exclude=exclude, step_down_p=step_down_p,
                                   t_power=t_power, out_type=out_type,
                                   check_disjoint=check_disjoint,
                                   buffer_size=buffer_size)
    return out


spatio_temporal_cluster_test.__test__ = False


def _st_mask_from_s_inds(n_times, n_vertices, vertices, set_as=True):
    """Compute mask to apply to a spatio-temporal connectivity matrix.

    This can be used to include (or exclude) certain spatial coordinates.
    This is useful for excluding certain regions from analysis (e.g.,
    medial wall vertices).

    Parameters
    ----------
    n_times : int
        Number of time points.
    n_vertices : int
        Number of spatial points.
    vertices : list or array of int
        Vertex numbers to set.
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
    """Specify disjoint subsets (e.g., hemispheres) based on connectivity."""
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

    part_clusts = _find_clusters(test, 0, 1, test_conn)[0]
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
    """Reshape cluster masks or indices to be of the correct shape."""
    # format of the bool mask and indices are ndarrays
    if len(clusters) > 0 and isinstance(clusters[0], np.ndarray):
        if clusters[0].dtype == bool:  # format of mask
            clusters = [c.reshape(sample_shape) for c in clusters]
        else:  # format of indices
            clusters = [np.unravel_index(c, sample_shape) for c in clusters]
    return clusters


def summarize_clusters_stc(clu, p_thresh=0.05, tstep=1e-3, tmin=0,
                           subject='fsaverage', vertices=None):
    """Assemble summary SourceEstimate from spatiotemporal cluster results.

    This helps visualizing results from spatio-temporal-clustering
    permutation tests.

    Parameters
    ----------
    clu : tuple
        the output from clustering permutation tests.
    p_thresh : float
        The significance threshold for inclusion of clusters.
    tstep : float
        The temporal difference between two time samples.
    tmin : float | int
        The time of the first sample.
    subject : str
        The name of the subject.
    vertices : list of arrays | None
        The vertex numbers associated with the source space locations. Defaults
        to None. If None, equals ```[np.arange(10242), np.arange(10242)]```.

    Returns
    -------
    out : instance of SourceEstimate
        A summary of the clusters. The first time point in this SourceEstimate
        object is the summation of all the clusters. Subsequent time points
        contain each individual cluster. The magnitude of the activity
        corresponds to the length the cluster spans in time (in samples).
    """
    if vertices is None:
        vertices = [np.arange(10242), np.arange(10242)]

    T_obs, clusters, clu_pvals, _ = clu
    n_times, n_vertices = T_obs.shape
    good_cluster_inds = np.where(clu_pvals < p_thresh)[0]

    #  Build a convenient representation of each cluster, where each
    #  cluster becomes a "time point" in the SourceEstimate
    if len(good_cluster_inds) > 0:
        data = np.zeros((n_vertices, n_times))
        data_summary = np.zeros((n_vertices, len(good_cluster_inds) + 1))
        for ii, cluster_ind in enumerate(good_cluster_inds):
            data.fill(0)
            v_inds = clusters[cluster_ind][1]
            t_inds = clusters[cluster_ind][0]
            data[v_inds, t_inds] = T_obs[t_inds, v_inds]
            # Store a nice visualization of the cluster by summing across time
            data = np.sign(data) * np.logical_not(data == 0) * tstep
            data_summary[:, ii + 1] = 1e3 * np.sum(data, axis=1)
            # Make the first "time point" a sum across all clusters for easy
            # visualization
        data_summary[:, 0] = np.sum(data_summary, axis=1)

        return SourceEstimate(data_summary, vertices, tmin=tmin, tstep=tstep,
                              subject=subject)
    else:
        raise RuntimeError('No significant clusters available. Please adjust '
                           'your threshold or check your statistical '
                           'analysis.')
