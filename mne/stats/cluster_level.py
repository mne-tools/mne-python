#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Thorsten Kranz <thorstenkranz@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Denis Engemann <denis.engemann@gmail.com>
#          Fernando Perez (bin_perm_rep function)
#
# License: Simplified BSD

import numpy as np

from .parametric import f_oneway, ttest_1samp_no_p
from ..parallel import parallel_func
from ..fixes import jit, has_numba
from ..utils import (split_list, logger, verbose, ProgressBar, warn, _pl,
                     check_random_state, _check_option, _validate_type)
from ..source_estimate import (SourceEstimate, VolSourceEstimate,
                               MixedSourceEstimate)
from ..source_space import SourceSpaces


def _get_buddies_fallback(r, s, neighbors, indices=None):
    if indices is None:
        buddies = np.where(r)[0]
    else:
        buddies = indices[r[indices]]
    buddies = buddies[np.in1d(s[buddies], neighbors, assume_unique=True)]
    r[buddies] = False
    return buddies.tolist()


def _get_selves_fallback(r, s, ind, inds, t, t_border, max_step):
    start = t_border[max(t[ind] - max_step, 0)]
    stop = t_border[min(t[ind] + max_step + 1, len(t_border) - 1)]
    indices = inds[start:stop]
    selves = indices[r[indices]]
    selves = selves[s[ind] == s[selves]]
    r[selves] = False
    return selves.tolist()


def _where_first_fallback(x):
    # this is equivalent to np.where(r)[0] for these purposes, but it's
    # a little bit faster. Unfortunately there's no way to tell numpy
    # just to find the first instance (to save checking every one):
    next_ind = int(np.argmax(x))
    if next_ind == 0:
        next_ind = -1
    return next_ind


if has_numba:  # pragma: no cover
    @jit()
    def _get_buddies(r, s, neighbors, indices=None):
        buddies = list()
        # At some point we might be able to use the sorted-ness of s or
        # neighbors to further speed this up
        if indices is None:
            n_check = len(r)
        else:
            n_check = len(indices)
        for ii in range(n_check):
            if indices is None:
                this_idx = ii
            else:
                this_idx = indices[ii]
            if r[this_idx]:
                this_s = s[this_idx]
                for ni in range(len(neighbors)):
                    if this_s == neighbors[ni]:
                        buddies.append(this_idx)
                        r[this_idx] = False
                        break
        return buddies

    @jit()
    def _get_selves(r, s, ind, inds, t, t_border, max_step):
        selves = list()
        start = t_border[max(t[ind] - max_step, 0)]
        stop = t_border[min(t[ind] + max_step + 1, len(t_border) - 1)]
        for ii in range(start, stop):
            this_idx = inds[ii]
            if r[this_idx] and s[ind] == s[this_idx]:
                selves.append(this_idx)
                r[this_idx] = False
        return selves

    @jit()
    def _where_first(x):
        for ii in range(len(x)):
            if x[ii]:
                return ii
        return -1
else:  # pragma: no cover
    # fastest ways we've found with NumPy
    _get_buddies = _get_buddies_fallback
    _get_selves = _get_selves_fallback
    _where_first = _where_first_fallback


@jit()
def _masked_sum(x, c):
    return np.sum(x[c])


@jit()
def _masked_sum_power(x, c, t_power):
    return np.sum(np.sign(x[c]) * np.abs(x[c]) ** t_power)


@jit()
def _sum_cluster_data(data, tstep):
    return np.sign(data) * np.logical_not(data == 0) * tstep


def _get_clusters_spatial(s, neighbors):
    """Form spatial clusters using neighbor lists.

    This is equivalent to _get_components with n_times = 1, with a properly
    reconfigured adjacency matrix (formed as "neighbors" list)
    """
    # s is a vector of spatial indices that are significant, like:
    #     s = np.where(x_in)[0]
    # for x_in representing a single time-instant
    r = np.ones(s.shape, bool)
    clusters = list()
    next_ind = 0 if s.size > 0 else -1
    while next_ind >= 0:
        # put first point in a cluster, adjust remaining
        t_inds = [next_ind]
        r[next_ind] = 0
        icount = 1  # count of nodes in the current cluster
        while icount <= len(t_inds):
            ind = t_inds[icount - 1]
            # look across other vertices
            buddies = _get_buddies(r, s, neighbors[s[ind]])
            t_inds.extend(buddies)
            icount += 1
        next_ind = _where_first(r)
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
    """Directly calculate clusters.

    This uses knowledge that time points are
    only adjacent to immediate neighbors for data organized as time x space.

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
    """Directly calculate clusters.

    This uses knowledge that time points are
    only adjacent to immediate neighbors for data organized as time x space.

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
    inds = np.arange(t_border[0], t_border[n_times])
    next_ind = 0 if s.size > 0 else -1
    while next_ind >= 0:
        # put first point in a cluster, adjust remaining
        t_inds = [next_ind]
        r[next_ind] = False
        icount = 1  # count of nodes in the current cluster
        # look for significant values at the next time point,
        # same sensor, not placed yet, and add those
        while icount <= len(t_inds):
            ind = t_inds[icount - 1]
            selves = _get_selves(r, s, ind, inds, t, t_border, max_step)

            # look at current time point across other vertices
            these_inds = inds[t_border[t[ind]]:t_border[t[ind] + 1]]
            buddies = _get_buddies(r, s, neighbors[s[ind]], these_inds)

            t_inds += buddies + selves
            icount += 1
        next_ind = _where_first(r)
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
        lims = [0]
        if isinstance(row, int):
            row = [row]
            col = [col]
        else:
            order = np.argsort(row)
            row = row[order]
            col = col[order]
            lims += (np.where(np.diff(row) > 0)[0] + 1).tolist()
            lims.append(len(row))

        for start, end in zip(lims[:-1], lims[1:]):
            keepers[row[start]] = np.sort(col[start:end])
        if max_step == 1:
            return _get_clusters_st_1step(keepers, neighbors)
        else:
            return _get_clusters_st_multistep(keepers, neighbors,
                                              max_step)
    else:
        return []


def _get_components(x_in, adjacency, return_list=True):
    """Get connected components from a mask and a adjacency matrix."""
    from scipy import sparse
    if adjacency is False:
        components = np.arange(len(x_in))
    else:
        from scipy.sparse.csgraph import connected_components
        mask = np.logical_and(x_in[adjacency.row], x_in[adjacency.col])
        data = adjacency.data[mask]
        row = adjacency.row[mask]
        col = adjacency.col[mask]
        shape = adjacency.shape
        idx = np.where(x_in)[0]
        row = np.concatenate((row, idx))
        col = np.concatenate((col, idx))
        data = np.concatenate((data, np.ones(len(idx), dtype=data.dtype)))
        adjacency = sparse.coo_matrix((data, (row, col)), shape=shape)
        _, components = connected_components(adjacency)
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


def _find_clusters(x, threshold, tail=0, adjacency=None, max_step=1,
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
    adjacency : scipy.sparse.coo_matrix, None, or list
        Defines adjacency between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        If adjacency is a list, it is assumed that each entry stores the
        indices of the spatial neighbors in a spatio-temporal dataset x.
        Default is None, i.e, a regular lattice adjacency.
        False means no adjacency.
    max_step : int
        If adjacency is a list, this defines the maximal number of steps
        between vertices along the second dimension (typically time) to be
        considered adjacent.
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
    sums : array
        Sum of x values in clusters.
    """
    from scipy import ndimage
    _check_option('tail', tail, [-1, 0, 1])

    x = np.asanyarray(x)

    if not np.isscalar(threshold):
        if not isinstance(threshold, dict):
            raise TypeError('threshold must be a number, or a dict for '
                            'threshold-free cluster enhancement')
        if not all(key in threshold for key in ['start', 'step']):
            raise KeyError('threshold, if dict, must have at least '
                           '"start" and "step"')
        tfce = True
        use_x = x[np.isfinite(x)]
        if use_x.size == 0:
            raise RuntimeError(
                'No finite values found in the observed statistic values')
        if tail == -1:
            if threshold['start'] > 0:
                raise ValueError('threshold["start"] must be <= 0 for '
                                 'tail == -1')
            if threshold['step'] >= 0:
                raise ValueError('threshold["step"] must be < 0 for '
                                 'tail == -1')
            stop = np.min(use_x)
        elif tail == 1:
            stop = np.max(use_x)
        else:  # tail == 0
            stop = max(np.max(use_x), -np.min(use_x))
        del use_x
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

    if tail in [0, 1] and not np.all(np.diff(thresholds) > 0):
        raise ValueError('Thresholds must be monotonically increasing')
    if tail == -1 and not np.all(np.diff(thresholds) < 0):
        raise ValueError('Thresholds must be monotonically decreasing')

    # set these here just in case thresholds == []
    clusters = list()
    sums = list()
    for ti, thresh in enumerate(thresholds):
        # these need to be reset on each run
        clusters = list()
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
                out = _find_clusters_1dir_parts(x, x_in, adjacency,
                                                max_step, partitions, t_power,
                                                ndimage)
                clusters += out[0]
                sums.append(out[1])
        if tfce:
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
    # turn sums into array
    sums = np.concatenate(sums) if sums else np.array([])
    if tfce:
        # each point gets treated independently
        clusters = np.arange(x.size)
        if adjacency is None or adjacency is False:
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


def _find_clusters_1dir_parts(x, x_in, adjacency, max_step, partitions,
                              t_power, ndimage):
    """Deal with partitions, and pass the work to _find_clusters_1dir."""
    if partitions is None:
        clusters, sums = _find_clusters_1dir(x, x_in, adjacency, max_step,
                                             t_power, ndimage)
    else:
        # cluster each partition separately
        clusters = list()
        sums = list()
        for p in range(np.max(partitions) + 1):
            x_i = np.logical_and(x_in, partitions == p)
            out = _find_clusters_1dir(x, x_i, adjacency, max_step, t_power,
                                      ndimage)
            clusters += out[0]
            sums.append(out[1])
        sums = np.concatenate(sums)
    return clusters, sums


def _find_clusters_1dir(x, x_in, adjacency, max_step, t_power, ndimage):
    """Actually call the clustering algorithm."""
    from scipy import sparse
    if adjacency is None:
        labels, n_labels = ndimage.label(x_in)

        if x.ndim == 1:
            # slices
            clusters = ndimage.find_objects(labels, n_labels)
            # equivalent to if len(clusters) == 0 but faster
            if not clusters:
                sums = list()
            else:
                index = list(range(1, n_labels + 1))
                if t_power == 1:
                    sums = ndimage.sum(x, labels, index=index)
                else:
                    sums = ndimage.sum(np.sign(x) * np.abs(x) ** t_power,
                                       labels, index=index)
        else:
            # boolean masks (raveled)
            clusters = list()
            sums = np.empty(n_labels)
            for label in range(n_labels):
                c = labels == label + 1
                clusters.append(c.ravel())
                if t_power == 1:
                    sums[label] = np.sum(x[c])
                else:
                    sums[label] = np.sum(np.sign(x[c]) *
                                         np.abs(x[c]) ** t_power)
    else:
        if x.ndim > 1:
            raise Exception("Data should be 1D when using a adjacency "
                            "to define clusters.")
        if isinstance(adjacency, sparse.spmatrix) or adjacency is False:
            clusters = _get_components(x_in, adjacency)
        elif isinstance(adjacency, list):  # use temporal adjacency
            clusters = _get_clusters_st(x_in, adjacency, max_step)
        else:
            raise ValueError('adjacency must be a sparse matrix or list')
        if t_power == 1:
            sums = [_masked_sum(x, c) for c in clusters]
        else:
            sums = [_masked_sum_power(x, c, t_power) for c in clusters]

    return clusters, np.atleast_1d(sums)


def _cluster_indices_to_mask(components, n_tot):
    """Convert to the old format of clusters, which were bool arrays."""
    for ci, c in enumerate(components):
        components[ci] = np.zeros((n_tot), dtype=bool)
        components[ci][c] = True
    return components


def _cluster_mask_to_indices(components, shape):
    """Convert to the old format of clusters, which were bool arrays."""
    for ci, c in enumerate(components):
        if isinstance(c, np.ndarray):  # mask
            components[ci] = np.where(c.reshape(shape))
        elif isinstance(c, slice):
            components[ci] = np.arange(c.start, c.stop)
        else:
            assert isinstance(c, tuple), type(c)
            c = list(c)  # tuple->list
            for ii, cc in enumerate(c):
                if isinstance(cc, slice):
                    c[ii] = np.arange(cc.start, cc.stop)
                else:
                    c[ii] = np.where(cc)[0]
            components[ci] = tuple(c)
    return components


def _pval_from_histogram(T, H0, tail):
    """Get p-values from stats values given an H0 distribution.

    For each stat compute a p-value as percentile of its statistics
    within all statistics in surrogate data
    """
    # from pct to fraction
    if tail == -1:  # up tail
        pval = np.array([np.mean(H0 <= t) for t in T])
    elif tail == 1:  # low tail
        pval = np.array([np.mean(H0 >= t) for t in T])
    else:  # both tails
        pval = np.array([np.mean(abs(H0) >= abs(t)) for t in T])

    return pval


def _setup_adjacency(adjacency, n_tests, n_times):
    from scipy import sparse
    if not sparse.issparse(adjacency):
        raise ValueError("If adjacency matrix is given, it must be a "
                         "SciPy sparse matrix.")
    if adjacency.shape[0] == n_tests:  # use global algorithm
        adjacency = adjacency.tocoo()
    else:  # use temporal adjacency algorithm
        got_times, mod = divmod(n_tests, adjacency.shape[0])
        if got_times != n_times or mod != 0:
            raise ValueError(
                'adjacency (len %d) must be of the correct size, i.e. be '
                'equal to or evenly divide the number of tests (%d).\n\n'
                'If adjacency was computed for a source space, try using '
                'the fwd["src"] or inv["src"] as some original source space '
                'vertices can be excluded during forward computation'
                % (adjacency.shape[0], n_tests))
        # we claim to only use upper triangular part... not true here
        adjacency = (adjacency + adjacency.transpose()).tocsr()
        adjacency = [
            adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]
            for i in range(len(adjacency.indptr) - 1)]
    return adjacency


def _do_permutations(X_full, slices, threshold, tail, adjacency, stat_fun,
                     max_step, include, partitions, t_power, orders,
                     sample_shape, buffer_size, progress_bar):
    n_samp, n_vars = X_full.shape

    if buffer_size is not None and n_vars <= buffer_size:
        buffer_size = None  # don't use buffer for few variables

    # allocate space for output
    max_cluster_sums = np.empty(len(orders), dtype=np.double)

    if buffer_size is not None:
        # allocate buffer, so we don't need to allocate memory during loop
        X_buffer = [np.empty((len(X_full[s]), buffer_size), dtype=X_full.dtype)
                    for s in slices]

    for seed_idx, order in enumerate(orders):
        # shuffle sample indices
        assert order is not None
        idx_shuffle_list = [order[s] for s in slices]

        if buffer_size is None:
            # shuffle all data at once
            X_shuffle_list = [X_full[idx, :] for idx in idx_shuffle_list]
            t_obs_surr = stat_fun(*X_shuffle_list)
        else:
            # only shuffle a small data buffer, so we need less memory
            t_obs_surr = np.empty(n_vars, dtype=X_full.dtype)

            for pos in range(0, n_vars, buffer_size):
                # number of variables for this loop
                n_var_loop = min(pos + buffer_size, n_vars) - pos

                # fill buffer
                for i, idx in enumerate(idx_shuffle_list):
                    X_buffer[i][:, :n_var_loop] =\
                        X_full[idx, pos: pos + n_var_loop]

                # apply stat_fun and store result
                tmp = stat_fun(*X_buffer)
                t_obs_surr[pos: pos + n_var_loop] = tmp[:n_var_loop]

        # The stat should have the same shape as the samples for no adj.
        if adjacency is None:
            t_obs_surr.shape = sample_shape

        # Find cluster on randomized stats
        out = _find_clusters(t_obs_surr, threshold=threshold, tail=tail,
                             max_step=max_step, adjacency=adjacency,
                             partitions=partitions, include=include,
                             t_power=t_power)
        perm_clusters_sums = out[1]

        if len(perm_clusters_sums) > 0:
            max_cluster_sums[seed_idx] = np.max(perm_clusters_sums)
        else:
            max_cluster_sums[seed_idx] = 0

        progress_bar.update(seed_idx + 1)

    return max_cluster_sums


def _do_1samp_permutations(X, slices, threshold, tail, adjacency, stat_fun,
                           max_step, include, partitions, t_power, orders,
                           sample_shape, buffer_size, progress_bar):
    n_samp, n_vars = X.shape
    assert slices is None  # should be None for the 1 sample case

    if buffer_size is not None and n_vars <= buffer_size:
        buffer_size = None  # don't use buffer for few variables

    # allocate space for output
    max_cluster_sums = np.empty(len(orders), dtype=np.double)

    if buffer_size is not None:
        # allocate a buffer so we don't need to allocate memory in loop
        X_flip_buffer = np.empty((n_samp, buffer_size), dtype=X.dtype)

    for seed_idx, order in enumerate(orders):
        assert isinstance(order, np.ndarray)
        # new surrogate data with specified sign flip
        assert order.size == n_samp  # should be guaranteed by parent
        signs = 2 * order[:, None].astype(int) - 1
        if not np.all(np.equal(np.abs(signs), 1)):
            raise ValueError('signs from rng must be +/- 1')

        if buffer_size is None:
            # be careful about non-writable memmap (GH#1507)
            if X.flags.writeable:
                X *= signs
                # Recompute statistic on randomized data
                t_obs_surr = stat_fun(X)
                # Set X back to previous state (trade memory eff. for CPU use)
                X *= signs
            else:
                t_obs_surr = stat_fun(X * signs)
        else:
            # only sign-flip a small data buffer, so we need less memory
            t_obs_surr = np.empty(n_vars, dtype=X.dtype)

            for pos in range(0, n_vars, buffer_size):
                # number of variables for this loop
                n_var_loop = min(pos + buffer_size, n_vars) - pos

                X_flip_buffer[:, :n_var_loop] =\
                    signs * X[:, pos: pos + n_var_loop]

                # apply stat_fun and store result
                tmp = stat_fun(X_flip_buffer)
                t_obs_surr[pos: pos + n_var_loop] = tmp[:n_var_loop]

        # The stat should have the same shape as the samples for no adj.
        if adjacency is None:
            t_obs_surr.shape = sample_shape

        # Find cluster on randomized stats
        out = _find_clusters(t_obs_surr, threshold=threshold, tail=tail,
                             max_step=max_step, adjacency=adjacency,
                             partitions=partitions, include=include,
                             t_power=t_power)
        perm_clusters_sums = out[1]
        if len(perm_clusters_sums) > 0:
            # get max with sign info
            idx_max = np.argmax(np.abs(perm_clusters_sums))
            max_cluster_sums[seed_idx] = perm_clusters_sums[idx_max]
        else:
            max_cluster_sums[seed_idx] = 0

        progress_bar.update(seed_idx + 1)

    return max_cluster_sums


def bin_perm_rep(ndim, a=0, b=1):
    """Ndim permutations with repetitions of (a,b).

    Returns an array with all the possible permutations with repetitions of
    (0,1) in ndim dimensions.  The array is shaped as (2**ndim,ndim), and is
    ordered with the last index changing fastest.  For examble, for ndim=3:

    Examples
    --------
    >>> bin_perm_rep(3)
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 1, 0],
           [0, 1, 1],
           [1, 0, 0],
           [1, 0, 1],
           [1, 1, 0],
           [1, 1, 1]])
    """
    # Create the leftmost column as 0,0,...,1,1,...
    nperms = 2 ** ndim
    perms = np.empty((nperms, ndim), type(a))
    perms.fill(a)
    half_point = nperms // 2
    perms[half_point:, 0] = b
    # Fill the rest of the table by sampling the previous column every 2 items
    for j in range(1, ndim):
        half_col = perms[::2, j - 1]
        perms[:half_point, j] = half_col
        perms[half_point:, j] = half_col
    # This is equivalent to something like:
    # orders = [np.fromiter(np.binary_repr(s + 1, ndim), dtype=int)
    #           for s in np.arange(2 ** ndim)]
    return perms


def _get_1samp_orders(n_samples, n_permutations, tail, rng):
    """Get the 1samp orders."""
    max_perms = 2 ** (n_samples - (tail == 0)) - 1
    extra = ''
    if isinstance(n_permutations, str):
        if n_permutations != 'all':
            raise ValueError('n_permutations as a string must be "all"')
        n_permutations = max_perms
    n_permutations = int(n_permutations)
    if max_perms < n_permutations:
        # omit first perm b/c accounted for in H0.append() later;
        # convert to binary array representation
        extra = ' (exact test)'
        orders = bin_perm_rep(n_samples)[1:max_perms + 1]
    elif n_samples <= 20:  # fast way to do it for small(ish) n_samples
        orders = rng.choice(max_perms, n_permutations - 1, replace=False)
        orders = [np.fromiter(np.binary_repr(s + 1, n_samples), dtype=int)
                  for s in orders]
    else:  # n_samples >= 64
        # Here we can just use the hash-table (w/collision detection)
        # functionality of a dict to ensure uniqueness
        orders = np.zeros((n_permutations - 1, n_samples), int)
        hashes = {}
        ii = 0
        # in the symmetric case, we should never flip one of the subjects
        # to prevent positive/negative equivalent collisions
        use_samples = n_samples - (tail == 0)
        while ii < n_permutations - 1:
            signs = tuple((rng.uniform(size=use_samples) < 0.5).astype(int))
            if signs not in hashes:
                orders[ii, :use_samples] = signs
                if tail == 0 and rng.uniform() < 0.5:
                    # To undo the non-flipping of the last subject in the
                    # tail == 0 case, half the time we use the positive
                    # last subject, half the time negative last subject
                    orders[ii] = 1 - orders[ii]
                hashes[signs] = None
                ii += 1
    return orders, n_permutations, extra


def _permutation_cluster_test(X, threshold, n_permutations, tail, stat_fun,
                              adjacency, n_jobs, seed, max_step,
                              exclude, step_down_p, t_power, out_type,
                              check_disjoint, buffer_size):
    """Aux Function.

    Note. X is required to be a list. Depending on the length of X
    either a 1 sample t-test or an F test / more sample permutation scheme
    is elicited.
    """
    _check_option('out_type', out_type, ['mask', 'indices'])
    _check_option('tail', tail, [-1, 0, 1])
    if not isinstance(threshold, dict):
        threshold = float(threshold)
        if (tail < 0 and threshold > 0 or tail > 0 and threshold < 0 or
                tail == 0 and threshold < 0):
            raise ValueError('incompatible tail and threshold signs, got '
                             '%s and %s' % (tail, threshold))

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

    if adjacency is not None and adjacency is not False:
        adjacency = _setup_adjacency(adjacency, n_tests, n_times)

    if (exclude is not None) and not exclude.size == n_tests:
        raise ValueError('exclude must be the same shape as X[0]')

    # Step 1: Calculate t-stat for original data
    # -------------------------------------------------------------
    t_obs = stat_fun(*X)
    _validate_type(t_obs, np.ndarray, 'return value of stat_fun')
    logger.info('stat_fun(H1): min=%f max=%f' % (np.min(t_obs), np.max(t_obs)))

    # test if stat_fun treats variables independently
    if buffer_size is not None:
        t_obs_buffer = np.zeros_like(t_obs)
        for pos in range(0, n_tests, buffer_size):
            t_obs_buffer[pos: pos + buffer_size] =\
                stat_fun(*[x[:, pos: pos + buffer_size] for x in X])

        if not np.alltrue(t_obs == t_obs_buffer):
            warn('Provided stat_fun does not treat variables independently. '
                 'Setting buffer_size to None.')
            buffer_size = None

    # The stat should have the same shape as the samples for no adj.
    if t_obs.size != np.prod(sample_shape):
        raise ValueError('t_obs.shape %s provided by stat_fun %s is not '
                         'compatible with the sample shape %s'
                         % (t_obs.shape, stat_fun, sample_shape))
    if adjacency is None or adjacency is False:
        t_obs.shape = sample_shape

    if exclude is not None:
        include = np.logical_not(exclude)
    else:
        include = None

    # determine if adjacency itself can be separated into disjoint sets
    if check_disjoint is True and (adjacency is not None and
                                   adjacency is not False):
        partitions = _get_partitions_from_adjacency(adjacency, n_times)
    else:
        partitions = None
    logger.info('Running initial clustering …')
    out = _find_clusters(t_obs, threshold, tail, adjacency,
                         max_step=max_step, include=include,
                         partitions=partitions, t_power=t_power,
                         show_info=True)
    clusters, cluster_stats = out

    # The stat should have the same shape as the samples
    t_obs.shape = sample_shape

    # For TFCE, return the "adjusted" statistic instead of raw scores
    if isinstance(threshold, dict):
        t_obs = cluster_stats.reshape(t_obs.shape) * np.sign(t_obs)

    logger.info(f'Found {len(clusters)} cluster{_pl(clusters)}')

    # convert clusters to old format
    if adjacency is not None and adjacency is not False:
        # our algorithms output lists of indices by default
        if out_type == 'mask':
            clusters = _cluster_indices_to_mask(clusters, n_tests)
    else:
        # ndimage outputs slices or boolean masks by default
        if out_type == 'indices':
            clusters = _cluster_mask_to_indices(clusters, t_obs.shape)

    # convert our seed to orders
    # check to see if we can do an exact test
    # (for a two-tailed test, we can exploit symmetry to just do half)
    extra = ''
    rng = check_random_state(seed)
    del seed
    if len(X) == 1:  # 1-sample test
        do_perm_func = _do_1samp_permutations
        X_full = X[0]
        slices = None
        orders, n_permutations, extra = _get_1samp_orders(
            n_samples, n_permutations, tail, rng)
    else:
        n_permutations = int(n_permutations)
        do_perm_func = _do_permutations
        X_full = np.concatenate(X, axis=0)
        n_samples_per_condition = [x.shape[0] for x in X]
        splits_idx = np.append([0], np.cumsum(n_samples_per_condition))
        slices = [slice(splits_idx[k], splits_idx[k + 1])
                  for k in range(len(X))]
        orders = [rng.permutation(len(X_full))
                  for _ in range(n_permutations - 1)]
    del rng
    parallel, my_do_perm_func, n_jobs = parallel_func(
        do_perm_func, n_jobs, verbose=False)

    if len(clusters) == 0:
        warn('No clusters found, returning empty H0, clusters, and cluster_pv')
        return t_obs, np.array([]), np.array([]), np.array([])

    # Step 2: If we have some clusters, repeat process on permuted data
    # -------------------------------------------------------------------
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

        with ProgressBar(
            iterable=range(len(orders)),
            mesg=f'Permuting{extra}'
        ) as progress_bar:
            H0 = parallel(
                my_do_perm_func(X_full, slices, threshold, tail, adjacency,
                                stat_fun, max_step, this_include, partitions,
                                t_power, order, sample_shape, buffer_size,
                                progress_bar.subset(idx))
                for idx, order in split_list(orders, n_jobs, idx=True))
        # include original (true) ordering
        if tail == -1:  # up tail
            orig = cluster_stats.min()
        elif tail == 1:
            orig = cluster_stats.max()
        else:
            orig = abs(cluster_stats).max()
        H0.insert(0, [orig])
        H0 = np.concatenate(H0)
        logger.debug('Computing cluster p-values')
        cluster_pv = _pval_from_histogram(cluster_stats, H0, tail)

        # figure out how many new ones will be removed for step-down
        to_remove = np.where(cluster_pv < step_down_p)[0]
        n_removed = to_remove.size - total_removed
        total_removed = to_remove.size
        step_down_include = np.ones(n_tests, dtype=bool)
        for ti in to_remove:
            step_down_include[clusters[ti]] = False
        if adjacency is None and adjacency is not False:
            step_down_include.shape = sample_shape
        n_step_downs += 1
        if step_down_p > 0:
            a_text = 'additional ' if n_step_downs > 1 else ''
            logger.info('Step-down-in-jumps iteration #%i found %i %s'
                        'cluster%s to exclude from subsequent iterations'
                        % (n_step_downs, n_removed, a_text,
                           _pl(n_removed)))

    # The clusters should have the same shape as the samples
    clusters = _reshape_clusters(clusters, sample_shape)
    return t_obs, clusters, cluster_pv, H0


def _check_fun(X, stat_fun, threshold, tail=0, kind='within'):
    """Check the stat_fun and threshold values."""
    from scipy import stats
    if kind == 'within':
        ppf = stats.t.ppf
        if threshold is None:
            if stat_fun is not None and stat_fun is not ttest_1samp_no_p:
                warn('Automatic threshold is only valid for stat_fun=None '
                     '(or ttest_1samp_no_p), got %s' % (stat_fun,))
            p_thresh = 0.05 / (1 + (tail == 0))
            n_samples = len(X)
            threshold = -ppf(p_thresh, n_samples - 1)
            if np.sign(tail) < 0:
                threshold = -threshold
            logger.info("Using a threshold of {:.6f}".format(threshold))
        stat_fun = ttest_1samp_no_p if stat_fun is None else stat_fun
    else:
        assert kind == 'between'
        ppf = stats.f.ppf
        if threshold is None:
            if stat_fun is not None and stat_fun is not f_oneway:
                warn('Automatic threshold is only valid for stat_fun=None '
                     '(or f_oneway), got %s' % (stat_fun,))
            elif tail != 1:
                warn('Ignoring argument "tail", performing 1-tailed F-test')
            p_thresh = 0.05
            dfn = len(X) - 1
            dfd = np.sum([len(x) for x in X]) - len(X)
            threshold = ppf(1. - p_thresh, dfn, dfd)
            logger.info("Using a threshold of {:.6f}".format(threshold))
        stat_fun = f_oneway if stat_fun is None else stat_fun
    return stat_fun, threshold


@verbose
def permutation_cluster_test(
        X, threshold=None, n_permutations=1024, tail=0, stat_fun=None,
        adjacency=None, n_jobs=None, seed=None, max_step=1, exclude=None,
        step_down_p=0, t_power=1, out_type='indices', check_disjoint=False,
        buffer_size=1000, verbose=None):
    """Cluster-level statistical permutation test.

    For a list of :class:`NumPy arrays <numpy.ndarray>` of data,
    calculate some statistics corrected for multiple comparisons using
    permutations and cluster-level correction. Each element of the list ``X``
    should contain the data for one group of observations (e.g., 2D arrays for
    time series, 3D arrays for time-frequency power values). Permutations are
    generated with random partitions of the data. For details, see
    :footcite:p:`MarisOostenveld2007,Sassenhagen2019`.

    Parameters
    ----------
    X : list of array, shape (n_observations, p[, q][, r])
        The data to be clustered. Each array in ``X`` should contain the
        observations for one group. The first dimension of each array is the
        number of observations from that group; remaining dimensions comprise
        the size of a single observation. For example if ``X = [X1, X2]``
        with ``X1.shape = (20, 50, 4)`` and ``X2.shape = (17, 50, 4)``, then
        ``X`` has 2 groups with respectively 20 and 17 observations in each,
        and each data point is of shape ``(50, 4)``. Note: that the
        *last dimension* of each element of ``X`` should correspond to the
        dimension represented in the ``adjacency`` parameter
        (e.g., spectral data should be provided as
        ``(observations, frequencies, channels/vertices)``).
    %(threshold_clust_f)s
    %(n_permutations_clust_int)s
    %(tail_clust)s
    %(stat_fun_clust_f)s
    %(adjacency_clust_n)s
    %(n_jobs)s
    %(seed)s
    %(max_step_clust)s
    %(exclude_clust)s
    %(step_down_p_clust)s
    %(f_power_clust)s
    %(out_type_clust)s
    %(check_disjoint_clust)s
    %(buffer_size_clust)s
    %(verbose)s

    Returns
    -------
    F_obs : array, shape (p[, q][, r])
        Statistic (F by default) observed for all variables.
    clusters : list
        List type defined by out_type above.
    cluster_pv : array
        P-value for each cluster.
    H0 : array, shape (n_permutations,)
        Max cluster level stats observed under permutation.

    Notes
    -----
    %(threshold_clust_f_notes)s

    References
    ----------
    .. footbibliography::
    """
    stat_fun, threshold = _check_fun(X, stat_fun, threshold, tail, 'between')
    return _permutation_cluster_test(
        X=X, threshold=threshold, n_permutations=n_permutations, tail=tail,
        stat_fun=stat_fun, adjacency=adjacency, n_jobs=n_jobs, seed=seed,
        max_step=max_step, exclude=exclude, step_down_p=step_down_p,
        t_power=t_power, out_type=out_type, check_disjoint=check_disjoint,
        buffer_size=buffer_size)


@verbose
def permutation_cluster_1samp_test(
        X, threshold=None, n_permutations=1024, tail=0, stat_fun=None,
        adjacency=None, n_jobs=None, seed=None, max_step=1,
        exclude=None, step_down_p=0, t_power=1, out_type='indices',
        check_disjoint=False, buffer_size=1000, verbose=None):
    """Non-parametric cluster-level paired t-test.

    For details, see :footcite:p:`MarisOostenveld2007,Sassenhagen2019`.

    Parameters
    ----------
    X : array, shape (n_observations, p[, q][, r])
        The data to be clustered. The first dimension should correspond to the
        difference between paired samples (observations) in two conditions.
        The subarrays ``X[k]`` can be 1D (e.g., time series), 2D (e.g.,
        time series over channels), or 3D (e.g., time-frequencies over
        channels) associated with the kth observation. For spatiotemporal data,
        see also :func:`mne.stats.spatio_temporal_cluster_1samp_test`.
    %(threshold_clust_t)s
    %(n_permutations_clust_all)s
    %(tail_clust)s
    %(stat_fun_clust_t)s
    %(adjacency_clust_1)s
    %(n_jobs)s
    %(seed)s
    %(max_step_clust)s
    %(exclude_clust)s
    %(step_down_p_clust)s
    %(t_power_clust)s
    %(out_type_clust)s
    %(check_disjoint_clust)s
    %(buffer_size_clust)s
    %(verbose)s

    Returns
    -------
    t_obs : array, shape (p[, q][, r])
        T-statistic observed for all variables.
    clusters : list
        List type defined by out_type above.
    cluster_pv : array
        P-value for each cluster.
    H0 : array, shape (n_permutations,)
        Max cluster level stats observed under permutation.

    Notes
    -----
    From an array of paired observations, e.g. a difference in signal
    amplitudes or power spectra in two conditions, calculate if the data
    distributions in the two conditions are significantly different.
    The procedure uses a cluster analysis with permutation test
    for calculating corrected p-values. Randomized data are generated with
    random sign flips. See :footcite:`MarisOostenveld2007` for more
    information.

    Because a 1-sample t-test on the difference in observations is
    mathematically equivalent to a paired t-test, internally this function
    computes a 1-sample t-test (by default) and uses sign flipping (always)
    to perform permutations. This might not be suitable for the case where
    there is truly a single observation under test; see :ref:`disc-stats`.
    %(threshold_clust_t_notes)s

    If ``n_permutations`` exceeds the maximum number of possible permutations
    given the number of observations, then ``n_permutations`` and ``seed``
    will be ignored since an exact test (full permutation test) will be
    performed (this is the case when
    ``n_permutations >= 2 ** (n_observations - (tail == 0))``).

    If no initial clusters are found because all points in the true
    distribution are below the threshold, then ``clusters``, ``cluster_pv``,
    and ``H0`` will all be empty arrays.

    References
    ----------
    .. footbibliography::
    """
    stat_fun, threshold = _check_fun(X, stat_fun, threshold, tail)
    return _permutation_cluster_test(
        X=[X], threshold=threshold, n_permutations=n_permutations, tail=tail,
        stat_fun=stat_fun, adjacency=adjacency, n_jobs=n_jobs, seed=seed,
        max_step=max_step, exclude=exclude, step_down_p=step_down_p,
        t_power=t_power, out_type=out_type, check_disjoint=check_disjoint,
        buffer_size=buffer_size)


@verbose
def spatio_temporal_cluster_1samp_test(
        X, threshold=None, n_permutations=1024, tail=0,
        stat_fun=None, adjacency=None, n_jobs=None, seed=None,
        max_step=1, spatial_exclude=None, step_down_p=0, t_power=1,
        out_type='indices', check_disjoint=False, buffer_size=1000,
        verbose=None):
    """Non-parametric cluster-level paired t-test for spatio-temporal data.

    This function provides a convenient wrapper for
    :func:`mne.stats.permutation_cluster_1samp_test`, for use with data
    organized in the form (observations × time × space),
    (observations × frequencies × space), or optionally
    (observations × time × frequencies × space). For details, see
    :footcite:p:`MarisOostenveld2007,Sassenhagen2019`.

    Parameters
    ----------
    X : array, shape (n_observations, p[, q], n_vertices)
        The data to be clustered. The first dimension should correspond to the
        difference between paired samples (observations) in two conditions.
        The second, and optionally third, dimensions correspond to the
        time or time-frequency data. And, the last dimension should be spatial.
    %(threshold_clust_t)s
    %(n_permutations_clust_all)s
    %(tail_clust)s
    %(stat_fun_clust_t)s
    %(adjacency_clust_st1)s
    %(n_jobs)s
    %(seed)s
    %(max_step_clust)s
    spatial_exclude : list of int or None
        List of spatial indices to exclude from clustering.
    %(step_down_p_clust)s
    %(t_power_clust)s
    %(out_type_clust)s
    %(check_disjoint_clust)s
    %(buffer_size_clust)s
    %(verbose)s

    Returns
    -------
    t_obs : array, shape (p[, q], n_vertices)
        T-statistic observed for all variables.
    clusters : list
        List type defined by out_type above.
    cluster_pv : array
        P-value for each cluster.
    H0 : array, shape (n_permutations,)
        Max cluster level stats observed under permutation.

    Notes
    -----
    %(threshold_clust_t_notes)s

    References
    ----------
    .. footbibliography::
    """
    # convert spatial_exclude before passing on if necessary
    if spatial_exclude is not None:
        exclude = _st_mask_from_s_inds(
            np.prod(X.shape[1:-1]), X.shape[-1], spatial_exclude, True)
    else:
        exclude = None
    return permutation_cluster_1samp_test(
        X, threshold=threshold, stat_fun=stat_fun, tail=tail,
        n_permutations=n_permutations, adjacency=adjacency,
        n_jobs=n_jobs, seed=seed, max_step=max_step, exclude=exclude,
        step_down_p=step_down_p, t_power=t_power, out_type=out_type,
        check_disjoint=check_disjoint, buffer_size=buffer_size)


@verbose
def spatio_temporal_cluster_test(
        X, threshold=None, n_permutations=1024, tail=0, stat_fun=None,
        adjacency=None, n_jobs=None, seed=None, max_step=1,
        spatial_exclude=None, step_down_p=0, t_power=1, out_type='indices',
        check_disjoint=False, buffer_size=1000,
        verbose=None):
    """Non-parametric cluster-level test for spatio-temporal data.

    This function provides a convenient wrapper for
    :func:`mne.stats.permutation_cluster_test`, for use with data
    organized in the form (observations × time × space),
    (observations × time × space), or optionally
    (observations × time × frequencies × space). For more information,
    see :footcite:p:`MarisOostenveld2007,Sassenhagen2019`.

    Parameters
    ----------
    X : list of array, shape (n_observations, p[, q], n_vertices)
        The data to be clustered. Each array in ``X`` should contain the
        observations for one group. The first dimension of each array is the
        number of observations from that group (and may vary between groups).
        The second, and optionally third, dimensions correspond to the
        time or time-frequency data. And, the last dimension should be spatial.
        All dimensions except the first should match across all groups.
    %(threshold_clust_f)s
    %(n_permutations_clust_int)s
    %(tail_clust)s
    %(stat_fun_clust_f)s
    %(adjacency_clust_stn)s
    %(n_jobs)s
    %(seed)s
    %(max_step_clust)s
    spatial_exclude : list of int or None
        List of spatial indices to exclude from clustering.
    %(step_down_p_clust)s
    %(f_power_clust)s
    %(out_type_clust)s
    %(check_disjoint_clust)s
    %(buffer_size_clust)s
    %(verbose)s

    Returns
    -------
    F_obs : array, shape (p[, q], n_vertices)
        Statistic (F by default) observed for all variables.
    clusters : list
        List type defined by out_type above.
    cluster_pv: array
        P-value for each cluster.
    H0 : array, shape (n_permutations,)
        Max cluster level stats observed under permutation.

    Notes
    -----
    %(threshold_clust_f_notes)s

    References
    ----------
    .. footbibliography::
    """
    # convert spatial_exclude before passing on if necessary
    if spatial_exclude is not None:
        exclude = _st_mask_from_s_inds(
            np.prod(X[0].shape[1:-1]), X[0].shape[-1], spatial_exclude, True)
    else:
        exclude = None
    return permutation_cluster_test(
        X, threshold=threshold, stat_fun=stat_fun, tail=tail,
        n_permutations=n_permutations, adjacency=adjacency,
        n_jobs=n_jobs, seed=seed, max_step=max_step, exclude=exclude,
        step_down_p=step_down_p, t_power=t_power, out_type=out_type,
        check_disjoint=check_disjoint, buffer_size=buffer_size)


def _st_mask_from_s_inds(n_times, n_vertices, vertices, set_as=True):
    """Compute mask to apply to a spatio-temporal adjacency matrix.

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
def _get_partitions_from_adjacency(adjacency, n_times, verbose=None):
    """Specify disjoint subsets (e.g., hemispheres) based on adjacency."""
    from scipy import sparse
    if isinstance(adjacency, list):
        test = np.ones(len(adjacency))
        test_adj = np.zeros((len(adjacency), len(adjacency)), dtype='bool')
        for vi in range(len(adjacency)):
            test_adj[adjacency[vi], vi] = True
        test_adj = sparse.coo_matrix(test_adj, dtype='float')
    else:
        test = np.ones(adjacency.shape[0])
        test_adj = adjacency

    part_clusts = _find_clusters(test, 0, 1, test_adj)[0]
    if len(part_clusts) > 1:
        logger.info('%i disjoint adjacency sets found'
                    % len(part_clusts))
        partitions = np.zeros(len(test), dtype='int')
        for ii, pc in enumerate(part_clusts):
            partitions[pc] = ii
        if isinstance(adjacency, list):
            partitions = np.tile(partitions, n_times)
    else:
        logger.info('No disjoint adjacency sets found')
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


def summarize_clusters_stc(clu, p_thresh=0.05, tstep=1.0, tmin=0,
                           subject='fsaverage', vertices=None):
    """Assemble summary SourceEstimate from spatiotemporal cluster results.

    This helps visualizing results from spatio-temporal-clustering
    permutation tests.

    Parameters
    ----------
    clu : tuple
        The output from clustering permutation tests.
    p_thresh : float
        The significance threshold for inclusion of clusters.
    tstep : float
        The time step between samples of the original :class:`STC
        <mne.SourceEstimate>`, in seconds (i.e., ``1 / stc.sfreq``). Defaults
        to ``1``, which will yield a colormap indicating cluster duration
        measured in *samples* rather than *seconds*.
    tmin : float | int
        The time of the first sample.
    subject : str
        The name of the subject.
    vertices : list of array | instance of SourceSpaces | None
        The vertex numbers associated with the source space locations. Defaults
        to None. If None, equals ``[np.arange(10242), np.arange(10242)]``.
        Can also be an instance of SourceSpaces to get vertex numbers from.

        .. versionchanged:: 0.21
           Added support for SourceSpaces.

    Returns
    -------
    out : instance of SourceEstimate
        A summary of the clusters. The first time point in this SourceEstimate
        object is the summation of all the clusters. Subsequent time points
        contain each individual cluster. The magnitude of the activity
        corresponds to the duration spanned by the cluster (duration units are
        determined by ``tstep``).

        .. versionchanged:: 0.21
           Added support for volume and mixed source estimates.
    """
    _validate_type(vertices, (None, list, SourceSpaces), 'vertices')
    if vertices is None:
        vertices = [np.arange(10242), np.arange(10242)]
        klass = SourceEstimate
    elif isinstance(vertices, SourceSpaces):
        klass = dict(surface=SourceEstimate,
                     volume=VolSourceEstimate,
                     mixed=MixedSourceEstimate)[vertices.kind]
        vertices = [s['vertno'] for s in vertices]
    else:
        klass = {1: VolSourceEstimate,
                 2: SourceEstimate}.get(len(vertices), MixedSourceEstimate)
    n_vertices_need = sum(len(v) for v in vertices)

    t_obs, clusters, clu_pvals, _ = clu
    n_times, n_vertices = t_obs.shape
    if n_vertices != n_vertices_need:
        raise ValueError(
            f'Number of cluster vertices ({n_vertices}) did not match the '
            f'provided vertices ({n_vertices_need})')
    good_cluster_inds = np.where(clu_pvals < p_thresh)[0]

    #  Build a convenient representation of each cluster, where each
    #  cluster becomes a "time point" in the SourceEstimate
    if len(good_cluster_inds) == 0:
        raise RuntimeError('No significant clusters available. Please adjust '
                           'your threshold or check your statistical '
                           'analysis.')
    data = np.zeros((n_vertices, n_times))
    data_summary = np.zeros((n_vertices, len(good_cluster_inds) + 1))
    for ii, cluster_ind in enumerate(good_cluster_inds):
        data.fill(0)
        t_inds, v_inds = clusters[cluster_ind]
        data[v_inds, t_inds] = t_obs[t_inds, v_inds]
        # Store a nice visualization of the cluster by summing across time
        data_summary[:, ii + 1] = np.sum(_sum_cluster_data(data, tstep),
                                         axis=1)
        # Make the first "time point" a sum across all clusters for easy
        # visualization
    data_summary[:, 0] = np.sum(data_summary, axis=1)

    return klass(data_summary, vertices, tmin, tstep, subject)
