#!/usr/bin/env python

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from __future__ import annotations

from functools import partial
from string import ascii_uppercase
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..epochs import BaseEpochs
from ..evoked import Evoked, combine_evoked
from ..parallel import parallel_func
from ..source_estimate import MixedSourceEstimate, SourceEstimate, VolSourceEstimate
from ..source_space import SourceSpaces
from ..time_frequency import BaseTFR, EpochsTFR
from ..utils import (
    GetEpochsMixin,
    ProgressBar,
    _check_option,
    _check_rng,
    _legacy_rng,
    _pl,
    _soft_import,
    _validate_type,
    legacy,
    logger,
    split_list,
    verbose,
    warn,
)
from .parametric import f_mway_rm, f_oneway, f_threshold_mway_rm, ttest_1samp_no_p

if TYPE_CHECKING:
    from scipy import sparse  # Used in type hints for cluster_test

# need this at top-level of file due to type hints
pd = _soft_import("pandas", purpose="DataFrame integration", strict=False)
DataFrame = getattr(pd, "DataFrame", None)


def _get_labels_st(x_in, adjacency, max_step):
    """Label connected components among the active spatio-temporal points.

    Only the supra-threshold ("active") points are ever placed in the graph
    passed to ``connected_components``, so its size scales with the number
    of active points rather than with the full ``n_times * n_src`` extent
    of ``x_in`` -- important since this is called on every permutation.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    n_src = adjacency.shape[0]
    n_total = len(x_in)
    active = np.where(x_in)[0]
    if len(active) == 0:
        return active, None

    indptr = adjacency.indptr
    indices = adjacency.indices

    active_t, active_s = np.divmod(active, n_src)

    # Spatial edges: vectorized CSR neighbor expansion
    neighbor_counts = indptr[active_s + 1] - indptr[active_s]
    src_flat = np.repeat(active, neighbor_counts)
    src_t = np.repeat(active_t, neighbor_counts)
    starts = indptr[active_s]
    offsets = np.arange(int(np.sum(neighbor_counts))) - np.repeat(
        np.cumsum(neighbor_counts) - neighbor_counts, neighbor_counts
    )
    nb_s = indices[np.repeat(starts, neighbor_counts) + offsets]
    nb_flat = src_t * n_src + nb_s
    mask = x_in[nb_flat]
    rows = [src_flat[mask]]
    cols = [nb_flat[mask]]

    # Temporal edges: same source, adjacent time steps
    for step in range(1, max_step + 1):
        mask_t = active_t >= step
        later = active[mask_t]
        earlier = later - step * n_src
        both = x_in[earlier]
        rows.extend([later[both], earlier[both]])
        cols.extend([earlier[both], later[both]])

    # Self-loops so isolated active vertices get their own component
    rows.append(active)
    cols.append(active)
    row = np.concatenate(rows)
    col = np.concatenate(cols)

    # Remap to a compact 0..len(active)-1 index space before building the
    # graph, so connected_components only has to traverse the (typically
    # much smaller) active subgraph instead of all n_total vertices.
    remap = np.empty(n_total, dtype=np.intp)
    remap[active] = np.arange(len(active))
    n_active = len(active)
    adj = sparse.coo_array(
        (np.ones(len(row)), (remap[row], remap[col])), shape=(n_active, n_active)
    )
    _, labels = connected_components(adj)
    return active, labels


def _labels_to_clusters(active, labels):
    """Group active indices by component label into a list of index arrays."""
    # A stable sort keeps clusters in ascending label order and indices in
    # ascending order within each cluster, i.e., the same output as masking
    # once per label, but without the O(n_active * n_clusters) cost.
    order = np.argsort(labels, kind="stable")
    active = active[order]
    labels = labels[order]
    return np.split(active, np.flatnonzero(np.diff(labels)) + 1)


def _get_clusters_st(x_in, adjacency, max_step=1):
    """Find spatio-temporal clusters via SciPy connected components."""
    active, labels = _get_labels_st(x_in, adjacency, max_step)
    if labels is None:
        return []
    return _labels_to_clusters(active, labels)


def _get_cluster_sums_st(x, x_in, adjacency, max_step, t_power):
    """Like _get_clusters_st, but return only the per-cluster sums of x."""
    active, labels = _get_labels_st(x_in, adjacency, max_step)
    if labels is None:
        return np.array([])
    weights = x[active]
    if t_power != 1:
        weights = np.sign(weights) * np.abs(weights) ** t_power
    return np.bincount(labels, weights=weights)


def _get_labels(x_in, adjacency):
    """Label connected components among the active points of a global adjacency.

    Same idea as :func:`_get_labels_st`, but for a plain (non spatio-temporal)
    sparse adjacency matrix that already spans all of ``x_in``.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    active = np.where(x_in)[0]
    if len(active) == 0:
        return active, None
    if adjacency is False:
        # no adjacency: every active point is its own cluster
        return active, np.arange(len(active))
    mask = np.logical_and(x_in[adjacency.row], x_in[adjacency.col])
    row = adjacency.row[mask]
    col = adjacency.col[mask]
    data = adjacency.data[mask]
    # Self-loops so isolated active vertices get their own component
    row = np.concatenate((row, active))
    col = np.concatenate((col, active))
    data = np.concatenate((data, np.ones(len(active), dtype=data.dtype)))
    remap = np.empty(len(x_in), dtype=np.intp)
    remap[active] = np.arange(len(active))
    n_active = len(active)
    adj = sparse.coo_array((data, (remap[row], remap[col])), shape=(n_active, n_active))
    _, labels = connected_components(adj)
    return active, labels


def _get_components(x_in, adjacency):
    """Get connected components from a mask and a adjacency matrix."""
    active, labels = _get_labels(x_in, adjacency)
    if labels is None:
        return []
    return _labels_to_clusters(active, labels)


def _get_cluster_sums(x, x_in, adjacency, t_power):
    """Like _get_components, but return only the per-cluster sums of x."""
    active, labels = _get_labels(x_in, adjacency)
    if labels is None:
        return np.array([])
    weights = x[active]
    if t_power != 1:
        weights = np.sign(weights) * np.abs(weights) ** t_power
    return np.bincount(labels, weights=weights)


def _find_clusters(
    x,
    threshold,
    tail=0,
    adjacency=None,
    max_step=1,
    include=None,
    partitions=None,
    t_power=1,
    show_info=False,
    sums_only=False,
):
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
    adjacency : scipy.sparse.coo_array, scipy.sparse.csr_array, None, or False
        Defines adjacency between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        If the adjacency is smaller than ``x``, it is assumed to be a
        spatial-only adjacency that should be applied at each step along
        the second (e.g., time) dimension of a spatio-temporal dataset x.
        Default is None, i.e, a regular lattice adjacency.
        False means no adjacency.
    max_step : int
        If adjacency is spatial-only (see above), this defines the maximal
        number of steps between vertices along the second dimension
        (typically time) to be considered adjacent.
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
    sums_only : bool
        If True, skip building the cluster index-array list and return only
        the per-cluster sums (``clusters`` will be ``None``). Ignored for
        TFCE (a dict ``threshold``), which always needs the cluster indices.
        Used to speed up the permutation loop, where only the maximum
        per-permutation cluster statistic is needed.

    Returns
    -------
    clusters : list of slices or list of arrays (boolean masks)
        We use slices for 1D signals and mask to multidimensional
        arrays. None is returned if threshold is a dict (TFCE) or if
        ``sums_only`` is True.
    sums : array
        Sum of x values in clusters.
    """
    from scipy import ndimage

    _check_option("tail", tail, [-1, 0, 1])

    x = np.asanyarray(x)

    if not np.isscalar(threshold):
        if not isinstance(threshold, dict):
            raise TypeError(
                "threshold must be a number, or a dict for "
                "threshold-free cluster enhancement"
            )
        if not all(key in threshold for key in ["start", "step"]):
            raise KeyError('threshold, if dict, must have at least "start" and "step"')
        tfce = True
        use_x = x[np.isfinite(x)]
        if use_x.size == 0:
            raise RuntimeError(
                "No finite values found in the observed statistic values"
            )
        if tail == -1:
            if threshold["start"] > 0:
                raise ValueError('threshold["start"] must be <= 0 for tail == -1')
            if threshold["step"] >= 0:
                raise ValueError('threshold["step"] must be < 0 for tail == -1')
            stop = np.min(use_x)
        elif tail == 1:
            stop = np.max(use_x)
        else:  # tail == 0
            stop = max(np.max(use_x), -np.min(use_x))
        del use_x
        thresholds = np.arange(threshold["start"], stop, threshold["step"], float)
        h_power = threshold.get("h_power", 2)
        e_power = threshold.get("e_power", 0.5)
        if show_info is True:
            if len(thresholds) == 0:
                warn(
                    f'threshold["start"] ({threshold["start"]}) is more extreme '
                    f"than data statistics with most extreme value {stop}"
                )
            else:
                logger.info(
                    "Using %d thresholds from %0.2f to %0.2f for TFCE "
                    "computation (h_power=%0.2f, e_power=%0.2f)",
                    len(thresholds),
                    thresholds[0],
                    thresholds[-1],
                    h_power,
                    e_power,
                )
        scores = np.zeros(x.size)
    else:
        thresholds = [threshold]
        tfce = False
    # TFCE always needs the actual cluster indices to accumulate scores
    sums_only = sums_only and not tfce

    # include all points by default
    if include is None:
        include = np.ones(x.shape, dtype=bool)

    if tail in [0, 1] and not np.all(np.diff(thresholds) > 0):
        raise ValueError("Thresholds must be monotonically increasing")
    if tail == -1 and not np.all(np.diff(thresholds) < 0):
        raise ValueError("Thresholds must be monotonically decreasing")

    # set these here just in case thresholds == []
    clusters = list()
    sums = list()
    for ti, thresh in enumerate(thresholds):
        # these need to be reset on each run
        clusters = None if sums_only else list()
        if tail == 0:
            x_ins = [
                np.logical_and(x > thresh, include),
                np.logical_and(x < -thresh, include),
            ]
        elif tail == -1:
            x_ins = [np.logical_and(x < thresh, include)]
        else:  # tail == 1
            x_ins = [np.logical_and(x > thresh, include)]
        # loop over tails
        for x_in in x_ins:
            if np.any(x_in):
                out = _find_clusters_1dir_parts(
                    x,
                    x_in,
                    adjacency,
                    max_step,
                    partitions,
                    t_power,
                    ndimage,
                    sums_only,
                )
                if not sums_only:
                    clusters += out[0]
                sums.append(out[1])
        if tfce:
            # the score of each point is the sum of the h^H * e^E for each
            # supporting section "rectangle" h x e.
            if ti == 0:
                h = abs(thresh)
            else:
                h = abs(thresh - thresholds[ti - 1])
            h = h**h_power
            for c in clusters:
                # triage based on cluster storage type
                if isinstance(c, slice):
                    len_c = c.stop - c.start
                elif isinstance(c, tuple):
                    len_c = len(c)
                elif c.dtype == np.dtype(bool):
                    len_c = np.sum(c)
                else:
                    len_c = len(c)
                scores[c] += h * (len_c**e_power)
    # turn sums into array
    sums = np.concatenate(sums) if sums else np.array([])
    if tfce:
        sums = scores
        clusters = None  # clusters construction is made in _permutation_cluster_test

    return clusters, sums


def _find_clusters_1dir_parts(
    x, x_in, adjacency, max_step, partitions, t_power, ndimage, sums_only=False
):
    """Deal with partitions, and pass the work to _find_clusters_1dir."""
    if partitions is None:
        clusters, sums = _find_clusters_1dir(
            x, x_in, adjacency, max_step, t_power, ndimage, sums_only
        )
    else:
        # cluster each partition separately
        clusters = None if sums_only else list()
        sums = list()
        for p in range(np.max(partitions) + 1):
            x_i = np.logical_and(x_in, partitions == p)
            out = _find_clusters_1dir(
                x, x_i, adjacency, max_step, t_power, ndimage, sums_only
            )
            if not sums_only:
                clusters += out[0]
            sums.append(out[1])
        sums = np.concatenate(sums)
    return clusters, sums


def _find_clusters_1dir(
    x, x_in, adjacency, max_step, t_power, ndimage, sums_only=False
):
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
                    sums = ndimage.sum(
                        np.sign(x) * np.abs(x) ** t_power, labels, index=index
                    )
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
                    sums[label] = np.sum(np.sign(x[c]) * np.abs(x[c]) ** t_power)
    else:
        if x.ndim > 1:
            raise Exception(
                "Data should be 1D when using a adjacency to define clusters."
            )
        if adjacency is False or adjacency.shape[0] == x_in.size:
            # global adjacency spans the whole (flattened) data;
            # _get_components/_get_cluster_sums need COO's .row/.col attributes
            if adjacency is not False and adjacency.format != "coo":
                adjacency = sparse.coo_array(adjacency)
            if sums_only:
                return None, _get_cluster_sums(x, x_in, adjacency, t_power)
            clusters = _get_components(x_in, adjacency)
        elif sparse.issparse(adjacency):
            # spatial-only adjacency, applied along the second (e.g. time) dim
            if sums_only:
                return None, _get_cluster_sums_st(x, x_in, adjacency, max_step, t_power)
            clusters = _get_clusters_st(x_in, adjacency, max_step)
        else:
            raise TypeError(
                f"adjacency must be a sparse array or False, got {type(adjacency)}"
            )
        from ._cluster_level_numba import _masked_sum, _masked_sum_power

        if t_power == 1:
            sums = [_masked_sum(x, c) for c in clusters]
        else:
            sums = [_masked_sum_power(x, c, t_power) for c in clusters]

    return clusters, np.atleast_1d(sums)


def _cluster_indices_to_mask(components, n_tot, slice_out):
    """Convert to the old format of clusters, which were bool arrays (or slices in 1D)."""  # noqa: E501
    for ci, c in enumerate(components):
        if not slice_out:
            # boolean array
            components[ci] = np.zeros((n_tot), dtype=bool)
            components[ci][c] = True
        else:
            # slice (similar as ndimage.find_object output)
            components[ci] = (slice(c.min(), c.max() + 1),)
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
        raise ValueError(
            "If adjacency matrix is given, it must be a SciPy sparse matrix."
        )
    if adjacency.shape[0] == n_tests:  # use global algorithm
        adjacency = adjacency.tocoo()
    else:  # use temporal adjacency algorithm
        got_times, mod = divmod(n_tests, adjacency.shape[0])
        if got_times != n_times or mod != 0:
            raise ValueError(
                f"adjacency (len {adjacency.shape[0]}) must be of the correct size, "
                "i.e. be equal to or evenly divide the number of tests ({n_tests}).\n\n"
                "If adjacency was computed for a source space, try using "
                'the fwd["src"] or inv["src"] as some original source space '
                "vertices can be excluded during forward computation"
            )
        # we claim to only use upper triangular part... not true here
        adjacency = (adjacency + adjacency.transpose()).tocsr()
    return adjacency


def _do_permutations(
    X_full,
    slices,
    threshold,
    tail,
    adjacency,
    stat_fun,
    max_step,
    include,
    partitions,
    t_power,
    orders,
    sample_shape,
    buffer_size,
    progress_bar,
):
    n_samp, n_vars = X_full.shape

    if buffer_size is not None and n_vars <= buffer_size:
        buffer_size = None  # don't use buffer for few variables

    # allocate space for output
    max_cluster_sums = np.empty(len(orders), dtype=np.double)

    if buffer_size is not None:
        # allocate buffer, so we don't need to allocate memory during loop
        X_buffer = [
            np.empty((len(X_full[s]), buffer_size), dtype=X_full.dtype) for s in slices
        ]

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
                    X_buffer[i][:, :n_var_loop] = X_full[idx, pos : pos + n_var_loop]

                # apply stat_fun and store result
                tmp = stat_fun(*X_buffer)
                t_obs_surr[pos : pos + n_var_loop] = tmp[:n_var_loop]

        # The stat should have the same shape as the samples for no adj.
        if adjacency is None:
            t_obs_surr = t_obs_surr.reshape(sample_shape, copy=False)

        # Find cluster on randomized stats (only the max cluster sum is
        # needed here, so skip building the cluster index-array list)
        out = _find_clusters(
            t_obs_surr,
            threshold=threshold,
            tail=tail,
            max_step=max_step,
            adjacency=adjacency,
            partitions=partitions,
            include=include,
            t_power=t_power,
            sums_only=True,
        )
        perm_clusters_sums = out[1]

        if len(perm_clusters_sums) > 0:
            max_cluster_sums[seed_idx] = np.max(perm_clusters_sums)
        else:
            max_cluster_sums[seed_idx] = 0

        progress_bar.update(seed_idx + 1)

    return max_cluster_sums


class _TTestReordered:
    """1-sample t-test that reuses a precomputed sum of squares.

    For sign-flip permutations, s**2 == 1, so sum(X ** 2) across samples is
    invariant to the permutation and only needs to be computed once. Each
    call then only needs the (already sign-flipped) column sums, so this
    can be used as a drop-in replacement for :func:`ttest_1samp_no_p` in
    the permutation loop below.
    """

    def __init__(self, X):
        self._n = X.shape[0]
        self._sum_sq = np.sum(X**2, axis=0)
        self._sqrt_n_nm1 = np.sqrt(self._n * (self._n - 1))

    def __call__(self, X):
        return self._stat(np.sum(X, axis=0))

    def from_signs(self, signs, X):
        """Compute the statistic from a +/-1 sign vector and unflipped data.

        Equivalent to ``self(X * signs[:, None])``, but ``signs @ X`` lets
        BLAS reduce over samples directly to a length-n_vars vector, so we
        never materialize a full sign-flipped copy of (or mutate) X.
        """
        return self._stat(signs @ X)

    def _stat(self, col_sum):
        mean = col_sum / self._n
        denom_sq = np.maximum(self._sum_sq - self._n * mean * mean, 0.0)
        # avoid divide-by-zero warnings for degenerate (zero-variance) columns
        out = np.zeros_like(mean)
        mask = denom_sq > 0
        out[mask] = mean[mask] / np.sqrt(denom_sq[mask]) * self._sqrt_n_nm1
        return out


def _do_1samp_permutations(
    X,
    slices,
    threshold,
    tail,
    adjacency,
    stat_fun,
    max_step,
    include,
    partitions,
    t_power,
    orders,
    sample_shape,
    buffer_size,
    progress_bar,
):
    n_samp, n_vars = X.shape
    assert slices is None  # should be None for the 1 sample case

    # _TTestReordered is already as efficient as possible unbuffered
    if buffer_size is not None and (
        n_vars <= buffer_size or isinstance(stat_fun, _TTestReordered)
    ):
        buffer_size = None  # don't use buffer for few variables

    # allocate space for output
    max_cluster_sums = np.empty(len(orders), dtype=np.double)

    if buffer_size is not None:
        # allocate a buffer so we don't need to allocate memory in loop
        X_flip_buffer = np.empty((n_samp, buffer_size), dtype=X.dtype)

    for seed_idx, order in enumerate(orders):
        assert isinstance(order, np.ndarray)
        assert order.size == n_samp  # should be guaranteed by parent

        if isinstance(stat_fun, _TTestReordered):
            # signs @ X is reduced by BLAS directly to a length-n_vars
            # vector, so there's no full-size sign-flipped copy to make (or
            # in-place mutation of X to undo afterward)
            t_obs_surr = stat_fun.from_signs(2.0 * order - 1.0, X)
        else:
            # new surrogate data with specified sign flip
            signs = 2 * order[:, None].astype(int) - 1

            if buffer_size is None:
                # be careful about non-writable memmap (GH#1507)
                if X.flags.writeable:
                    X *= signs
                    try:
                        # Recompute statistic on randomized data
                        t_obs_surr = stat_fun(X)
                    finally:
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

                    X_flip_buffer[:, :n_var_loop] = signs * X[:, pos : pos + n_var_loop]

                    # apply stat_fun and store result
                    tmp = stat_fun(X_flip_buffer)
                    t_obs_surr[pos : pos + n_var_loop] = tmp[:n_var_loop]

        # The stat should have the same shape as the samples for no adj.
        if adjacency is None:
            t_obs_surr = t_obs_surr.reshape(sample_shape, copy=False)

        # Find cluster on randomized stats (only the max cluster sum is
        # needed here, so skip building the cluster index-array list)
        out = _find_clusters(
            t_obs_surr,
            threshold=threshold,
            tail=tail,
            max_step=max_step,
            adjacency=adjacency,
            partitions=partitions,
            include=include,
            t_power=t_power,
            sums_only=True,
        )
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
    nperms = 2**ndim
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
    extra = ""
    if isinstance(n_permutations, str):
        if n_permutations != "all":
            raise ValueError('n_permutations as a string must be "all"')
        n_permutations = max_perms
    n_permutations = int(n_permutations)
    if max_perms < n_permutations:
        # omit first perm b/c accounted for in H0.append() later;
        # convert to binary array representation
        extra = " (exact test)"
        orders = bin_perm_rep(n_samples)[1 : max_perms + 1]
    elif n_samples <= 20:  # fast way to do it for small(ish) n_samples
        orders = rng.choice(max_perms, n_permutations - 1, replace=False)
        orders = [
            np.fromiter(np.binary_repr(s + 1, n_samples), dtype=int) for s in orders
        ]
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


def _permutation_cluster_test(
    X,
    threshold,
    n_permutations,
    tail,
    stat_fun,
    adjacency,
    n_jobs,
    rng,
    max_step,
    exclude,
    step_down_p,
    t_power,
    out_type,
    check_disjoint,
    buffer_size,
    within_subject=False,
):
    """Aux Function.

    Note. X is required to be a list. Depending on the length of X
    either a 1 sample t-test or an F test / more sample permutation scheme
    is elicited.

    ``within_subject=True`` restricts multi-group permutations to swapping each
    subject's observations across the groups (repeated-measures designs); rows
    of each element of X must then be aligned by subject.
    """
    _check_option("out_type", out_type, ["mask", "indices"])
    _check_option("tail", tail, [-1, 0, 1])
    if not isinstance(threshold, dict):
        threshold = float(threshold)
        if (
            tail < 0
            and threshold > 0
            or tail > 0
            and threshold < 0
            or tail == 0
            and threshold < 0
        ):
            raise ValueError(
                f"incompatible tail and threshold signs, got {tail} and {threshold}"
            )

    # check dimensions for each group in X (a list at this stage).
    X = [x[:, np.newaxis] if x.ndim == 1 else x for x in X]
    n_samples = X[0].shape[0]
    n_times = X[0].shape[1]

    sample_shape = X[0].shape[1:]
    for x in X:
        if x.shape[1:] != sample_shape:
            raise ValueError("All samples must have the same size")

    # flatten the last dimensions in case the data is high dimensional
    X = [np.reshape(x, (x.shape[0], -1)) for x in X]
    n_tests = X[0].shape[1]

    if adjacency is not None and adjacency is not False:
        adjacency = _setup_adjacency(adjacency, n_tests, n_times)

    if (exclude is not None) and not exclude.size == n_tests:
        raise ValueError("exclude must be the same shape as X[0]")

    # Step 1: Calculate t-stat for original data
    # -------------------------------------------------------------
    t_obs = stat_fun(*X)
    _validate_type(t_obs, np.ndarray, "return value of stat_fun")
    logger.info(f"stat_fun(H1): min={np.min(t_obs)} max={np.max(t_obs)}")

    # test if stat_fun treats variables independently
    # (skip for built-in stat functions which are known to be independent)
    if buffer_size is not None and (
        stat_fun is not ttest_1samp_no_p and stat_fun is not f_oneway
    ):
        t_obs_buffer = np.zeros_like(t_obs)
        for pos in range(0, n_tests, buffer_size):
            t_obs_buffer[pos : pos + buffer_size] = stat_fun(
                *[x[:, pos : pos + buffer_size] for x in X]
            )

        if not np.all(t_obs == t_obs_buffer):
            warn(
                "Provided stat_fun does not treat variables independently. "
                "Setting buffer_size to None."
            )
            buffer_size = None

    # The stat should have the same shape as the samples for no adj.
    if t_obs.size != np.prod(sample_shape):
        raise ValueError(
            f"t_obs.shape {t_obs.shape} provided by stat_fun {stat_fun} is not "
            f"compatible with the sample shape {sample_shape}"
        )
    if adjacency is None or adjacency is False:
        t_obs = t_obs.reshape(sample_shape, copy=False)

    if exclude is not None:
        include = np.logical_not(exclude)
    else:
        include = None

    # determine if adjacency itself can be separated into disjoint sets
    if check_disjoint is True and (adjacency is not None and adjacency is not False):
        partitions = _get_partitions_from_adjacency(adjacency, n_tests, n_times)
    else:
        partitions = None
    logger.info("Running initial clustering …")
    out = _find_clusters(
        t_obs,
        threshold,
        tail,
        adjacency,
        max_step=max_step,
        include=include,
        partitions=partitions,
        t_power=t_power,
        show_info=True,
    )
    clusters, cluster_stats = out

    # The stat should have the same shape as the samples
    t_obs = t_obs.reshape(sample_shape, copy=False)

    # For TFCE, return the "adjusted" statistic instead of raw scores
    # and for clusters, each point gets treated independently
    tfce = isinstance(threshold, dict)
    if tfce:
        t_obs = cluster_stats.reshape(t_obs.shape) * np.sign(t_obs)
        clusters = [np.array([c]) for c in range(t_obs.size)]

    logger.info(f"Found {len(clusters)} cluster{_pl(clusters)}")

    # convert clusters to old format
    if (adjacency is not None and adjacency is not False) or tfce:
        # our algorithms output lists of indices by default
        if out_type == "mask":
            slice_out = (adjacency is None) & (len(sample_shape) == 1)
            clusters = _cluster_indices_to_mask(clusters, n_tests, slice_out)
    else:
        # ndimage outputs slices or boolean masks by default,
        if out_type == "indices":
            clusters = _cluster_mask_to_indices(clusters, t_obs.shape)

    # Convert the RNG state to permutation orders.
    # check to see if we can do an exact test
    # (for a two-tailed test, we can exploit symmetry to just do half)
    extra = ""
    if len(X) == 1:  # 1-sample test
        do_perm_func = _do_1samp_permutations
        X_full = X[0]
        slices = None
        orders, n_permutations, extra = _get_1samp_orders(
            n_samples, n_permutations, tail, rng
        )
        # For sign-flips, sum(X ** 2) is invariant across permutations, so
        # precompute it once instead of recomputing it on every permutation.
        if stat_fun is ttest_1samp_no_p:
            stat_fun = _TTestReordered(X_full)
    else:
        n_permutations = int(n_permutations)
        do_perm_func = _do_permutations
        X_full = np.concatenate(X, axis=0)
        n_samples_per_condition = [x.shape[0] for x in X]
        splits_idx = np.append([0], np.cumsum(n_samples_per_condition))
        slices = [slice(splits_idx[k], splits_idx[k + 1]) for k in range(len(X))]
        if within_subject:
            # Repeated-measures design: permute each subject's observations
            # only across the conditions (cells), never across subjects -- the
            # exchangeability assumption for repeated measures (FieldTrip's
            # depsamples* statistics permute the same way).
            n_cells, n_subjects = len(X), len(X[0])
            assert all(len(x) == n_subjects for x in X)  # checked by callers
            # a random permutation of the cells per (permutation, subject)
            cell_orders = np.argsort(
                rng.uniform(size=(n_permutations - 1, n_subjects, n_cells)), axis=-1
            )
            # the row index of (cell j, subject s) in X_full is
            # j * n_subjects + s, so position (j, s) draws from row
            # (cell_orders[:, s, j], s):
            orders = list(
                (
                    cell_orders.transpose(0, 2, 1) * n_subjects
                    + np.arange(n_subjects)[np.newaxis, np.newaxis]
                ).reshape(n_permutations - 1, -1)
            )
        else:
            orders = [rng.permutation(len(X_full)) for _ in range(n_permutations - 1)]
    del rng
    parallel, my_do_perm_func, n_jobs = parallel_func(
        do_perm_func, n_jobs, verbose=False
    )

    if len(clusters) == 0:
        warn("No clusters found, returning empty H0, clusters, and cluster_pv")
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
            iterable=range(len(orders)), mesg=f"Permuting{extra}"
        ) as progress_bar:
            H0 = parallel(
                my_do_perm_func(
                    X_full,
                    slices,
                    threshold,
                    tail,
                    adjacency,
                    stat_fun,
                    max_step,
                    this_include,
                    partitions,
                    t_power,
                    order,
                    sample_shape,
                    buffer_size,
                    progress_bar.subset(idx),
                )
                for idx, order in split_list(orders, n_jobs, idx=True)
            )
        # include original (true) ordering
        if tail == -1:  # up tail
            orig = cluster_stats.min()
        elif tail == 1:
            orig = cluster_stats.max()
        else:
            orig = abs(cluster_stats).max()
        H0.insert(0, [orig])
        H0 = np.concatenate(H0)
        logger.debug("Computing cluster p-values")
        cluster_pv = _pval_from_histogram(cluster_stats, H0, tail)

        # figure out how many new ones will be removed for step-down
        to_remove = np.where(cluster_pv < step_down_p)[0]
        n_removed = to_remove.size - total_removed
        total_removed = to_remove.size
        step_down_include = np.ones(n_tests, dtype=bool)
        for ti in to_remove:
            step_down_include[clusters[ti]] = False
        if adjacency is None and adjacency is not False:
            step_down_include = step_down_include.reshape(sample_shape, copy=False)
        n_step_downs += 1
        if step_down_p > 0:
            a_text = "additional " if n_step_downs > 1 else ""
            logger.info(
                "Step-down-in-jumps iteration #%i found %i %s"
                "cluster%s to exclude from subsequent iterations",
                n_step_downs,
                n_removed,
                a_text,
                _pl(n_removed),
            )

    # The clusters should have the same shape as the samples
    clusters = _reshape_clusters(clusters, sample_shape)
    return t_obs, clusters, cluster_pv, H0


def _rm_anova_stat_fun(*X, factor_levels, effects):
    """Wrap `f_mway_rm` for use as a cluster-test ``stat_fun``.

    ``X`` arrives as one 2D array (replications x flattened locations) per cell of
    the design, ordered so that the first factor varies slowest (matching how
    :func:`pandas.DataFrame.groupby` orders a multi-column group-by, and what
    :func:`~mne.stats.f_mway_rm` expects).
    """
    data = np.stack(X, axis=1)  # subjects x conditions x locations
    return f_mway_rm(
        data, factor_levels=factor_levels, effects=effects, return_pvals=False
    )[0]


def _check_fun(
    X, stat_fun, threshold, tail=0, kind="within", factor_levels=None, effects=None
):
    """Check the stat_fun and threshold values."""
    from scipy.stats import f as fstat
    from scipy.stats import t as tstat

    if kind == "within":
        if threshold is None:
            if stat_fun is not None and stat_fun is not ttest_1samp_no_p:
                warn(
                    "Automatic threshold is only valid for stat_fun=None "
                    f"(or ttest_1samp_no_p), got {stat_fun}"
                )
            p_thresh = 0.05 / (1 + (tail == 0))
            n_samples = len(X)
            threshold = -tstat.ppf(p_thresh, n_samples - 1)
            if np.sign(tail) < 0:
                threshold = -threshold
            logger.info(f"Using a threshold of {threshold:.6f}")
        stat_fun = ttest_1samp_no_p if stat_fun is None else stat_fun
    elif kind == "within_rm":
        n_subjects = len(X[0])
        if threshold is None:
            if stat_fun is not None:
                warn(
                    "Automatic threshold is only valid for stat_fun=None "
                    f"(uses f_mway_rm internally), got {stat_fun}"
                )
            elif tail != 1:
                warn('Ignoring argument "tail", performing 1-tailed F-test')
            threshold = f_threshold_mway_rm(n_subjects, factor_levels, effects=effects)
            logger.info(f"Using a threshold of {threshold:.6f}")
        if stat_fun is None:
            stat_fun = partial(
                _rm_anova_stat_fun, factor_levels=factor_levels, effects=effects
            )
    else:
        assert kind == "between"
        if threshold is None:
            if stat_fun is not None and stat_fun is not f_oneway:
                warn(
                    "Automatic threshold is only valid for stat_fun=None "
                    f"(or f_oneway), got {stat_fun}"
                )
            elif tail != 1:
                warn('Ignoring argument "tail", performing 1-tailed F-test')
            p_thresh = 0.05
            dfn = len(X) - 1
            dfd = np.sum([len(x) for x in X]) - len(X)
            threshold = fstat.ppf(1.0 - p_thresh, dfn, dfd)
            logger.info(f"Using a threshold of {threshold:.6f}")
        stat_fun = f_oneway if stat_fun is None else stat_fun
    return stat_fun, threshold


@legacy(alt="mne.stats.cluster_test(...)")
@_legacy_rng("seed")
@verbose
def permutation_cluster_test(
    X,
    threshold=None,
    n_permutations=1024,
    tail=0,
    stat_fun=None,
    adjacency=None,
    n_jobs=None,
    max_step=1,
    exclude=None,
    step_down_p=0,
    t_power=1,
    out_type="indices",
    check_disjoint=False,
    buffer_size=1000,
    verbose=None,
    *,
    rng=None,
    seed=None,
):
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
    %(max_step_clust)s
    %(exclude_clust)s
    %(step_down_p_clust)s
    %(f_power_clust)s
    %(out_type_clust)s
    %(check_disjoint_clust)s
    %(buffer_size_clust)s
    %(verbose)s
    %(rng)s
    %(seed_rng)s

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
    stat_fun, threshold = _check_fun(X, stat_fun, threshold, tail, "between")
    return _permutation_cluster_test(
        X=X,
        threshold=threshold,
        n_permutations=n_permutations,
        tail=tail,
        stat_fun=stat_fun,
        adjacency=adjacency,
        n_jobs=n_jobs,
        rng=rng,
        max_step=max_step,
        exclude=exclude,
        step_down_p=step_down_p,
        t_power=t_power,
        out_type=out_type,
        check_disjoint=check_disjoint,
        buffer_size=buffer_size,
    )


@_legacy_rng("seed")
@verbose
def permutation_cluster_1samp_test(
    X,
    threshold=None,
    n_permutations=1024,
    tail=0,
    stat_fun=None,
    adjacency=None,
    n_jobs=None,
    max_step=1,
    exclude=None,
    step_down_p=0,
    t_power=1,
    out_type="indices",
    check_disjoint=False,
    buffer_size=1000,
    verbose=None,
    *,
    rng=None,
    seed=None,
):
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
    %(max_step_clust)s
    %(exclude_clust)s
    %(step_down_p_clust)s
    %(t_power_clust)s
    %(out_type_clust)s
    %(check_disjoint_clust)s
    %(buffer_size_clust)s
    %(verbose)s
    %(rng)s
    %(seed_rng)s

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
    given the number of observations, then ``n_permutations``, ``seed``, and
    ``rng`` will be ignored since an exact test (full permutation test) will
    be performed (this is the case when
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
        X=[X],
        threshold=threshold,
        n_permutations=n_permutations,
        tail=tail,
        stat_fun=stat_fun,
        adjacency=adjacency,
        n_jobs=n_jobs,
        rng=rng,
        max_step=max_step,
        exclude=exclude,
        step_down_p=step_down_p,
        t_power=t_power,
        out_type=out_type,
        check_disjoint=check_disjoint,
        buffer_size=buffer_size,
    )


@_legacy_rng("seed")
@verbose
def spatio_temporal_cluster_1samp_test(
    X,
    threshold=None,
    n_permutations=1024,
    tail=0,
    stat_fun=None,
    adjacency=None,
    n_jobs=None,
    max_step=1,
    spatial_exclude=None,
    step_down_p=0,
    t_power=1,
    out_type="indices",
    check_disjoint=False,
    buffer_size=1000,
    verbose=None,
    *,
    rng=None,
    seed=None,
):
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
    %(max_step_clust)s
    spatial_exclude : list of int or None
        List of spatial indices to exclude from clustering.
    %(step_down_p_clust)s
    %(t_power_clust)s
    %(out_type_clust)s
    %(check_disjoint_clust)s
    %(buffer_size_clust)s
    %(verbose)s
    %(rng)s
    %(seed_rng)s

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
            np.prod(X.shape[1:-1]), X.shape[-1], spatial_exclude, True
        )
    else:
        exclude = None
    return permutation_cluster_1samp_test(
        X,
        threshold=threshold,
        stat_fun=stat_fun,
        tail=tail,
        n_permutations=n_permutations,
        adjacency=adjacency,
        n_jobs=n_jobs,
        rng=rng,
        max_step=max_step,
        exclude=exclude,
        step_down_p=step_down_p,
        t_power=t_power,
        out_type=out_type,
        check_disjoint=check_disjoint,
        buffer_size=buffer_size,
    )


@_legacy_rng("seed")
@verbose
def spatio_temporal_cluster_test(
    X,
    threshold=None,
    n_permutations=1024,
    tail=0,
    stat_fun=None,
    adjacency=None,
    n_jobs=None,
    max_step=1,
    spatial_exclude=None,
    step_down_p=0,
    t_power=1,
    out_type="indices",
    check_disjoint=False,
    buffer_size=1000,
    verbose=None,
    *,
    rng=None,
    seed=None,
):
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
    %(max_step_clust)s
    spatial_exclude : list of int or None
        List of spatial indices to exclude from clustering.
    %(step_down_p_clust)s
    %(f_power_clust)s
    %(out_type_clust)s
    %(check_disjoint_clust)s
    %(buffer_size_clust)s
    %(verbose)s
    %(rng)s
    %(seed_rng)s

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
            np.prod(X[0].shape[1:-1]), X[0].shape[-1], spatial_exclude, True
        )
    else:
        exclude = None
    return permutation_cluster_test(
        X,
        threshold=threshold,
        stat_fun=stat_fun,
        tail=tail,
        n_permutations=n_permutations,
        adjacency=adjacency,
        n_jobs=n_jobs,
        rng=rng,
        max_step=max_step,
        exclude=exclude,
        step_down_p=step_down_p,
        t_power=t_power,
        out_type=out_type,
        check_disjoint=check_disjoint,
        buffer_size=buffer_size,
    )


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
def _get_partitions_from_adjacency(adjacency, n_tests, n_times, verbose=None):
    """Specify disjoint subsets (e.g., hemispheres) based on adjacency."""
    # adjacency is spatial-only (see _setup_adjacency) when it is smaller
    # than the full (flattened) data; those partitions need to be tiled
    # across the (e.g., time) dimension that adjacency doesn't cover.
    is_spatial_only = adjacency.shape[0] != n_tests
    test = np.ones(adjacency.shape[0])

    part_clusts = _find_clusters(test, 0, 1, adjacency)[0]
    if len(part_clusts) > 1:
        logger.info(f"{len(part_clusts)} disjoint adjacency sets found")
        partitions = np.zeros(len(test), dtype="int")
        for ii, pc in enumerate(part_clusts):
            partitions[pc] = ii
        if is_spatial_only:
            partitions = np.tile(partitions, n_times)
    else:
        logger.info("No disjoint adjacency sets found")
        partitions = None

    return partitions


def _reshape_clusters(clusters, sample_shape):
    """Reshape cluster masks or indices to be of the correct shape."""
    # format of the bool mask and indices are ndarrays
    if len(clusters) > 0 and isinstance(clusters[0], np.ndarray):
        if clusters[0].dtype == np.dtype(bool):  # format of mask
            clusters = [c.reshape(sample_shape) for c in clusters]
        else:  # format of indices
            clusters = [np.unravel_index(c, sample_shape) for c in clusters]
    return clusters


def summarize_clusters_stc(
    clu, p_thresh=0.05, tstep=1.0, tmin=0, subject="fsaverage", vertices=None
):
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
    _validate_type(vertices, (None, list, SourceSpaces), "vertices")
    if vertices is None:
        vertices = [np.arange(10242), np.arange(10242)]
        klass = SourceEstimate
    elif isinstance(vertices, SourceSpaces):
        klass = dict(
            surface=SourceEstimate, volume=VolSourceEstimate, mixed=MixedSourceEstimate
        )[vertices.kind]
        vertices = [s["vertno"] for s in vertices]
    else:
        klass = {1: VolSourceEstimate, 2: SourceEstimate}.get(
            len(vertices), MixedSourceEstimate
        )
    n_vertices_need = sum(len(v) for v in vertices)

    t_obs, clusters, clu_pvals, _ = clu
    n_times, n_vertices = t_obs.shape
    if n_vertices != n_vertices_need:
        raise ValueError(
            f"Number of cluster vertices ({n_vertices}) did not match the "
            f"provided vertices ({n_vertices_need})"
        )
    good_cluster_inds = np.where(clu_pvals < p_thresh)[0]

    #  Build a convenient representation of each cluster, where each
    #  cluster becomes a "time point" in the SourceEstimate
    if len(good_cluster_inds) == 0:
        raise RuntimeError(
            "No significant clusters available. Please adjust "
            "your threshold or check your statistical "
            "analysis."
        )
    data = np.zeros((n_vertices, n_times))
    data_summary = np.zeros((n_vertices, len(good_cluster_inds) + 1))
    from ._cluster_level_numba import _sum_cluster_data

    for ii, cluster_ind in enumerate(good_cluster_inds):
        data.fill(0)
        t_inds, v_inds = clusters[cluster_ind]
        data[v_inds, t_inds] = t_obs[t_inds, v_inds]
        # Store a nice visualization of the cluster by summing across time
        data_summary[:, ii + 1] = np.sum(_sum_cluster_data(data, tstep), axis=1)
        # Make the first "time point" a sum across all clusters for easy
        # visualization
    data_summary[:, 0] = np.sum(data_summary, axis=1)

    return klass(data_summary, vertices, tmin, tstep, subject)


def _validate_cluster_df(df: DataFrame, dv_name: str, iv_names: list[str]):
    """Validate the input DataFrame for cluster tests."""
    # check if all necessary columns are present
    missing = ({dv_name} | set(iv_names)) - set(df.columns)  # should be empty
    sep = '", "'
    if missing:  # if not empty, there are missing columns
        raise ValueError(
            f"DataFrame must contain a column named for each term in `formula`. "
            f"Column{_pl(missing)} missing for term{_pl(missing)} "  # _pl = pluralize
            f'"{sep.join(missing)}".'
        )
    # check if the data column contains valid (and consistent) instance types
    inst = df[dv_name].iloc[0]
    valid_types = (
        Evoked,
        BaseEpochs,
        BaseTFR,
        np.ndarray,
    )  # Base covers all Epochs and TFRs
    _validate_type(inst, valid_types, f"Data in dependent variable column '{dv_name}'")
    all_types = set(df[dv_name].map(type))
    all_type_names = ", ".join([type(x).__name__ for x in all_types])
    prologue = f"Data in dependent variable column '{dv_name}' must all have "
    if len(all_types) > 1:
        raise ValueError(
            f"{prologue} the same type, but found types {{{all_type_names}}}."
        )
    # check if the shape of the data is consistent
    if isinstance(inst, np.ndarray):
        all_shapes = set(
            df[dv_name].map(lambda x: x.shape[1:])
        )  # first dim may vary (participants or epochs)
    elif isinstance(inst, (BaseEpochs | EpochsTFR)):
        all_shapes = set(df[dv_name].map(lambda x: x.get_data().shape[1:]))
    else:
        all_shapes = set(df[dv_name].map(lambda x: x.get_data().shape))
    if len(all_shapes) > 1:
        raise ValueError(
            f"{prologue} consistent shape, but {len(all_shapes)} different "
            f"shapes were found: {'; '.join(all_shapes)}."
        )
    obj_type = all_types.pop()
    is_epo = GetEpochsMixin in obj_type.__mro__
    is_tfr = BaseTFR in obj_type.__mro__
    is_arr = np.ndarray in obj_type.__mro__
    return is_epo, is_tfr, is_arr


# TODO: design/analysis features FieldTrip's cluster stats support that
# cluster_test does not (yet):
# - continuous predictors / regression & correlation designs
#   (ft_statfun_indepsamplesregrT, _depsamplesregrT, _correlationT); the
#   formula right-hand side currently must be categorical
# - multivariate within-subject F across conditions
#   (ft_statfun_depsamplesFmultivariate)
# - activation-versus-baseline tests (ft_statfun_actvsblT)
# - control variables / stratified or blocked resampling (cfg.cvar, cfg.wvar)
# - requiring a minimum number of neighboring channels for cluster membership
#   (cfg.minnbchan)
# - the weighted cluster mass statistic (cfg.clusterstatistic='wcm');
#   ``t_power`` covers maxsum (t_power=1) and maxsize (t_power=0) only
@verbose
def cluster_test(
    df: DataFrame,
    formula: str,
    *,  # end of positional-only parameters
    within_id: str | None = None,
    stat_fun: callable | None = None,
    tail: Literal[-1, 0, 1] = 0,
    threshold=None,
    n_permutations: str | int = 1024,
    adjacency: sparse.spmatrix
    | None
    | Literal[False] = None,  # should be None (default)
    max_step: int = 1,  # TODO may need to provide `max_step_time` and `max_step_freq`
    exclude: list | None = None,  # TODO needs rethink because user passes MNE objects
    step_down_p: float = 0.0,
    t_power: float = 1.0,
    check_disjoint: bool = False,
    out_type: Literal["indices", "mask"] = "indices",
    rng: None | int | np.random.Generator | np.random.RandomState = None,
    buffer_size: int | None = None,
    n_jobs: int = 1,
    verbose=None,
):
    """Run a cluster permutation test from a DataFrame and a formula.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data, dependent and independent variables.
    formula : str
        Wilkinson notation formula naming the dependent variable and either a single
        independent variable (e.g. ``"data ~ condition"``) or a single interaction
        term between two or more independent variables (e.g. ``"data ~ a:b"``, tested
        with a repeated-measures ANOVA; see ``within_id``). All names must match
        columns in ``df``. Testing several effects (e.g. two main effects, or a main
        effect and an interaction) requires calling :func:`cluster_test` once per
        effect.
    within_id : None | str
        Name of column in ``df`` to use in identifying within-group contrasts.

        - If ``within_id`` is not ``None``:
            ``within_id`` must match a column name in ``df``, e.g. ``"subject_index"``
            (a name not in ``df.columns`` will result in an error). If the independent
            variable has 1 level per participant, the data will be treated as
            already subtracted (e.g., condition A - condition B) and a paired t-test
            against zero will be performed (using
            :func:`mne.stats.ttest_1samp_no_p`). If the independent
            variable has 2 levels, the data will be subtracted for each participant
            (e.g., condition A - condition B) first. If it has more than 2 levels,
            a one-way repeated-measures ANOVA is performed (using
            :func:`mne.stats.f_mway_rm`), with permutations swapping each
            subject's observations across the levels (never across subjects).

        - If ``within_id`` is ``None``:
            Will perform a between-group test (using :func:`mne.stats.f_oneway`; This
            works for 2 levels or more).

        - This parameter is required if:
            ``formula``'s right-hand side is an interaction term (e.g.
            ``"data ~ a:b"``), in which case each combination of ``within_id`` and the
            factors must appear exactly once (a fully balanced repeated-measures
            design).
    %(stat_fun_clust_both)s
    %(tail_clust)s
    %(threshold_clust_both)s
    %(n_permutations_clust_all)s
    %(adjacency_clust_both)s
    max_step : int
        Maximum distance between samples (time points). Default is 1.
    exclude : array-like of bool | None
        Mask to apply to the data to exclude certain points from clustering
        (e.g., medial wall vertices). Should be the same shape as the channels/vertices
        dimension of the data objects. If ``None``, no points are excluded.
    %(step_down_p_clust)s
    %(t_power_clust)s
    check_disjoint : bool
        Whether to check if the ``adjacency`` matrix can be separated into disjoint
        sets before clustering. This may lead to faster clustering, especially if
        the "time" and/or "frequency" dimensions are large.
    out_type : 'mask' | 'indices'
        Format used to represent each cluster in the list of clusters stored in
        the ``clusters`` attribute of :class:`mne.stats.ClusterResult`:

        - ``'mask'``:s
            Each cluster is represented by a boolean array of the same shape as
            the ``stat_obs`` attribute array of :class:`mne.stats.ClusterResult`,
            with ``True`` values indicating locations that are part of a cluster.  Note
            that MNE-Python's legacy API
            (e.g. :func:`mne.stats.permutation_cluster_test`) would return slices if the
            shape is 1D and adjacency is ``None``, whereas ``cluster_test`` will always
            return a boolean array.

        - ``'indices'``:
            Each cluster is represented by a tuple of 1D integer arrays, one array per
            dimension of the array in the ``stat_obs`` attribute of
            :class:`mne.stats.ClusterResult`. The arrays
            together give the coordinates of all locations belonging to the cluster and
            can be used to index ``stat_obs``.
            Note that for large datasets, ``'indices'`` may use far less memory than
            ``'mask'``.
    %(rng)s
    buffer_size : int | None
        Block size to use when computing test statistics. This can significantly
        reduce memory usage when ``n_jobs > 1`` and memory sharing between
        processes is enabled (see :func:`mne.set_cache_dir`), because the data will be
        shared between processes and each process only needs to allocate space for
        a small block of locations at a time.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    mne.stats.ClusterResult
        Object containing the results of the cluster permutation test.

    Notes
    -----
    %(threshold_clust_t_or_f_notes)s

    .. versionadded:: 1.13
    """
    # parse formula
    formulaic = _soft_import("formulaic", purpose="parse formula for clustering")
    parser = formulaic.parser.DefaultFormulaParser(include_intercept=False)
    rng = _check_rng(rng)

    formula_str = formula
    formula = formulaic.Formula(formula, _parser=parser)
    # extract the dependent variable name
    dv_name = str(formula.lhs)
    # the right-hand side must be a single term: either one factor (main effect,
    # e.g. "a") or a single interaction between factors (e.g. "a:b")
    rhs_terms = list(formula.rhs)
    if len(rhs_terms) != 1:
        raise ValueError(
            "the right-hand side of `formula` must be a single term: either one "
            'factor (e.g. "data ~ a") or a single interaction (e.g. "data ~ a:b"). '
            f'Got "{formula.rhs}", which has {len(rhs_terms)} terms. To test '
            "several effects, call `cluster_test` once per effect."
        )
    factor_names = [str(factor) for factor in rhs_terms[0].factors]
    is_interaction = len(factor_names) > 1
    iv_name = factor_names[0] if not is_interaction else ":".join(factor_names)

    # validate the input dataframe and return the type of the data column entries
    is_epo, is_tfr, is_arr = _validate_cluster_df(df, dv_name, factor_names)

    _validate_type(within_id, (str, None), "within_id")
    if within_id is not None and within_id not in df.columns:
        raise ValueError(
            f"within_id must be one of {list(df.columns)}, got {within_id!r}"
        )

    # check if within_id has 1 or 2 levels to do paired t-test (within)
    if is_interaction and within_id is None:
        raise ValueError(
            f'testing the interaction "{iv_name}" requires repeated-measures data; '
            "pass `within_id` naming the column that identifies each subject/"
            "replication."
        )
    # for within-subject designs, check that each subject has one observation per
    # combination of factor(s) (2 for a simple paired test; more for a one-way
    # repeated-measures ANOVA or an interaction)
    n_groups = df[factor_names].drop_duplicates().shape[0]
    if within_id and (is_interaction or n_groups >= 2):
        df = df.copy(deep=False)  # Don't mutate input dataframe row order!
        df.sort_values([*factor_names, within_id], inplace=True)
        counts = df[within_id].value_counts()

        iv_names = iv_name.split(":")
        groups = df[[dv_name, *iv_names, within_id]].groupby([*iv_names, within_id])
        elem = df[dv_name].iloc[0]
        # TODO: Support this for other input types e.g. array, epochs, TFR, etc.
        if isinstance(elem, Evoked):
            reduce = set(df.columns) - set([*iv_names, within_id, dv_name])
            if reduce:
                logger.info(
                    f"To test '{formula_str}', reducing along column(s): {reduce}"
                )
            func = {dv_name: lambda evs: combine_evoked(evs.tolist(), weights="nave")}
            df = groups.agg(func).reset_index()

        else:
            if any(counts != n_groups):
                raise ValueError(
                    f"for a within-subject test, each subject (column {within_id!r}) "
                    f"must have exactly {n_groups} observations, one per combination "
                    f"of {factor_names}."
                )
    # extract the data from the dataframe
    outer_func = np.concatenate if is_epo else np.array
    axes = (-3, -1) if is_tfr else (-2, -1)

    def func_arr(series):
        return np.concatenate(series.values)

    def func_mne(series):
        return outer_func(
            series.map(lambda inst: inst.get_data().swapaxes(*axes)).to_list()
        )

    func = func_arr if is_arr else func_mne

    # convert to a list-like X for clustering. Grouping by multiple columns sorts
    # lexicographically (first factor varies slowest), which is what f_mway_rm
    # expects for interaction effects.
    X = df.groupby(factor_names).agg({dv_name: func})[dv_name].to_list()

    # determine test type
    if is_interaction:
        kind = "within_rm"
        factor_levels = [df[name].nunique() for name in factor_names]
        # f_mway_rm/f_threshold_mway_rm only understand generic "A", "B", ...
        # factor labels (in the order given in `formula`), not the actual column
        # names, so translate the interaction accordingly.
        rm_effects = ":".join(ascii_uppercase[: len(factor_names)])
    elif len(X) == 1:
        kind = "within"  # single group -- e.g. already-subtracted paired data
        X = X[0]
    elif within_id is not None and len(X) > 2:
        # one within-subject factor with 3+ levels: one-way repeated-measures
        # ANOVA (each subject contributes one observation per level)
        kind = "within_rm"
        factor_levels = [len(X)]
        rm_effects = "A"
    elif len(X) > 2:
        kind = "between"
    elif (
        len(set(x.shape for x in X)) > 1
    ):  # check if there are unequal observations in each group
        kind = "between"
    # by now we know there are exactly 2 elements in X, and their shapes match
    elif within_id in df:
        kind = "within"

        n_vals = df[factor_names].nunique().item()
        vals = df[factor_names].squeeze().unique().tolist()
        assert len(X) == 2
        assert n_vals == 2

        logger.info(
            f"Subtracting ({vals[0]} - {vals[1]}) of column {factor_names} before "
            "computing cluster statistics."
        )
        X = X[0] - X[1]
    else:  # 2 elements in X but no within_id provided → unpaired test
        kind = "between"

    # Now, for the within case check if there are unequal observations in each group
    # and whether the data is already subtracted (1 level) or not (2 levels)
    if kind == "within":
        if len(set(x.shape for x in X)) > 1:
            raise ValueError(
                "for within-group tests, all participants must have the same number of "
                "observations, check your data frame"
            )
        if len(X) == 1:
            # turn it into an array
            X = X[0]  # already subtracted, just use the data as is

        elif len(X) == 2:
            X = X[0] - X[1]  # do subtraction for paired t-test

    # define stat function and threshold
    if kind == "within_rm":
        stat_fun, threshold = _check_fun(
            X=X,
            stat_fun=stat_fun,
            threshold=threshold,
            tail=tail,
            kind=kind,
            factor_levels=factor_levels,
            effects=rm_effects,
        )
    else:
        stat_fun, threshold = _check_fun(
            X=X, stat_fun=stat_fun, threshold=threshold, tail=tail, kind=kind
        )

    # check_fun doesn't work with list input`
    if kind == "within":  # will this create an issue for already subtracted data?
        X = [X]

    kind_descs = {
        "between": "between-groups F-test",
        "within": "one-sample T-test",
        "within_rm": "M-way repeated measures ANOVA",
    }
    logger.info(f"Chosen statistic: {kind_descs[kind]} ({stat_fun.__name__})")
    assert 1 == 0

    # Run the cluster-based permutation test
    stat_obs, clusters, cluster_p_values, H0 = _permutation_cluster_test(
        X,
        n_permutations=n_permutations,
        threshold=threshold,
        stat_fun=stat_fun,
        tail=tail,
        n_jobs=n_jobs,
        adjacency=adjacency,
        max_step=max_step,  # maximum distance between samples (time points)
        exclude=exclude,  # exclude no time points or channels
        step_down_p=step_down_p,  # step down in jumps test
        t_power=t_power,  # weigh each location by its stats score
        out_type=out_type,
        check_disjoint=check_disjoint,
        buffer_size=buffer_size,  # block size for chunking the data
        rng=rng,
        # repeated-measures ANOVA: permute within subjects only
        within_subject=kind == "within_rm",
    )

    stat_obs = stat_obs.T
    if out_type == "mask":
        if isinstance(clusters[0], np.ndarray) and clusters[0].dtype == "bool":
            clusters = [cl.T for cl in clusters]
        elif isinstance(clusters[0], tuple) and isinstance(clusters[0][0], slice):
            clusters = [tuple(reversed(cluster)) for cluster in clusters]
            # Convert from old form of slices to mask, make users life easier.
            new_clusters = list()
            for clust in clusters:
                new_clust = np.zeros(stat_obs.shape, bool)
                new_clust[clust] = True
                new_clusters.append(new_clust)
            clusters = new_clusters
    elif out_type == "indices":
        clusters = [tuple(reversed(cluster)) for cluster in clusters]
    return ClusterResult(
        stat_obs=stat_obs,
        clusters=clusters,
        cluster_p_values=cluster_p_values,
        H0=H0,
        stat_fun=stat_fun,
        n_permutations=n_permutations,
        t_power=t_power,
    )


def _cluster_mass(stat_obs, cluster, t_power):
    """Compute a cluster's mass, matching _find_clusters_1dir's own formula."""
    vals = stat_obs[cluster]
    if t_power == 1:
        return vals.sum()
    return (np.sign(vals) * np.abs(vals) ** t_power).sum()


class ClusterResult:
    """Object containing the results of the cluster permutation test.

    .. note::
       This class is not meant to be instantiated directly, but rather returned
       by :func:`~mne.stats.cluster_test`.

    Parameters
    ----------
    stat_obs : np.ndarray
        The observed test statistic.
    clusters : list
        List of clusters.
    cluster_p_values : np.ndarray
        P-values for each cluster.
    H0 : np.ndarray
        Max cluster level stats observed under permutation.
    stat_fun : callable | None
        Function called to calculate the test statistic. Must accept 1D-array as
        input and return a 1D array. If ``None`` (the default), uses
        :func:`mne.stats.ttest_1samp_no_p` for paired tests and
        :func:`mne.stats.f_oneway` for unpaired tests or tests of more than 2 groups.
    n_permutations : int
        The number of permutations that were taken to compute the test statistic.
    t_power : float
        Power to which the observed statistic was raised (sign retained) before
        summing within a cluster to obtain its mass (see ``cluster_masses``).
        Should match whatever ``t_power`` was passed to :func:`cluster_test`.

    Attributes
    ----------
    cluster_masses : np.ndarray
        The mass of each cluster, i.e. the sum (optionally ``t_power``-weighted)
        of ``stat_obs`` within that cluster. This is the same per-cluster
        statistic that is compared against the permutation distribution (``H0``)
        to obtain ``cluster_p_values``, so it is a natural way to rank clusters by
        how extreme they are, independent of the resulting p-value.

    Notes
    -----
    .. versionadded:: 1.13
    """

    def __init__(
        self,
        *,
        stat_obs: np.typing.NDArray,
        clusters: list,
        cluster_p_values: np.typing.NDArray,
        H0: np.typing.NDArray,
        stat_fun: callable,
        n_permutations: int,
        t_power: float = 1.0,
    ):
        self.stat_obs = stat_obs
        self.clusters = clusters
        self.cluster_p_values = cluster_p_values
        self.H0 = H0
        self.stat_fun = stat_fun
        self.t_power = t_power
        self.cluster_masses = np.array(
            [_cluster_mass(stat_obs, c, t_power) for c in clusters]
        )
        self.n_permutations = n_permutations

        # unpaired t-test equivalent to f_oneway w/ 2 groups
        if stat_fun is f_oneway:
            self.stat_name = "F-statistic"
        elif stat_fun is ttest_1samp_no_p:
            self.stat_name = "paired T-statistic"
        elif isinstance(stat_fun, partial) and stat_fun.func is _rm_anova_stat_fun:
            self.stat_name = "F-statistic (repeated-measures ANOVA)"
        else:
            self.stat_name = "test statistic"

    def __repr__(self):  # noqa: D105
        return (
            f"<ClusterResult | p={self.cluster_p_values.min()}, "
            f"{len(self.clusters)} clusters."
        )
