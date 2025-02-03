"""Some utility functions."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import inspect
import numbers
import operator
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from io import BytesIO, StringIO
from math import ceil, sqrt
from pathlib import Path

import numpy as np
from scipy import sparse

from ..fixes import (
    _infer_dimension_,
    _safe_svd,
    has_numba,
    jit,
    stable_cumsum,
    svd_flip,
)
from ._logging import logger, verbose, warn
from .check import (
    _check_pandas_installed,
    _ensure_int,
    _validate_type,
    check_random_state,
)
from .docs import fill_doc
from .misc import _empty_hash, _pl


def split_list(v, n, idx=False):
    """Split list in n (approx) equal pieces, possibly giving indices."""
    n = int(n)
    tot = len(v)
    sz = tot // n
    start = stop = 0
    for i in range(n - 1):
        stop += sz
        yield (np.arange(start, stop), v[start:stop]) if idx else v[start:stop]
        start += sz
    yield (np.arange(start, tot), v[start:]) if idx else v[start]


def array_split_idx(ary, indices_or_sections, axis=0, n_per_split=1):
    """Do what numpy.array_split does, but add indices."""
    # this only works for indices_or_sections as int
    indices_or_sections = _ensure_int(indices_or_sections)
    ary_split = np.array_split(ary, indices_or_sections, axis=axis)
    idx_split = np.array_split(np.arange(ary.shape[axis]), indices_or_sections)
    idx_split = (
        np.arange(sp[0] * n_per_split, (sp[-1] + 1) * n_per_split) for sp in idx_split
    )
    return zip(idx_split, ary_split)


def sum_squared(X):
    """Compute norm of an array.

    Parameters
    ----------
    X : array
        Data whose norm must be found.

    Returns
    -------
    value : float
        Sum of squares of the input array X.
    """
    X_flat = X.ravel(order="F" if np.isfortran(X) else "C")
    return np.dot(X_flat, X_flat)


def _compute_row_norms(data):
    """Compute scaling based on estimated norm."""
    norms = np.sqrt(np.sum(data**2, axis=1))
    norms[norms == 0] = 1.0
    return norms


def _reg_pinv(x, reg=0, rank="full", rcond=1e-15):
    """Compute a regularized pseudoinverse of Hermitian matrices.

    Regularization is performed by adding a constant value to each diagonal
    element of the matrix before inversion. This is known as "diagonal
    loading". The loading factor is computed as ``reg * np.trace(x) / len(x)``.

    The pseudo-inverse is computed through SVD decomposition and inverting the
    singular values. When the matrix is rank deficient, some singular values
    will be close to zero and will not be used during the inversion. The number
    of singular values to use can either be manually specified or automatically
    estimated.

    Parameters
    ----------
    x : ndarray, shape (..., n, n)
        Square, Hermitian matrices to invert.
    reg : float
        Regularization parameter. Defaults to 0.
    rank : int | None | 'full'
        This controls the effective rank of the covariance matrix when
        computing the inverse. The rank can be set explicitly by specifying an
        integer value. If ``None``, the rank will be automatically estimated.
        Since applying regularization will always make the covariance matrix
        full rank, the rank is estimated before regularization in this case. If
        'full', the rank will be estimated after regularization and hence
        will mean using the full rank, unless ``reg=0`` is used.
        Defaults to 'full'.
    rcond : float | 'auto'
        Cutoff for detecting small singular values when attempting to estimate
        the rank of the matrix (``rank='auto'``). Singular values smaller than
        the cutoff are set to zero. When set to 'auto', a cutoff based on
        floating point precision will be used. Defaults to 1e-15.

    Returns
    -------
    x_inv : ndarray, shape (..., n, n)
        The inverted matrix.
    loading_factor : float
        Value added to the diagonal of the matrix during regularization.
    rank : int
        If ``rank`` was set to an integer value, this value is returned,
        else the estimated rank of the matrix, before regularization, is
        returned.
    """
    from ..rank import _estimate_rank_from_s

    if rank is not None and rank != "full":
        rank = int(operator.index(rank))
    if x.ndim < 2 or x.shape[-2] != x.shape[-1]:
        raise ValueError("Input matrix must be square.")
    if not np.allclose(x, x.conj().swapaxes(-2, -1)):
        raise ValueError("Input matrix must be Hermitian (symmetric)")
    assert x.ndim >= 2 and x.shape[-2] == x.shape[-1]
    n = x.shape[-1]

    # Decompose the matrix, not necessarily positive semidefinite
    U, s, Vh = np.linalg.svd(x, hermitian=True)

    # Estimate the rank before regularization
    tol = "auto" if rcond == "auto" else rcond * s[..., :1]
    rank_before = _estimate_rank_from_s(s, tol)

    # Decompose the matrix again after regularization
    loading_factor = reg * np.mean(s, axis=-1)
    if reg:
        U, s, Vh = np.linalg.svd(
            x + loading_factor[..., np.newaxis, np.newaxis] * np.eye(n), hermitian=True
        )

    # Estimate the rank after regularization
    tol = "auto" if rcond == "auto" else rcond * s[..., :1]
    rank_after = _estimate_rank_from_s(s, tol)

    # Warn the user if both all parameters were kept at their defaults and the
    # matrix is rank deficient.
    if (rank_after < n).any() and reg == 0 and rank == "full" and rcond == 1e-15:
        warn("Covariance matrix is rank-deficient and no regularization is done.")
    elif isinstance(rank, int) and rank > n:
        raise ValueError(
            f"Invalid value for the rank parameter ({rank}) given "
            f"the shape of the input matrix ({x.shape[0]} x {x.shape[1]})."
        )

    # Pick the requested number of singular values
    mask = np.arange(s.shape[-1]).reshape((1,) * (x.ndim - 2) + (-1,))
    if rank is None:
        cmp = ret = rank_before
    elif rank == "full":
        cmp = rank_after
        ret = rank_before
    else:
        cmp = ret = rank
    mask = mask < np.asarray(cmp)[..., np.newaxis]
    mask &= s > 0

    # Invert only non-zero singular values
    s_inv = np.zeros(s.shape)
    s_inv[mask] = 1.0 / s[mask]

    # Compute the pseudo inverse
    x_inv = np.matmul(U * s_inv[..., np.newaxis, :], Vh)

    return x_inv, loading_factor, ret


def _gen_events(n_epochs):
    """Generate event structure from number of epochs."""
    events = np.c_[np.arange(n_epochs), np.zeros(n_epochs, int), np.ones(n_epochs, int)]
    return events


def _reject_data_segments(data, reject, flat, decim, info, tstep):
    """Reject data segments using peak-to-peak amplitude."""
    from .._fiff.pick import channel_indices_by_type
    from ..epochs import _is_good

    data_clean = np.empty_like(data)
    idx_by_type = channel_indices_by_type(info)
    step = int(ceil(tstep * info["sfreq"]))
    if decim is not None:
        step = int(ceil(step / float(decim)))
    this_start = 0
    this_stop = 0
    drop_inds = []
    for first in range(0, data.shape[1], step):
        last = first + step
        data_buffer = data[:, first:last]
        if data_buffer.shape[1] < (last - first):
            break  # end of the time segment
        if _is_good(
            data_buffer,
            info["ch_names"],
            idx_by_type,
            reject,
            flat,
            ignore_chs=info["bads"],
        ):
            this_stop = this_start + data_buffer.shape[1]
            data_clean[:, this_start:this_stop] = data_buffer
            this_start += data_buffer.shape[1]
        else:
            logger.info(f"Artifact detected in [{first}, {last}]")
            drop_inds.append((first, last))
    data = data_clean[:, :this_stop]
    if not data.any():
        raise RuntimeError(
            "No clean segment found. Please "
            "consider updating your rejection "
            "thresholds."
        )
    return data, drop_inds


def _get_inst_data(inst):
    """Get data view from MNE object instance like Raw, Epochs or Evoked."""
    from ..epochs import BaseEpochs
    from ..evoked import Evoked
    from ..io import BaseRaw
    from ..time_frequency.tfr import BaseTFR

    _validate_type(inst, (BaseRaw, BaseEpochs, Evoked, BaseTFR), "Instance")
    if not inst.preload:
        inst.load_data()
    return inst._data


def compute_corr(x, y):
    """Compute pearson correlations between a vector and a matrix."""
    if len(x) == 0 or len(y) == 0:
        raise ValueError("x or y has zero length")
    X = np.array(x, float)
    Y = np.array(y, float)
    X -= X.mean(0)
    Y -= Y.mean(0)
    x_sd = X.std(0, ddof=1)
    # if covariance matrix is fully expanded, Y needs a
    # transpose / broadcasting else Y is correct
    y_sd = Y.std(0, ddof=1)[:, None if X.shape == Y.shape else Ellipsis]
    return (np.dot(X.T, Y) / float(len(X) - 1)) / (x_sd * y_sd)


@fill_doc
def random_permutation(n_samples, random_state=None):
    """Emulate the randperm matlab function.

    It returns a vector containing a random permutation of the
    integers between 0 and n_samples-1. It returns the same random numbers
    than randperm matlab function whenever the random_state is the same
    as the matlab's random seed.

    This function is useful for comparing against matlab scripts
    which use the randperm function.

    Note: the randperm(n_samples) matlab function generates a random
    sequence between 1 and n_samples, whereas
    random_permutation(n_samples, random_state) function generates
    a random sequence between 0 and n_samples-1, that is:
    randperm(n_samples) = random_permutation(n_samples, random_state) - 1

    Parameters
    ----------
    n_samples : int
        End point of the sequence to be permuted (excluded, i.e., the end point
        is equal to n_samples-1)
    %(random_state)s

    Returns
    -------
    randperm : ndarray, int
        Randomly permuted sequence between 0 and n-1.
    """
    rng = check_random_state(random_state)
    # This can't just be rng.permutation(n_samples) because it's not identical
    # to what MATLAB produces
    idx = rng.uniform(size=n_samples)
    randperm = np.argsort(idx)
    return randperm


@verbose
def _apply_scaling_array(data, picks_list, scalings, verbose=None):
    """Scale data type-dependently for estimation."""
    scalings = _check_scaling_inputs(data, picks_list, scalings)
    if isinstance(scalings, dict):
        logger.debug(f"    Scaling using mapping {scalings}.")
        picks_dict = dict(picks_list)
        scalings = [(picks_dict[k], v) for k, v in scalings.items() if k in picks_dict]
        for idx, scaling in scalings:
            data[idx, :] *= scaling  # F - order
    else:
        logger.debug("    Scaling using computed norms.")
        data *= scalings[:, np.newaxis]  # F - order


def _invert_scalings(scalings):
    if isinstance(scalings, dict):
        scalings = {k: 1.0 / v for k, v in scalings.items()}
    elif isinstance(scalings, np.ndarray):
        scalings = 1.0 / scalings
    return scalings


def _undo_scaling_array(data, picks_list, scalings):
    scalings = _invert_scalings(_check_scaling_inputs(data, picks_list, scalings))
    return _apply_scaling_array(data, picks_list, scalings, verbose=False)


@contextmanager
def _scaled_array(data, picks_list, scalings):
    """Scale, use, unscale array."""
    _apply_scaling_array(data, picks_list=picks_list, scalings=scalings)
    try:
        yield
    finally:
        _undo_scaling_array(data, picks_list=picks_list, scalings=scalings)


def _apply_scaling_cov(data, picks_list, scalings):
    """Scale resulting data after estimation."""
    scalings = _check_scaling_inputs(data, picks_list, scalings)
    scales = None
    if isinstance(scalings, dict):
        n_channels = len(data)
        covinds = list(zip(*picks_list))[1]
        assert len(data) == sum(len(k) for k in covinds)
        assert list(sorted(np.concatenate(covinds))) == list(range(len(data)))
        scales = np.zeros(n_channels)
        for ch_t, idx in picks_list:
            scales[idx] = scalings[ch_t]
    elif isinstance(scalings, np.ndarray):
        if len(scalings) != len(data):
            raise ValueError("Scaling factors and data are of incompatible shape")
        scales = scalings
    elif scalings is None:
        pass
    else:
        raise RuntimeError("Arff...")
    if scales is not None:
        assert np.sum(scales == 0.0) == 0
        data *= scales[None, :] * scales[:, None]


def _undo_scaling_cov(data, picks_list, scalings):
    scalings = _invert_scalings(_check_scaling_inputs(data, picks_list, scalings))
    return _apply_scaling_cov(data, picks_list, scalings)


def _check_scaling_inputs(data, picks_list, scalings):
    """Aux function."""
    rescale_dict_ = dict(mag=1e15, grad=1e13, eeg=1e6)

    scalings_ = None
    if isinstance(scalings, str) and scalings == "norm":
        scalings_ = 1.0 / _compute_row_norms(data)
    elif isinstance(scalings, dict):
        rescale_dict_.update(scalings)
        scalings_ = rescale_dict_
    elif isinstance(scalings, np.ndarray):
        scalings_ = scalings
    elif scalings is None:
        pass
    else:
        raise NotImplementedError(f"Not a valid rescaling option: {scalings}")
    return scalings_


def hashfunc(fname, block_size=1048576, hash_type="md5"):  # 2 ** 20
    """Calculate the hash for a file.

    Parameters
    ----------
    fname : str
        Filename.
    block_size : int
        Block size to use when reading.

    Returns
    -------
    hash_ : str
        The hexadecimal digest of the hash.
    """
    hasher = _empty_hash(kind=hash_type)
    with open(fname, "rb") as fid:
        while True:
            data = fid.read(block_size)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def create_slices(start, stop, step=None, length=1):
    """Generate slices of time indexes.

    Parameters
    ----------
    start : int
        Index where first slice should start.
    stop : int
        Index where last slice should maximally end.
    length : int
        Number of time sample included in a given slice.
    step: int | None
        Number of time samples separating two slices.
        If step = None, step = length.

    Returns
    -------
    slices : list
        List of slice objects.
    """
    # default parameters
    if step is None:
        step = length

    # slicing
    slices = [slice(t, t + length, 1) for t in range(start, stop - length + 1, step)]
    return slices


def _time_mask(
    times, tmin=None, tmax=None, sfreq=None, raise_error=True, include_tmax=True
):
    """Safely find sample boundaries."""
    orig_tmin = tmin
    orig_tmax = tmax
    tmin = -np.inf if tmin is None else tmin
    tmax = np.inf if tmax is None else tmax
    if not np.isfinite(tmin):
        tmin = times[0]
    if not np.isfinite(tmax):
        tmax = times[-1]
        include_tmax = True  # ignore this param when tmax is infinite
    if sfreq is not None:
        # Push to a bit past the nearest sample boundary first
        sfreq = float(sfreq)
        tmin = int(round(tmin * sfreq)) / sfreq - 0.5 / sfreq
        tmax = int(round(tmax * sfreq)) / sfreq
        tmax += (0.5 if include_tmax else -0.5) / sfreq
    else:
        assert include_tmax  # can only be used when sfreq is known
    if raise_error and tmin > tmax:
        raise ValueError(
            f"tmin ({orig_tmin}) must be less than or equal to tmax ({orig_tmax})"
        )
    mask = times >= tmin
    mask &= times <= tmax
    if raise_error and not mask.any():
        extra = "" if include_tmax else "when include_tmax=False "
        raise ValueError(
            f"No samples remain when using tmin={orig_tmin} and tmax={orig_tmax} "
            f"{extra}(original time bounds are [{times[0]}, {times[-1]}] containing "
            f"{len(times)} sample{_pl(times)})"
        )
    return mask


def _freq_mask(freqs, sfreq, fmin=None, fmax=None, raise_error=True):
    """Safely find frequency boundaries."""
    orig_fmin = fmin
    orig_fmax = fmax
    fmin = -np.inf if fmin is None else fmin
    fmax = np.inf if fmax is None else fmax
    if not np.isfinite(fmin):
        fmin = freqs[0]
    if not np.isfinite(fmax):
        fmax = freqs[-1]
    if sfreq is None:
        raise ValueError("sfreq can not be None")
    # Push 0.5/sfreq past the nearest frequency boundary first
    sfreq = float(sfreq)
    fmin = int(round(fmin * sfreq)) / sfreq - 0.5 / sfreq
    fmax = int(round(fmax * sfreq)) / sfreq + 0.5 / sfreq
    if raise_error and fmin > fmax:
        raise ValueError(
            f"fmin ({orig_fmin}) must be less than or equal to fmax ({orig_fmax})"
        )
    mask = freqs >= fmin
    mask &= freqs <= fmax
    if raise_error and not mask.any():
        raise ValueError(
            f"No frequencies remain when using fmin={orig_fmin} and fmax={orig_fmax} "
            f"(original frequency bounds are [{freqs[0]}, {freqs[-1]}])"
        )
    return mask


def grand_average(all_inst, interpolate_bads=True, drop_bads=True):
    """Make grand average of a list of Evoked, AverageTFR, or Spectrum data.

    For :class:`mne.Evoked` data, the function interpolates bad channels based on the
    ``interpolate_bads`` parameter. If ``interpolate_bads`` is True, the grand average
    file will contain good channels and the bad channels interpolated from the good
    MEG/EEG channels.
    For :class:`mne.time_frequency.AverageTFR` and :class:`mne.time_frequency.Spectrum`
    data, the function takes the subset of channels not marked as bad in any of the
    instances.

    The ``grand_average.nave`` attribute will be equal to the number of datasets used to
    calculate the grand average.

    .. note:: A grand average evoked should not be used for source localization.

    Parameters
    ----------
    all_inst : list of Evoked, AverageTFR or Spectrum
        The datasets.

        .. versionchanged:: 1.10.0
            Added support for :class:`~mne.time_frequency.Spectrum` objects.

    interpolate_bads : bool
        If True, bad MEG and EEG channels are interpolated. Ignored for
        :class:`~mne.time_frequency.AverageTFR` and
        :class:`~mne.time_frequency.Spectrum` data.
    drop_bads : bool
        If True, drop all bad channels marked as bad in any data set. If neither
        ``interpolate_bads`` nor ``drop_bads`` is `True`, in the output file, every
        channel marked as bad in at least one of the input files will be marked as bad,
        but no interpolation or dropping will be performed.

    Returns
    -------
    grand_average : Evoked | AverageTFR | Spectrum
        The grand average data. Same type as input.

    Notes
    -----
    Aggregating multitaper TFR datasets with a taper dimension such as for complex or
    phase data is not supported.

    .. versionadded:: 0.11.0
    """
    # check if all elements in the given list are evoked data
    from ..channels.channels import equalize_channels
    from ..evoked import Evoked
    from ..time_frequency import AverageTFR, Spectrum

    if not all_inst:
        raise ValueError(
            "Please pass a list of Evoked, AverageTFR, or Spectrum objects."
        )
    elif len(all_inst) == 1:
        warn("Only a single dataset was passed to mne.grand_average().")

    inst_type = type(all_inst[0])
    _validate_type(all_inst[0], (Evoked, AverageTFR, Spectrum), "All elements")
    for inst in all_inst:
        _validate_type(inst, inst_type, "All elements", "of the same type")

    # Copy channels to leave the original evoked datasets intact.
    all_inst = [inst.copy() for inst in all_inst]

    # Interpolates if necessary
    if isinstance(all_inst[0], Evoked):
        if interpolate_bads:
            all_inst = [
                inst.interpolate_bads() if len(inst.info["bads"]) > 0 else inst
                for inst in all_inst
            ]
        from ..evoked import combine_evoked as combine
    elif isinstance(all_inst[0], Spectrum):
        from ..time_frequency.spectrum import combine_spectrum as combine
    else:  # isinstance(all_inst[0], AverageTFR):
        from ..time_frequency.tfr import combine_tfr as combine

    if drop_bads:
        bads = list({b for inst in all_inst for b in inst.info["bads"]})
        if bads:
            for inst in all_inst:
                inst.drop_channels(bads)

    equalize_channels(all_inst, copy=False)
    # make grand_average object using combine_[evoked/tfr/spectrum]
    grand_average = combine(all_inst, weights="equal")
    # change the grand_average.nave to the number of datasets
    grand_average.nave = len(all_inst)
    # change comment field
    grand_average.comment = f"Grand average (n = {grand_average.nave})"
    return grand_average


class _HashableNdarray(np.ndarray):
    def __hash__(self):
        return object_hash(self)

    def __eq__(self, other):
        return NotImplementedError  # defer to hash


def _hashable_ndarray(x):
    return x.view(_HashableNdarray)


def object_hash(x, h=None):
    """Hash a reasonable python object.

    Parameters
    ----------
    x : object
        Object to hash. Can be anything comprised of nested versions of:
        {dict, list, tuple, ndarray, str, bytes, float, int, None}.
    h : hashlib HASH object | None
        Optional, object to add the hash to. None creates an MD5 hash.

    Returns
    -------
    digest : int
        The digest resulting from the hash.
    """
    if h is None:
        h = _empty_hash()
    if hasattr(x, "keys"):
        # dict-like types
        keys = _sort_keys(x)
        for key in keys:
            object_hash(key, h)
            object_hash(x[key], h)
    elif isinstance(x, bytes):
        # must come before "str" below
        h.update(x)
    elif isinstance(x, str | float | int | type(None)):
        h.update(str(type(x)).encode("utf-8"))
        h.update(str(x).encode("utf-8"))
    elif isinstance(x, np.ndarray | np.number | np.bool_):
        x = np.asarray(x)
        h.update(str(x.shape).encode("utf-8"))
        h.update(str(x.dtype).encode("utf-8"))
        h.update(x.tobytes())
    elif isinstance(x, datetime):
        object_hash(_dt_to_stamp(x))
    elif sparse.issparse(x):
        h.update(str(type(x)).encode("utf-8"))
        if not isinstance(x, sparse.csr_array | sparse.csc_array):
            raise RuntimeError(f"Unsupported sparse type {type(x)}")
        h.update(x.data.tobytes())
        h.update(x.indices.tobytes())
        h.update(x.indptr.tobytes())
    elif hasattr(x, "__len__"):
        # all other list-like types
        h.update(str(type(x)).encode("utf-8"))
        for xx in x:
            object_hash(xx, h)
    else:
        raise RuntimeError(f"unsupported type: {type(x)} ({x})")
    return int(h.hexdigest(), 16)


def object_size(x, memo=None):
    """Estimate the size of a reasonable python object.

    Parameters
    ----------
    x : object
        Object to approximate the size of.
        Can be anything comprised of nested versions of:
        {dict, list, tuple, ndarray, str, bytes, float, int, None}.
    memo : dict | None
        The memodict.

    Returns
    -------
    size : int
        The estimated size in bytes of the object.
    """
    # Note: this will not process object arrays properly (since those only)
    # hold references
    if memo is None:
        memo = dict()
    id_ = id(x)
    if id_ in memo:
        return 0  # do not add already existing ones
    if isinstance(x, bytes | str | int | float | type(None) | Path):
        size = sys.getsizeof(x)
    elif isinstance(x, np.ndarray):
        # On newer versions of NumPy, just doing sys.getsizeof(x) works,
        # but on older ones you always get something small :(
        size = sys.getsizeof(np.array([]))
        if x.base is None or id(x.base) not in memo:
            size += x.nbytes
    elif isinstance(x, np.generic):
        size = x.nbytes
    elif isinstance(x, dict):
        size = sys.getsizeof(x)
        for key, value in x.items():
            size += object_size(key, memo)
            size += object_size(value, memo)
    elif isinstance(x, list | tuple):
        size = sys.getsizeof(x) + sum(object_size(xx, memo) for xx in x)
    elif isinstance(x, datetime):
        size = object_size(_dt_to_stamp(x), memo)
    elif isinstance(x, date):
        size = 24  # 3 8-byte integers
    elif _is_sparse_cs(x):
        size = sum(sys.getsizeof(xx) for xx in [x, x.data, x.indices, x.indptr])
    else:
        raise RuntimeError(f"unsupported type: {type(x)} ({x})")
    memo[id_] = size
    return size


def _is_sparse_cs(x):
    return isinstance(
        x, sparse.csr_matrix | sparse.csc_matrix | sparse.csr_array | sparse.csc_array
    )


def _sort_keys(x):
    """Sort and return keys of dict."""
    keys = list(x.keys())  # note: not thread-safe
    idx = np.argsort([str(k) for k in keys])
    keys = [keys[ii] for ii in idx]
    return keys


def _array_equal_nan(a, b, allclose=False):
    try:
        if allclose:
            func = np.testing.assert_allclose
        else:
            func = np.testing.assert_array_equal
        func(a, b)
    except AssertionError:
        return False
    return True


def object_diff(a, b, pre="", *, allclose=False):
    """Compute all differences between two python variables.

    Parameters
    ----------
    a : object
        Currently supported: class, dict, list, tuple, ndarray,
        int, str, bytes, float, StringIO, BytesIO.
    b : object
        Must be same type as ``a``.
    pre : str
        String to prepend to each line.
    allclose : bool
        If True (default False), use assert_allclose.

    Returns
    -------
    diffs : str
        A string representation of the differences.
    """
    pd = _check_pandas_installed(strict=False)
    out = ""
    if type(a) is not type(b):
        # Deal with NamedInt and NamedFloat
        for sub in (int, float):
            if isinstance(a, sub) and isinstance(b, sub):
                break
        else:
            return f"{pre} type mismatch ({type(a)}, {type(b)})\n"
    if inspect.isclass(a):
        if inspect.isclass(b) and a != b:
            return f"{pre} class mismatch ({a}, {b})\n"
    elif isinstance(a, dict):
        k1s = _sort_keys(a)
        k2s = _sort_keys(b)
        m1 = set(k2s) - set(k1s)
        if len(m1):
            out += pre + f" left missing keys {m1}\n"
        for key in k1s:
            if key not in k2s:
                out += pre + f" right missing key {key}\n"
            else:
                out += object_diff(
                    a[key], b[key], pre=(pre + f"[{repr(key)}]"), allclose=allclose
                )
    elif isinstance(a, list | tuple):
        if len(a) != len(b):
            out += pre + f" length mismatch ({len(a)}, {len(b)})\n"
        else:
            for ii, (xx1, xx2) in enumerate(zip(a, b)):
                out += object_diff(xx1, xx2, pre + f"[{ii}]", allclose=allclose)
    elif isinstance(a, float):
        if not _array_equal_nan(a, b, allclose):
            out += pre + f" value mismatch ({a}, {b})\n"
    elif isinstance(a, str | int | bytes | np.generic):
        if a != b:
            out += pre + f" value mismatch ({a}, {b})\n"
    elif a is None:
        if b is not None:
            out += pre + f" left is None, right is not ({b})\n"
    elif isinstance(a, np.ndarray):
        if not _array_equal_nan(a, b, allclose):
            out += pre + " array mismatch\n"
    elif isinstance(a, StringIO | BytesIO):
        if a.getvalue() != b.getvalue():
            out += pre + " StringIO mismatch\n"
    elif isinstance(a, datetime | date):
        ts = (a - b).total_seconds()
        if ts != 0:
            out += pre + f" {a.__class__.__name__} mismatch ({a} vs {b} by {ts} sec)\n"
    elif sparse.issparse(a):
        # sparsity and sparse type of b vs a already checked above by type()
        if b.shape != a.shape:
            out += pre + (
                f" sparse matrix a and b shape mismatch ({a.shape} vs {b.shape})"
            )
        else:
            c = a - b
            c.eliminate_zeros()
            if c.nnz > 0:
                out += pre + (f" sparse matrix a and b differ on {c.nnz} elements")
    elif pd and isinstance(a, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(a, b)
        except AssertionError:
            out += pre + " DataFrame mismatch\n"
    elif hasattr(a, "__getstate__") and a.__getstate__() is not None:
        out += object_diff(a.__getstate__(), b.__getstate__(), pre, allclose=allclose)
    else:
        raise RuntimeError(pre + f": unsupported type {type(a)} ({a})")
    return out


class _PCA:
    """Principal component analysis (PCA)."""

    # Adapted from sklearn and stripped down to just use linalg.svd
    # and make it easier to later provide a "center" option if we want

    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = whiten

    def fit_transform(self, X, y=None):
        X = X.copy()
        U, S, _ = self._fit(X)
        U = U[:, : self.n_components_]

        if self.whiten:
            # X_new = X * V / S * sqrt(n_samples) = U * sqrt(n_samples)
            U *= sqrt(X.shape[0] - 1)
        else:
            # X_new = X * V = U * S * V^T * V = U * S
            U *= S[: self.n_components_]

        return U

    def fit(self, X):
        self._fit(X)

    def _fit(self, X):
        if self.n_components is None:
            n_components = min(X.shape)
        else:
            n_components = self.n_components
        n_samples, n_features = X.shape

        if n_components == "mle":
            if n_samples < n_features:
                raise ValueError(
                    "n_components='mle' is only supported if n_samples >= n_features"
                )
        elif not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError(
                f"n_components={repr(n_components)} must be between 0 and "
                f"min(n_samples, n_features)={repr(min(n_samples, n_features))} with "
                "svd_solver='full'"
            )
        elif n_components >= 1:
            if not isinstance(n_components, numbers.Integral | np.integer):
                raise ValueError(
                    f"n_components={repr(n_components)} must be of type int "
                    f"when greater than or equal to 1, "
                    f"was of type={repr(type(n_components))}"
                )

        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        U, S, V = _safe_svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, V = svd_flip(U, V)

        components_ = V

        # Get variance explained by singular values
        explained_variance_ = (S**2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = S.copy()  # Store the singular values.

        # Postprocess the number of components required
        if n_components == "mle":
            n_components = _infer_dimension_(explained_variance_, n_samples, n_features)
        elif 0 < n_components < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, n_components) + 1

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.0

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]

        return U, S, V


def _mask_to_onsets_offsets(mask):
    """Group boolean mask into contiguous onset:offset pairs."""
    assert mask.dtype == np.dtype(bool) and mask.ndim == 1
    mask = mask.astype(int)
    diff = np.diff(mask)
    onsets = np.where(diff > 0)[0] + 1
    if mask[0]:
        onsets = np.concatenate([[0], onsets])
    offsets = np.where(diff < 0)[0] + 1
    if mask[-1]:
        offsets = np.concatenate([offsets, [len(mask)]])
    assert len(onsets) == len(offsets)
    return onsets, offsets


def _julian_to_date(jd):
    """Convert Julian integer to a date object.

    Parameters
    ----------
    jd : int
        Julian date - number of days since julian day 0
        Julian day number 0 assigned to the day starting at
        noon on January 1, 4713 BC, proleptic Julian calendar
        November 24, 4714 BC, in the proleptic Gregorian calendar

    Returns
    -------
    jd_date : datetime
        Datetime representation of jd

    """
    # https://aa.usno.navy.mil/data/docs/JulianDate.php
    # Thursday, A.D. 1970 Jan 1 12:00:00.0  2440588.000000
    jd_t0 = 2440588
    datetime_t0 = datetime(1970, 1, 1, 12, 0, 0, 0, tzinfo=timezone.utc)

    dt = timedelta(days=(jd - jd_t0))
    return (datetime_t0 + dt).date()


def _date_to_julian(jd_date):
    """Convert datetime object to a Julian integer.

    Parameters
    ----------
    jd_date : date

    Returns
    -------
    jd : float
        Julian date corresponding to jd_date
        - number of days since julian day 0
        Julian day number 0 assigned to the day starting at
        noon on January 1, 4713 BC, proleptic Julian calendar
        November 24, 4714 BC, in the proleptic Gregorian calendar

    """
    # https://aa.usno.navy.mil/data/docs/JulianDate.php
    # Thursday, A.D. 1970 Jan 1 12:00:00.0  2440588.000000
    jd_t0 = 2440588
    date_t0 = date(1970, 1, 1)
    dt = jd_date - date_t0
    return jd_t0 + dt.days


def _check_dt(dt):
    if (
        not isinstance(dt, datetime)
        or dt.tzinfo is None
        or dt.tzinfo is not timezone.utc
    ):
        raise ValueError(f"Date must be datetime object in UTC: {repr(dt)}")


def _dt_to_stamp(inp_date):
    """Convert a datetime object to a timestamp."""
    _check_dt(inp_date)
    return int(inp_date.timestamp() // 1), inp_date.microsecond


def _stamp_to_dt(utc_stamp):
    """Convert timestamp to datetime object in Windows-friendly way."""
    # The min on windows is 86400
    stamp = [int(s) for s in utc_stamp]
    if len(stamp) == 1:  # In case there is no microseconds information
        stamp.append(0)
    return datetime.fromtimestamp(0, tz=timezone.utc) + timedelta(
        seconds=stamp[0], microseconds=stamp[1]
    )


class _ReuseCycle:
    """Cycle over a variable, preferring to reuse earlier indices.

    Requires the values in ``x`` to be hashable and unique. This holds
    nicely for matplotlib's color cycle, which gives HTML hex color strings.
    """

    def __init__(self, x):
        self.indices = list()
        self.popped = dict()
        assert len(x) > 0
        self.x = x

    def __iter__(self):
        while True:
            yield self.__next__()

    def __next__(self):
        if not len(self.indices):
            self.indices = list(range(len(self.x)))
            self.popped = dict()
        idx = self.indices.pop(0)
        val = self.x[idx]
        assert val not in self.popped
        self.popped[val] = idx
        return val

    def restore(self, val):
        try:
            idx = self.popped.pop(val)
        except KeyError:
            warn(f"Could not find value: {val}")
        else:
            loc = np.searchsorted(self.indices, idx)
            self.indices.insert(loc, idx)


def _arange_div_fallback(n, d):
    x = np.arange(n, dtype=np.float64)
    x /= d
    return x


if has_numba:

    @jit(fastmath=False)
    def _arange_div(n, d):
        out = np.empty(n, np.float64)
        for i in range(n):
            out[i] = i / d
        return out

else:  # pragma: no cover
    _arange_div = _arange_div_fallback


_LRU_CACHES = dict()
_LRU_CACHE_MAXSIZES = dict()


def _custom_lru_cache(maxsize):
    def dec(fun):
        fun_hash = hash(fun)
        this_cache = _LRU_CACHES[fun_hash] = dict()
        _LRU_CACHE_MAXSIZES[fun_hash] = maxsize

        def cache_fun(*args):
            hash_ = object_hash(args)
            if hash_ in this_cache:
                this_val = this_cache.pop(hash_)
            else:
                this_val = fun(*args)
            this_cache[hash_] = this_val  # (re)insert in last pos
            while len(this_cache) > _LRU_CACHE_MAXSIZES[fun_hash]:
                for key in this_cache:  # just an easy way to get first element
                    this_cache.pop(key)
                    break  # first in, first out
            return this_val

        return cache_fun

    return dec


def _array_repr(x):
    """Produce compact info about float ndarray x."""
    assert isinstance(x, np.ndarray), type(x)
    return f"shape : {x.shape}, range : [{np.nanmin(x):+0.2g}, {np.nanmax(x):+0.2g}]"


def _replace_md5(fname):
    """Replace a file based on MD5sum."""
    # adapted from sphinx-gallery
    assert fname.endswith(".new")
    fname_old = fname[:-4]
    if os.path.isfile(fname_old) and hashfunc(fname) == hashfunc(fname_old):
        os.remove(fname)
    else:
        shutil.move(fname, fname_old)
