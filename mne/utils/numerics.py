# -*- coding: utf-8 -*-
"""Some utility functions."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)


import hashlib
import operator
from math import ceil

import numpy as np
from scipy import linalg

from .logging import logger, warn
from .check import check_random_state, _ensure_int, _validate_type
from .docs import deprecated


def split_list(l, n, idx=False):
    """Split list in n (approx) equal pieces, possibly giving indices."""
    n = int(n)
    tot = len(l)
    sz = tot // n
    start = stop = 0
    for i in range(n - 1):
        stop += sz
        yield (np.arange(start, stop), l[start:stop]) if idx else l[start:stop]
        start += sz
    yield (np.arange(start, tot), l[start:]) if idx else l[start]


def array_split_idx(ary, indices_or_sections, axis=0, n_per_split=1):
    """Do what numpy.array_split does, but add indices."""
    # this only works for indices_or_sections as int
    indices_or_sections = _ensure_int(indices_or_sections)
    ary_split = np.array_split(ary, indices_or_sections, axis=axis)
    idx_split = np.array_split(np.arange(ary.shape[axis]), indices_or_sections)
    idx_split = (np.arange(sp[0] * n_per_split, (sp[-1] + 1) * n_per_split)
                 for sp in idx_split)
    return zip(idx_split, ary_split)


def create_chunks(sequence, size):
    """Generate chunks from a sequence.

    Parameters
    ----------
    sequence : iterable
        Any iterable object
    size : int
        The chunksize to be returned
    """
    return (sequence[p:p + size] for p in range(0, len(sequence), size))


def sum_squared(X):
    """Compute norm of an array.

    Parameters
    ----------
    X : array
        Data whose norm must be found

    Returns
    -------
    value : float
        Sum of squares of the input array X
    """
    X_flat = X.ravel(order='F' if np.isfortran(X) else 'C')
    return np.dot(X_flat, X_flat)


def estimate_rank(data, tol='auto', return_singular=False, norm=True):
    """Estimate the rank of data.

    This function will normalize the rows of the data (typically
    channels or vertices) such that non-zero singular values
    should be close to one.

    Parameters
    ----------
    data : array
        Data to estimate the rank of (should be 2-dimensional).
    tol : float | 'auto'
        Tolerance for singular values to consider non-zero in
        calculating the rank. The singular values are calculated
        in this method such that independent data are expected to
        have singular value around one. Can be 'auto' to use the
        same thresholding as ``scipy.linalg.orth``.
    return_singular : bool
        If True, also return the singular values that were used
        to determine the rank.
    norm : bool
        If True, data will be scaled by their estimated row-wise norm.
        Else data are assumed to be scaled. Defaults to True.

    Returns
    -------
    rank : int
        Estimated rank of the data.
    s : array
        If return_singular is True, the singular values that were
        thresholded to determine the rank are also returned.
    """
    data = data.copy()  # operate on a copy
    if norm is True:
        norms = _compute_row_norms(data)
        data /= norms[:, np.newaxis]
    s = linalg.svd(data, compute_uv=False, overwrite_a=True)
    rank = _estimate_rank_from_s(s, tol)
    if return_singular is True:
        return rank, s
    else:
        return rank


def _estimate_rank_from_s(s, tol='auto'):
    """Estimate the rank of a matrix from its singular values.

    Parameters
    ----------
    s : list of float
        The singular values of the matrix.
    tol : float | 'auto'
        Tolerance for singular values to consider non-zero in calculating the
        rank. Can be 'auto' to use the same thresholding as
        ``scipy.linalg.orth``.

    Returns
    -------
    rank : int
        The estimated rank.
    """
    if isinstance(tol, str):
        if tol != 'auto':
            raise ValueError('tol must be "auto" or float')
        eps = np.finfo(float).eps
        tol = len(s) * np.amax(s) * eps

    tol = float(tol)
    rank = np.sum(s > tol)
    return rank


def _compute_row_norms(data):
    """Compute scaling based on estimated norm."""
    norms = np.sqrt(np.sum(data ** 2, axis=1))
    norms[norms == 0] = 1.0
    return norms


def _reg_pinv(x, reg=0, rank='full', rcond=1e-15):
    """Compute a regularized pseudoinverse of a square matrix.

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
    x : ndarray, shape (n, n)
        Square matrix to invert.
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
    x_inv : ndarray, shape (n, n)
        The inverted matrix.
    loading_factor : float
        Value added to the diagonal of the matrix during regularization.
    rank : int
        If ``rank`` was set to an integer value, this value is returned,
        else the estimated rank of the matrix, before regularization, is
        returned.
    """
    if rank is not None and rank != 'full':
        rank = int(operator.index(rank))
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError('Input matrix must be square.')
    if not np.allclose(x, x.conj().T):
        raise ValueError('Input matrix must be Hermitian (symmetric)')

    # Decompose the matrix
    U, s, V = linalg.svd(x)

    # Estimate the rank before regularization
    tol = 'auto' if rcond == 'auto' else rcond * s.max()
    rank_before = _estimate_rank_from_s(s, tol)

    # Decompose the matrix again after regularization
    loading_factor = reg * np.mean(s)
    U, s, V = linalg.svd(x + loading_factor * np.eye(len(x)))

    # Estimate the rank after regularization
    tol = 'auto' if rcond == 'auto' else rcond * s.max()
    rank_after = _estimate_rank_from_s(s, tol)

    # Warn the user if both all parameters were kept at their defaults and the
    # matrix is rank deficient.
    if rank_after < len(x) and reg == 0 and rank == 'full' and rcond == 1e-15:
        warn('Covariance matrix is rank-deficient and no regularization is '
             'done.')
    elif isinstance(rank, int) and rank > len(x):
        raise ValueError('Invalid value for the rank parameter (%d) given '
                         'the shape of the input matrix (%d x %d).' %
                         (rank, x.shape[0], x.shape[1]))

    # Pick the requested number of singular values
    if rank is None:
        sel_s = s[:rank_before]
    elif rank == 'full':
        sel_s = s[:rank_after]
    else:
        sel_s = s[:rank]

    # Invert only non-zero singular values
    s_inv = np.zeros(s.shape)
    nonzero_inds = np.flatnonzero(sel_s != 0)
    if len(nonzero_inds) > 0:
        s_inv[nonzero_inds] = 1. / sel_s[nonzero_inds]

    # Compute the pseudo inverse
    x_inv = np.dot(V.T, s_inv[:, np.newaxis] * U.T)

    if rank is None or rank == 'full':
        return x_inv, loading_factor, rank_before
    else:
        return x_inv, loading_factor, rank


def _gen_events(n_epochs):
    """Generate event structure from number of epochs."""
    events = np.c_[np.arange(n_epochs), np.zeros(n_epochs, int),
                   np.ones(n_epochs, int)]
    return events


def _reject_data_segments(data, reject, flat, decim, info, tstep):
    """Reject data segments using peak-to-peak amplitude."""
    from ..epochs import _is_good
    from ..io.pick import channel_indices_by_type

    data_clean = np.empty_like(data)
    idx_by_type = channel_indices_by_type(info)
    step = int(ceil(tstep * info['sfreq']))
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
        if _is_good(data_buffer, info['ch_names'], idx_by_type, reject,
                    flat, ignore_chs=info['bads']):
            this_stop = this_start + data_buffer.shape[1]
            data_clean[:, this_start:this_stop] = data_buffer
            this_start += data_buffer.shape[1]
        else:
            logger.info("Artifact detected in [%d, %d]" % (first, last))
            drop_inds.append((first, last))
    data = data_clean[:, :this_stop]
    if not data.any():
        raise RuntimeError('No clean segment found. Please '
                           'consider updating your rejection '
                           'thresholds.')
    return data, drop_inds


def _get_inst_data(inst):
    """Get data view from MNE object instance like Raw, Epochs or Evoked."""
    from ..io.base import BaseRaw
    from ..epochs import BaseEpochs
    from .. import Evoked
    from ..time_frequency.tfr import _BaseTFR

    _validate_type(inst, (BaseRaw, BaseEpochs, Evoked, _BaseTFR), "Instance")
    if not inst.preload:
        inst.load_data()
    return inst._data


def compute_corr(x, y):
    """Compute pearson correlations between a vector and a matrix."""
    if len(x) == 0 or len(y) == 0:
        raise ValueError('x or y has zero length')
    X = np.array(x, float)
    Y = np.array(y, float)
    X -= X.mean(0)
    Y -= Y.mean(0)
    x_sd = X.std(0, ddof=1)
    # if covariance matrix is fully expanded, Y needs a
    # transpose / broadcasting else Y is correct
    y_sd = Y.std(0, ddof=1)[:, None if X.shape == Y.shape else Ellipsis]
    return (np.dot(X.T, Y) / float(len(X) - 1)) / (x_sd * y_sd)


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
    random_state : int | None
        Random seed for initializing the pseudo-random number generator.

    Returns
    -------
    randperm : ndarray, int
        Randomly permuted sequence between 0 and n-1.
    """
    rng = check_random_state(random_state)
    idx = rng.rand(n_samples)
    randperm = np.argsort(idx)
    return randperm


def _apply_scaling_array(data, picks_list, scalings):
    """Scale data type-dependently for estimation."""
    scalings = _check_scaling_inputs(data, picks_list, scalings)
    if isinstance(scalings, dict):
        picks_dict = dict(picks_list)
        scalings = [(picks_dict[k], v) for k, v in scalings.items()
                    if k in picks_dict]
        for idx, scaling in scalings:
            data[idx, :] *= scaling  # F - order
    else:
        data *= scalings[:, np.newaxis]  # F - order


def _invert_scalings(scalings):
    if isinstance(scalings, dict):
        scalings = dict((k, 1. / v) for k, v in scalings.items())
    elif isinstance(scalings, np.ndarray):
        scalings = 1. / scalings
    return scalings


def _undo_scaling_array(data, picks_list, scalings):
    scalings = _invert_scalings(_check_scaling_inputs(data, picks_list,
                                                      scalings))
    return _apply_scaling_array(data, picks_list, scalings)


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
            raise ValueError('Scaling factors and data are of incompatible '
                             'shape')
        scales = scalings
    elif scalings is None:
        pass
    else:
        raise RuntimeError('Arff...')
    if scales is not None:
        assert np.sum(scales == 0.) == 0
        data *= (scales[None, :] * scales[:, None])


def _undo_scaling_cov(data, picks_list, scalings):
    scalings = _invert_scalings(_check_scaling_inputs(data, picks_list,
                                                      scalings))
    return _apply_scaling_cov(data, picks_list, scalings)


def _check_scaling_inputs(data, picks_list, scalings):
    """Aux function."""
    rescale_dict_ = dict(mag=1e15, grad=1e13, eeg=1e6)

    scalings_ = None
    if isinstance(scalings, str) and scalings == 'norm':
        scalings_ = 1. / _compute_row_norms(data)
    elif isinstance(scalings, dict):
        rescale_dict_.update(scalings)
        scalings_ = rescale_dict_
    elif isinstance(scalings, np.ndarray):
        scalings_ = scalings
    elif scalings is None:
        pass
    else:
        raise NotImplementedError("No way! That's not a rescaling "
                                  'option: %s' % scalings)
    return scalings_


@deprecated('mne.utils.md5sum will be deprecated in 0.19, please use '
            'mne.utils.hashfunc(... , hash_type="md5") instead.')
def md5sum(fname, block_size=1048576):  # 2 ** 20
    """Calculate the md5sum for a file.

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
    md5 = hashlib.md5()
    with open(fname, 'rb') as fid:
        while True:
            data = fid.read(block_size)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


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
    if hash_type == "md5":
        hasher = hashlib.md5()
    elif hash_type == "sha1":
        hasher = hashlib.sha1()
    with open(fname, 'rb') as fid:
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
    slices = [slice(t, t + length, 1) for t in
              range(start, stop - length + 1, step)]
    return slices


def _time_mask(times, tmin=None, tmax=None, sfreq=None, raise_error=True):
    """Safely find sample boundaries."""
    orig_tmin = tmin
    orig_tmax = tmax
    tmin = -np.inf if tmin is None else tmin
    tmax = np.inf if tmax is None else tmax
    if not np.isfinite(tmin):
        tmin = times[0]
    if not np.isfinite(tmax):
        tmax = times[-1]
    if sfreq is not None:
        # Push to a bit past the nearest sample boundary first
        sfreq = float(sfreq)
        tmin = int(round(tmin * sfreq)) / sfreq - 0.5 / sfreq
        tmax = int(round(tmax * sfreq)) / sfreq + 0.5 / sfreq
    if raise_error and tmin > tmax:
        raise ValueError('tmin (%s) must be less than or equal to tmax (%s)'
                         % (orig_tmin, orig_tmax))
    mask = (times >= tmin)
    mask &= (times <= tmax)
    if raise_error and not mask.any():
        raise ValueError('No samples remain when using tmin=%s and tmax=%s '
                         '(original time bounds are [%s, %s])'
                         % (orig_tmin, orig_tmax, times[0], times[-1]))
    return mask


def grand_average(all_inst, interpolate_bads=True, drop_bads=True):
    """Make grand average of a list evoked or AverageTFR data.

    For evoked data, the function interpolates bad channels based on
    `interpolate_bads` parameter. If `interpolate_bads` is True, the grand
    average file will contain good channels and the bad channels interpolated
    from the good MEG/EEG channels.
    For AverageTFR data, the function takes the subset of channels not marked
    as bad in any of the instances.

    The grand_average.nave attribute will be equal to the number
    of evoked datasets used to calculate the grand average.

    Note: Grand average evoked should not be used for source localization.

    Parameters
    ----------
    all_inst : list of Evoked or AverageTFR data
        The evoked datasets.
    interpolate_bads : bool
        If True, bad MEG and EEG channels are interpolated. Ignored for
        AverageTFR.
    drop_bads : bool
        If True, drop all bad channels marked as bad in any data set.
        If neither interpolate_bads nor drop_bads is True, in the output file,
        every channel marked as bad in at least one of the input files will be
        marked as bad, but no interpolation or dropping will be performed.

    Returns
    -------
    grand_average : Evoked | AverageTFR
        The grand average data. Same type as input.

    Notes
    -----
    .. versionadded:: 0.11.0
    """
    # check if all elements in the given list are evoked data
    from ..evoked import Evoked
    from ..time_frequency import AverageTFR
    from ..channels.channels import equalize_channels
    assert len(all_inst) > 1
    inst_type = type(all_inst[0])
    _validate_type(all_inst[0], (Evoked, AverageTFR), 'All elements')
    for inst in all_inst:
        _validate_type(inst, inst_type, 'All elements', 'of the same type')

    # Copy channels to leave the original evoked datasets intact.
    all_inst = [inst.copy() for inst in all_inst]

    # Interpolates if necessary
    if isinstance(all_inst[0], Evoked):
        if interpolate_bads:
            all_inst = [inst.interpolate_bads() if len(inst.info['bads']) > 0
                        else inst for inst in all_inst]
        equalize_channels(all_inst)  # apply equalize_channels
        from ..evoked import combine_evoked as combine
    else:  # isinstance(all_inst[0], AverageTFR):
        from ..time_frequency.tfr import combine_tfr as combine

    if drop_bads:
        bads = list(set((b for inst in all_inst for b in inst.info['bads'])))
        if bads:
            for inst in all_inst:
                inst.drop_channels(bads)

    # make grand_average object using combine_[evoked/tfr]
    grand_average = combine(all_inst, weights='equal')
    # change the grand_average.nave to the number of Evokeds
    grand_average.nave = len(all_inst)
    # change comment field
    grand_average.comment = "Grand average (n = %d)" % grand_average.nave
    return grand_average
