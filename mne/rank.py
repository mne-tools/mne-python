# -*- coding: utf-8 -*-
"""Some utility functions for rank estimation."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

# import operator

from numbers import Integral

import numpy as np
from scipy import linalg

from .io.pick import _picks_by_type, _pick_data_channels
from .defaults import _handle_default
from .utils import logger
from .utils import _compute_row_norms
from .utils import _apply_scaling_cov, _undo_scaling_cov
from .utils import _apply_scaling_array, _undo_scaling_array
from .utils import warn, _check_rank


# XXX : Add tests
def estimate_rank_by_ratio(X, ratio=1.0e4):
    X = np.asarray(X)
    assert X.ndim == 2
    # Compute the singular values
    sing = linalg.svd(X, compute_uv=False)
    # If the largest singular value is zero, the rank is zero
    if sing[0] < np.finfo(X.dtype).eps:
        return 0
    # Scale and check for zero singular values within the numerical
    # precision and remove them
    sing_scaled = sing / sing[0]
    n_nonzero = np.count_nonzero(sing_scaled > np.finfo(X.dtype).eps)
    sing = sing[:n_nonzero]
    # Search for abrupt changes in the singular value spectra by
    # computing the ratios of consecutive singular values
    sing_shifted = np.hstack([sing[0], sing[:-1]])
    sing_ratio = sing_shifted / sing
    # Check if any/some of the ratios exceed the threshold
    ex = np.where(sing_ratio > ratio)[0]
    if len(ex) == 0:
        return n_nonzero
    else:
        return ex[0]


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


def _estimate_rank_meeg_signals(data, info, scalings, tol='auto',
                                return_singular=False):
    """Estimate rank for M/EEG data.

    Parameters
    ----------
    data : np.ndarray of float, shape(n_channels, n_samples)
        The M/EEG signals.
    info : Info
        The measurement info.
    scalings : dict | 'norm' | np.ndarray | None
        The rescaling method to be applied. If dict, it will override the
        following default dict:

            dict(mag=1e15, grad=1e13, eeg=1e6)

        If 'norm' data will be scaled by channel-wise norms. If array,
        pre-specified norms will be used. If None, no scaling will be applied.
    tol : float | str
        Tolerance. See ``estimate_rank``.
    return_singular : bool
        If True, also return the singular values that were used
        to determine the rank.

    Returns
    -------
    rank : int
        Estimated rank of the data.
    s : array
        If return_singular is True, the singular values that were
        thresholded to determine the rank are also returned.
    """
    picks_list = _picks_by_type(info)
    _apply_scaling_array(data, picks_list, scalings)
    if data.shape[1] < data.shape[0]:
        ValueError("You've got fewer samples than channels, your "
                   "rank estimate might be inaccurate.")
    out = estimate_rank(data, tol=tol, norm=False,
                        return_singular=return_singular)
    rank = out[0] if isinstance(out, tuple) else out
    ch_type = ' + '.join(list(zip(*picks_list))[0])
    logger.info('estimated rank (%s): %d' % (ch_type, rank))
    _undo_scaling_array(data, picks_list, scalings)
    return out


def _estimate_rank_meeg_cov(data, info, scalings, tol='auto',
                            return_singular=False):
    """Estimate rank of M/EEG covariance data, given the covariance.

    Parameters
    ----------
    data : np.ndarray of float, shape (n_channels, n_channels)
        The M/EEG covariance.
    info : Info
        The measurement info.
    scalings : dict | 'norm' | np.ndarray | None
        The rescaling method to be applied. If dict, it will override the
        following default dict:

            dict(mag=1e12, grad=1e11, eeg=1e5)

        If 'norm' data will be scaled by channel-wise norms. If array,
        pre-specified norms will be used. If None, no scaling will be applied.
    tol : float | str
        Tolerance. See ``estimate_rank``.
    return_singular : bool
        If True, also return the singular values that were used
        to determine the rank.

    Returns
    -------
    rank : int
        Estimated rank of the data.
    s : array
        If return_singular is True, the singular values that were
        thresholded to determine the rank are also returned.
    """
    picks_list = _picks_by_type(info)
    scalings = _handle_default('scalings_cov_rank', scalings)
    _apply_scaling_cov(data, picks_list, scalings)
    if data.shape[1] < data.shape[0]:
        ValueError("You've got fewer samples than channels, your "
                   "rank estimate might be inaccurate.")
    out = estimate_rank(data, tol=tol, norm=False,
                        return_singular=return_singular)
    rank = out[0] if isinstance(out, tuple) else out
    ch_type = ' + '.join(list(zip(*picks_list))[0])
    logger.info('estimated rank (%s): %d' % (ch_type, rank))
    _undo_scaling_cov(data, picks_list, scalings)
    return out


def _get_sss_rank(sss):
    """Get SSS rank."""
    inside = sss['sss_info']['in_order']
    nfree = (inside + 1) ** 2 - 1
    nfree -= (len(sss['sss_info']['components'][:nfree]) -
              sss['sss_info']['components'][:nfree].sum())
    return nfree


def _get_rank_sss(inst):
    """Look up rank from SSS data.

    .. note::
        Throws an error if SSS has not been applied.

    Parameters
    ----------
    inst : instance of Raw, Epochs or Evoked, or Info
        Any MNE object with an .info attribute

    Returns
    -------
    rank : int
        The numerical rank as predicted by the number of SSS
        components.
    """
    from .io.meas_info import Info
    info = inst if isinstance(inst, Info) else inst.info
    del inst

    max_infos = list()
    for proc_info in info.get('proc_history', list()):
        max_info = proc_info.get('max_info')
        if max_info is not None:
            if len(max_info) > 0:
                max_infos.append(max_info)
            elif len(max_info) > 1:
                logger.info('found multiple SSS records. Using the first.')
            elif len(max_info) == 0:
                raise ValueError(
                    'Did not find any SSS record. You should use data-based '
                    'rank estimate instead')
    if len(max_infos) > 0:
        max_info = max_infos[0]
    else:
        raise ValueError('There is no `max_info` here. Sorry.')
    return _get_sss_rank(max_info)


def _get_data_channels_picks(info):
    with_ref_meg = False  # XXX To test
    picks = _pick_data_channels(info, exclude='bads',
                                with_ref_meg=with_ref_meg)
    return picks


def _get_n_data_channels(info):
    """Returns the number of good data channels."""
    n_data_channels = len(_get_data_channels_picks(info))
    return n_data_channels


def _get_rank_info(info):
    try:
        rank = _get_rank_sss(info)
    except ValueError:
        rank = _get_n_data_channels(info)
    rank -= len(info['bads']) + len(info['projs'])
    return rank


# compute whitener says rank can be a dict to specify
# per channel type...
def compute_rank(inst, scalings=None, rank='auto', info=None):
    """Compute the rank of data or noise covariance.

    This function will normalize the rows of the data (typically
    channels or vertices) such that non-zero singular values
    should be close to one.

    Parameters
    ----------
    inst : instance of Raw, Epochs or Covariance
        Raw measurements to compute the rank from or the covariance.
    scalings : dict | None (default None)
        Defaults to ``dict(mag=1e15, grad=1e13, eeg=1e6)``.
        These defaults will scale different channel types
        to comparable values.
    rank : int | None | 'full' | 'auto'
        This controls the rank computation that can be read from the
        measurement info or estimated from the data:

            If int, then this value is returned as is. Nothing
            is estimated or read from the info.

            If ``None``, the rank will be automatically estimated
            from the data after proper scaling of the different
            channel types.

            If ``auto``, the rank is inferred from the info if available
            otherwise it's estimated. It is typically estimated from the
            Maxwell Filter header if present, and using the available
            'projs'.

            If 'full', the rank is assumed to be full, i.e. equal to the
            number of good channels. If a Covariance
            is passed this can make sense if it has been regularized.

        The defaults is 'auto'.
    info : instance of Info | None
        The measurement info used to compute the covariance. It is
        only necessary if inst is a Covariance object.

    Returns
    -------
    rank : int
        Estimated rank of the data.
    """
    from .io.base import BaseRaw
    from .epochs import BaseEpochs
    from . import Covariance

    rank = _check_rank(rank)

    if not isinstance(inst, Covariance):
        info = inst.info

    if info is None:
        raise ValueError('info cannot be None if inst is a Covariance.')

    if rank is 'auto':
        rank = _get_rank_info(info)
    elif rank is None:
        picks = _get_data_channels_picks(info)
        if isinstance(inst, BaseRaw):
            reject_by_annotation = 'omit'
            data = inst.get_data(picks, None, None, reject_by_annotation)
        elif isinstance(inst, BaseEpochs):
            data = inst.get_data()[:, picks, :]
            data = np.concatenate(data, axis=1)
        info_rank = _get_rank_info(info)
        rank = estimate_rank(data)
        if rank > info_rank:
            warn('Something went wrong in the data-driven estimation of the'
                 ' data rank as it exceeds the theoretical rank from the '
                 'info (%d > %d). Consider setting rank to "auto" or '
                 'setting it explicitely as an integer.' %
                 (rank, info_rank))
    elif rank == 'full':
        rank = _get_n_data_channels(info)
    elif not isinstance(rank, Integral):
        raise ValueError("'rank' should be 'auto', 'full', None or int. "
                         "Got %s." % rank)

    return rank
