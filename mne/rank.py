# -*- coding: utf-8 -*-
"""Some utility functions for rank estimation."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from .defaults import _handle_default
from .io.meas_info import _simplify_info
from .io.pick import (_picks_by_type, pick_info, pick_channels_cov,
                      _picks_to_idx)
from .io.proj import make_projector
from .utils import (logger, _compute_row_norms, _pl, _validate_type,
                    _apply_scaling_cov, _undo_scaling_cov,
                    _scaled_array, warn, _check_rank, verbose)


@verbose
def estimate_rank(data, tol='auto', return_singular=False, norm=True,
                  verbose=None):
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
    if norm:
        data = data.copy()  # operate on a copy
        norms = _compute_row_norms(data)
        data /= norms[:, np.newaxis]
    s = linalg.svdvals(data)
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
        ``scipy.linalg.orth`` (assuming np.float64 datatype) adjusted
        by a factor of 2.

    Returns
    -------
    rank : int
        The estimated rank.
    """
    if isinstance(tol, str):
        if tol not in ('auto', 'float32'):
            raise ValueError('tol must be "auto" or float, got %r' % (tol,))
        # XXX this should be float32 probably due to how we save and
        # load data, but it breaks test_make_inverse_operator (!)
        # The factor of 2 gets test_compute_covariance_auto_reg[None]
        # to pass without breaking minimum norm tests. :(
        # Passing 'float32' is a hack workaround for test_maxfilter_get_rank :(
        if tol == 'float32':
            eps = np.finfo(np.float32).eps
        else:
            eps = np.finfo(np.float64).eps
        max_s = np.amax(s)
        tol = len(s) * max_s * eps
        logger.info('    Using tolerance %0.2g (%0.2g eps * %d dim * %0.2g '
                    ' max singular value)' % (tol, eps, len(s), max_s))

    tol = float(tol)
    rank = np.sum(s > tol)
    return rank


def _estimate_rank_raw(raw, picks=None, tol=1e-4, scalings='norm',
                       with_ref_meg=False):
    """Aid the deprecation of raw.estimate_rank."""
    if picks is None:
        picks = _picks_to_idx(raw.info, picks, with_ref_meg=with_ref_meg)
    # conveniency wrapper to expose the expert "tol" option + scalings options
    return _estimate_rank_meeg_signals(
        raw[picks][0], pick_info(raw.info, picks), scalings, tol)


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
    if data.shape[1] < data.shape[0]:
        ValueError("You've got fewer samples than channels, your "
                   "rank estimate might be inaccurate.")
    with _scaled_array(data, picks_list, scalings):
        out = estimate_rank(data, tol=tol, norm=False,
                            return_singular=return_singular)
    rank = out[0] if isinstance(out, tuple) else out
    ch_type = ' + '.join(list(zip(*picks_list))[0])
    logger.info('    Estimated rank (%s): %d' % (ch_type, rank))
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
    logger.info('    Estimated rank (%s): %d' % (ch_type, rank))
    _undo_scaling_cov(data, picks_list, scalings)
    return out


@verbose
def _get_rank_sss(inst, msg='You should use data-based rank estimate instead',
                  verbose=None):
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
    # XXX this is too basic for movement compensated data
    # https://github.com/mne-tools/mne-python/issues/4676
    from .io.meas_info import Info
    info = inst if isinstance(inst, Info) else inst.info
    del inst

    proc_info = info.get('proc_history', [])
    if len(proc_info) > 1:
        logger.info('Found multiple SSS records. Using the first.')
    if len(proc_info) == 0 or 'max_info' not in proc_info[0] or \
            'in_order' not in proc_info[0]['max_info']['sss_info']:
        raise ValueError('Could not find Maxfilter information in '
                         'info["proc_history"]. %s' % msg)
    proc_info = proc_info[0]
    max_info = proc_info['max_info']
    inside = max_info['sss_info']['in_order']
    nfree = (inside + 1) ** 2 - 1
    nfree -= (len(max_info['sss_info']['components'][:nfree]) -
              max_info['sss_info']['components'][:nfree].sum())
    return nfree


def _info_rank(info, ch_type, picks, rank):
    if ch_type == 'meg' and rank != 'full':
        try:
            return _get_rank_sss(info)
        except ValueError:
            pass
    return len(picks)


def _compute_rank_int(inst, *args, **kwargs):
    """Wrap compute_rank but yield an int."""
    # XXX eventually we should unify how channel types are handled
    # so that we don't need to do this, or we do it everywhere.
    # Using pca=True in compute_whitener might help.
    return sum(compute_rank(inst, *args, **kwargs).values())


@verbose
def compute_rank(inst, rank=None, scalings=None, info=None, tol='auto',
                 proj=True, verbose=None):
    """Compute the rank of data or noise covariance.

    This function will normalize the rows of the data (typically
    channels or vertices) such that non-zero singular values
    should be close to one.

    Parameters
    ----------
    inst : instance of Raw, Epochs, or Covariance
        Raw measurements to compute the rank from or the covariance.
    %(rank_None)s
    scalings : dict | None (default None)
        Defaults to ``dict(mag=1e15, grad=1e13, eeg=1e6)``.
        These defaults will scale different channel types
        to comparable values.
    info : instance of Info | None
        The measurement info used to compute the covariance. It is
        only necessary if inst is a Covariance object (since this does
        not provide ``inst.info``).
    tol : float | str
        Tolerance. See ``estimate_rank``.
    proj : bool
        If True, all projs in ``inst`` and ``info`` will be applied or
        considered when ``rank=None`` or ``rank='info'``.
    %(verbose)s

    Returns
    -------
    rank : dict
        Estimated rank of the data for each channel type.
        To get the total rank, you can use ``sum(rank.values())``.

    Notes
    -----
    The ``rank`` parameter can be:

    :data:`python:None` (default)
        Rank will be estimated from the data after proper scaling of
        different channel types.
    ``'info'``
        Rank is inferred from `info`. If data have been processed
        with Maxwell filtering, the Maxwell filtering header is used.
        Otherwise, the channel counts themselves are used.
        In both cases, the number of projectors is subtracted from
        the (effective) number of channels in the data.
        For example, if Maxwell filtering reduces the rank to 68, with
        two projectors the returned value will be 68.
    ``'full'``
        Rank is assumed to be full, i.e. equal to the
        number of good channels. If a `Covariance` is passed, this can make
        sense if it has been (possibly improperly) regularized without taking
        into account the true data rank.

    .. versionadded:: 0.18
    """
    from .io.base import BaseRaw
    from .epochs import BaseEpochs
    from . import Covariance

    rank = _check_rank(rank)
    scalings = _handle_default('scalings_cov_rank', scalings)

    if isinstance(inst, Covariance):
        inst_type = 'covariance'
        if info is None:
            raise ValueError('info cannot be None if inst is a Covariance.')
        inst = pick_channels_cov(
            inst, set(inst['names']) & set(info['ch_names']))
        if info['ch_names'] != inst['names']:
            info = pick_info(info, [info['ch_names'].index(name)
                                    for name in inst['names']])
    else:
        info = inst.info
        inst_type = 'data'
    logger.info('Computing rank from %s with rank=%r' % (inst_type, rank))

    _validate_type(rank, (str, dict, None), 'rank')
    if isinstance(rank, str):  # string, either 'info' or 'full'
        rank_type = 'info'
        info_type = rank
        rank = dict()
    else:  # None or dict
        rank_type = 'estimated'
        if rank is None:
            rank = dict()

    simple_info = _simplify_info(info)
    picks_list = _picks_by_type(info, meg_combined=True, ref_meg=False,
                                exclude='bads')
    for ch_type, picks in picks_list:
        if ch_type in rank:
            continue
        ch_names = [info['ch_names'][pick] for pick in picks]
        n_chan = len(ch_names)
        if proj:
            proj_op, n_proj, _ = make_projector(info['projs'], ch_names)
        else:
            proj_op, n_proj = None, 0
        if rank_type == 'info':
            # use info
            rank[ch_type] = _info_rank(info, ch_type, picks, info_type)
            if info_type != 'full':
                rank[ch_type] -= n_proj
                logger.info('    %s: rank %d after %d projector%s applied to '
                            '%d channel%s'
                            % (ch_type.upper(), rank[ch_type],
                               n_proj, _pl(n_proj), n_chan, _pl(n_chan)))
            else:
                logger.info('    %s: rank %d from info'
                            % (ch_type.upper(), rank[ch_type]))
        else:
            # Use empirical estimation
            assert rank_type == 'estimated'
            if isinstance(inst, (BaseRaw, BaseEpochs)):
                if isinstance(inst, BaseRaw):
                    data = inst.get_data(picks, None, None,
                                         reject_by_annotation='omit')
                else:  # isinstance(inst, BaseEpochs):
                    data = inst.get_data()[:, picks, :]
                    data = np.concatenate(data, axis=1)
                if proj:
                    data = np.dot(proj_op, data)
                rank[ch_type] = _estimate_rank_meeg_signals(
                    data, pick_info(simple_info, picks), scalings, tol)
            else:
                assert isinstance(inst, Covariance)
                if inst['diag']:
                    rank[ch_type] = (inst['data'][picks] > 0).sum() - n_proj
                else:
                    data = inst['data'][picks][:, picks]
                    if proj:
                        data = np.dot(np.dot(proj_op, data), proj_op.T)
                    rank[ch_type] = _estimate_rank_meeg_cov(
                        data, pick_info(simple_info, picks), scalings, tol)
            this_info_rank = _info_rank(info, ch_type, picks, 'info')
            logger.info('    %s: rank %d computed from %d data channel%s '
                        'with %d projector%s'
                        % (ch_type.upper(), rank[ch_type], n_chan, _pl(n_chan),
                           n_proj, _pl(n_proj)))
            if rank[ch_type] > this_info_rank:
                warn('Something went wrong in the data-driven estimation of '
                     'the data rank as it exceeds the theoretical rank from '
                     'the info (%d > %d). Consider setting rank to "auto" or '
                     'setting it explicitly as an integer.' %
                     (rank[ch_type], this_info_rank))

    return rank
