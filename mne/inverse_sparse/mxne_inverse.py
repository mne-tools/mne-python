# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Daniel Strohmeier <daniel.strohmeier@gmail.com>
#
# License: Simplified BSD

from copy import deepcopy
import numpy as np
from scipy import linalg, signal

from ..source_estimate import SourceEstimate
from ..minimum_norm.inverse import combine_xyz, _prepare_forward
from ..minimum_norm.inverse import _check_reference
from ..forward import compute_orient_prior, is_fixed_orient, _to_fixed_ori
from ..io.pick import pick_channels_evoked
from ..io.proj import deactivate_proj
from ..utils import logger, verbose
from ..externals.six.moves import xrange as range

from .mxne_optim import (mixed_norm_solver, iterative_mixed_norm_solver,
                         norm_l2inf, tf_mixed_norm_solver)


@verbose
def _prepare_weights(forward, gain, source_weighting, weights, weights_min):
    mask = None
    if isinstance(weights, SourceEstimate):
        # weights = np.sqrt(np.sum(weights.data ** 2, axis=1))
        weights = np.max(np.abs(weights.data), axis=1)
    weights_max = np.max(weights)
    if weights_min > weights_max:
        raise ValueError('weights_min > weights_max (%s > %s)' %
                         (weights_min, weights_max))
    weights_min = weights_min / weights_max
    weights = weights / weights_max
    n_dip_per_pos = 1 if is_fixed_orient(forward) else 3
    weights = np.ravel(np.tile(weights, [n_dip_per_pos, 1]).T)
    if len(weights) != gain.shape[1]:
        raise ValueError('weights do not have the correct dimension '
                         ' (%d != %d)' % (len(weights), gain.shape[1]))
    if len(source_weighting.shape) == 1:
        source_weighting *= weights
    else:
        source_weighting *= weights[:, None]
    gain *= weights[None, :]

    if weights_min is not None:
        mask = (weights > weights_min)
        gain = gain[:, mask]
        n_sources = np.sum(mask) // n_dip_per_pos
        logger.info("Reducing source space to %d sources" % n_sources)

    return gain, source_weighting, mask


@verbose
def _prepare_gain_column(forward, info, noise_cov, pca, depth, loose, weights,
                         weights_min, verbose=None):
    gain_info, gain, _, whitener, _ = _prepare_forward(forward, info,
                                                       noise_cov, pca)

    logger.info('Whitening lead field matrix.')
    gain = np.dot(whitener, gain)

    if depth is not None:
        depth_prior = np.sum(gain ** 2, axis=0) ** depth
        source_weighting = np.sqrt(depth_prior ** -1.)
    else:
        source_weighting = np.ones(gain.shape[1], dtype=gain.dtype)

    if loose is not None and loose != 1.0:
        source_weighting *= np.sqrt(compute_orient_prior(forward, loose))

    gain *= source_weighting[None, :]

    if weights is None:
        mask = None
    else:
        gain, source_weighting, mask = _prepare_weights(forward, gain,
                                                        source_weighting,
                                                        weights, weights_min)

    return gain, gain_info, whitener, source_weighting, mask


def _prepare_gain(forward, info, noise_cov, pca, depth, loose, weights,
                  weights_min, verbose=None):
    if not isinstance(depth, float):
        raise ValueError('Invalid depth parameter. '
                         'A float is required (got %s).'
                         % type(depth))
    elif depth < 0.0:
        raise ValueError('Depth parameter must be positive (got %s).'
                         % depth)

    gain, gain_info, whitener, source_weighting, mask = \
        _prepare_gain_column(forward, info, noise_cov, pca, depth,
                             loose, weights, weights_min)

    return gain, gain_info, whitener, source_weighting, mask


def _reapply_source_weighting(X, source_weighting, active_set,
                              n_dip_per_pos):
    X *= source_weighting[active_set][:, None]
    return X


def _compute_residual(forward, evoked, X, active_set, info):
    # OK, picking based on row_names is safe
    sel = [forward['sol']['row_names'].index(c) for c in info['ch_names']]
    residual = evoked.copy()
    residual = pick_channels_evoked(residual, include=info['ch_names'])
    r_tmp = residual.copy()

    r_tmp.data = np.dot(forward['sol']['data'][sel, :][:, active_set], X)

    # Take care of proj
    active_projs = list()
    non_active_projs = list()
    for p in evoked.info['projs']:
        if p['active']:
            active_projs.append(p)
        else:
            non_active_projs.append(p)

    if len(active_projs) > 0:
        r_tmp.info['projs'] = deactivate_proj(active_projs, copy=True)
        r_tmp.apply_proj()
        r_tmp.add_proj(non_active_projs, remove_existing=False)

    residual.data -= r_tmp.data

    return residual


@verbose
def _make_sparse_stc(X, active_set, forward, tmin, tstep,
                     active_is_idx=False, verbose=None):
    if not is_fixed_orient(forward):
        logger.info('combining the current components...')
        X = combine_xyz(X)

    if not active_is_idx:
        active_idx = np.where(active_set)[0]
    else:
        active_idx = active_set

    n_dip_per_pos = 1 if is_fixed_orient(forward) else 3
    if n_dip_per_pos > 1:
        active_idx = np.unique(active_idx // n_dip_per_pos)

    src = forward['src']

    n_lh_points = len(src[0]['vertno'])
    lh_vertno = src[0]['vertno'][active_idx[active_idx < n_lh_points]]
    rh_vertno = src[1]['vertno'][active_idx[active_idx >= n_lh_points] -
                                 n_lh_points]
    vertices = [lh_vertno, rh_vertno]
    stc = SourceEstimate(X, vertices=vertices, tmin=tmin, tstep=tstep)
    return stc


@verbose
def mixed_norm(evoked, forward, noise_cov, alpha, loose=0.2, depth=0.8,
               maxit=3000, tol=1e-4, active_set_size=10, pca=True,
               debias=True, time_pca=True, weights=None, weights_min=None,
               solver='auto', n_mxne_iter=1, return_residual=False,
               verbose=None):
    """Mixed-norm estimate (MxNE) and iterative reweighted MxNE (irMxNE).

    Compute L1/L2 mixed-norm solution [1]_ or L0.5/L2 [2]_ mixed-norm
    solution on evoked data.

    Parameters
    ----------
    evoked : instance of Evoked or list of instances of Evoked
        Evoked data to invert.
    forward : dict
        Forward operator.
    noise_cov : instance of Covariance
        Noise covariance to compute whitener.
    alpha : float
        Regularization parameter.
    loose : float in [0, 1]
        Value that weights the source variances of the dipole components
        that are parallel (tangential) to the cortical surface. If loose
        is 0 or None then the solution is computed with fixed orientation.
        If loose is 1, it corresponds to free orientations.
    depth: None | float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed.
    maxit : int
        Maximum number of iterations.
    tol : float
        Tolerance parameter.
    active_set_size : int | None
        Size of active set increment. If None, no active set strategy is used.
    pca : bool
        If True the rank of the data is reduced to true dimension.
    debias : bool
        Remove coefficient amplitude bias due to L1 penalty.
    time_pca : bool or int
        If True the rank of the concatenated epochs is reduced to
        its true dimension. If is 'int' the rank is limited to this value.
    weights : None | array | SourceEstimate
        Weight for penalty in mixed_norm. Can be None, a
        1d array with shape (n_sources,), or a SourceEstimate (e.g. obtained
        with wMNE, dSPM, or fMRI).
    weights_min : float
        Do not consider in the estimation sources for which weights
        is less than weights_min.
    solver : 'prox' | 'cd' | 'bcd' | 'auto'
        The algorithm to use for the optimization. 'prox' stands for
        proximal interations using the FISTA algorithm, 'cd' uses
        coordinate descent, and 'bcd' applies block coordinate descent.
        'cd' is only available for fixed orientation.
    n_mxne_iter : int
        The number of MxNE iterations. If > 1, iterative reweighting
        is applied.
    return_residual : bool
        If True, the residual is returned as an Evoked instance.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | list of SourceEstimate
        Source time courses for each evoked data passed as input.
    residual : instance of Evoked
        The residual a.k.a. data not explained by the sources.
        Only returned if return_residual is True.

    See Also
    --------
    tf_mixed_norm

    References
    ----------
    .. [1] Gramfort A., Kowalski M. and Hamalainen, M.,
       "Mixed-norm estimates for the M/EEG inverse problem using accelerated
       gradient methods", Physics in Medicine and Biology, 2012.
       http://dx.doi.org/10.1088/0031-9155/57/7/1937

    .. [2] Strohmeier D., Haueisen J., and Gramfort A.,
       "Improved MEG/EEG source localization with reweighted mixed-norms",
       4th International Workshop on Pattern Recognition in Neuroimaging,
       Tuebingen, 2014.
    """
    if n_mxne_iter < 1:
        raise ValueError('MxNE has to be computed at least 1 time. '
                         'Requires n_mxne_iter >= 1, got %d' % n_mxne_iter)

    if not isinstance(evoked, list):
        evoked = [evoked]

    _check_reference(evoked[0])

    all_ch_names = evoked[0].ch_names
    if not all(all_ch_names == evoked[i].ch_names
               for i in range(1, len(evoked))):
        raise Exception('All the datasets must have the same good channels.')

    # put the forward solution in fixed orientation if it's not already
    if loose is None and not is_fixed_orient(forward):
        forward = deepcopy(forward)
        _to_fixed_ori(forward)

    gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked[0].info, noise_cov, pca, depth, loose, weights,
        weights_min)

    sel = [all_ch_names.index(name) for name in gain_info['ch_names']]
    M = np.concatenate([e.data[sel] for e in evoked], axis=1)

    # Whiten data
    logger.info('Whitening data matrix.')
    M = np.dot(whitener, M)

    if time_pca:
        U, s, Vh = linalg.svd(M, full_matrices=False)
        if not isinstance(time_pca, bool) and isinstance(time_pca, int):
            U = U[:, :time_pca]
            s = s[:time_pca]
            Vh = Vh[:time_pca]
        M = U * s

    # Scaling to make setting of alpha easy
    n_dip_per_pos = 1 if is_fixed_orient(forward) else 3
    alpha_max = norm_l2inf(np.dot(gain.T, M), n_dip_per_pos, copy=False)
    alpha_max *= 0.01
    gain /= alpha_max
    source_weighting /= alpha_max

    if n_mxne_iter == 1:
        X, active_set, E = mixed_norm_solver(
            M, gain, alpha, maxit=maxit, tol=tol,
            active_set_size=active_set_size, n_orient=n_dip_per_pos,
            debias=debias, solver=solver, verbose=verbose)
    else:
        X, active_set, E = iterative_mixed_norm_solver(
            M, gain, alpha, n_mxne_iter, maxit=maxit, tol=tol,
            n_orient=n_dip_per_pos, active_set_size=active_set_size,
            debias=debias, solver=solver, verbose=verbose)

    if mask is not None:
        active_set_tmp = np.zeros(len(mask), dtype=np.bool)
        active_set_tmp[mask] = active_set
        active_set = active_set_tmp
        del active_set_tmp

    if time_pca:
        X = np.dot(X, Vh)

    if active_set.sum() == 0:
        raise Exception("No active dipoles found. alpha is too big.")

    # Reapply weights to have correct unit
    X = _reapply_source_weighting(X, source_weighting,
                                  active_set, n_dip_per_pos)

    stcs = list()
    residual = list()
    cnt = 0
    for e in evoked:
        tmin = e.times[0]
        tstep = 1.0 / e.info['sfreq']
        Xe = X[:, cnt:(cnt + len(e.times))]
        stc = _make_sparse_stc(Xe, active_set, forward, tmin, tstep)
        stcs.append(stc)
        cnt += len(e.times)

        if return_residual:
            residual.append(_compute_residual(forward, e, Xe, active_set,
                            gain_info))

    logger.info('[done]')

    if len(stcs) == 1:
        out = stcs[0]
        if return_residual:
            residual = residual[0]
    else:
        out = stcs

    if return_residual:
        out = out, residual

    return out


def _window_evoked(evoked, size):
    """Window evoked (size in seconds)."""
    if isinstance(size, (float, int)):
        lsize = rsize = float(size)
    else:
        lsize, rsize = size
    evoked = evoked.copy()
    sfreq = float(evoked.info['sfreq'])
    lsize = int(lsize * sfreq)
    rsize = int(rsize * sfreq)
    lhann = signal.hann(lsize * 2)
    rhann = signal.hann(rsize * 2)
    window = np.r_[lhann[:lsize],
                   np.ones(len(evoked.times) - lsize - rsize),
                   rhann[-rsize:]]
    evoked.data *= window[None, :]
    return evoked


@verbose
def tf_mixed_norm(evoked, forward, noise_cov, alpha_space, alpha_time,
                  loose=0.2, depth=0.8, maxit=3000, tol=1e-4,
                  weights=None, weights_min=None, pca=True, debias=True,
                  wsize=64, tstep=4, window=0.02, return_residual=False,
                  verbose=None):
    """Time-Frequency Mixed-norm estimate (TF-MxNE).

    Compute L1/L2 + L1 mixed-norm solution on time-frequency
    dictionary. Works with evoked data.

    References:

    A. Gramfort, D. Strohmeier, J. Haueisen, M. Hamalainen, M. Kowalski
    Time-Frequency Mixed-Norm Estimates: Sparse M/EEG imaging with
    non-stationary source activations
    Neuroimage, Volume 70, 15 April 2013, Pages 410-422, ISSN 1053-8119,
    DOI: 10.1016/j.neuroimage.2012.12.051.

    A. Gramfort, D. Strohmeier, J. Haueisen, M. Hamalainen, M. Kowalski
    Functional Brain Imaging with M/EEG Using Structured Sparsity in
    Time-Frequency Dictionaries
    Proceedings Information Processing in Medical Imaging
    Lecture Notes in Computer Science, 2011, Volume 6801/2011,
    600-611, DOI: 10.1007/978-3-642-22092-0_49
    http://dx.doi.org/10.1007/978-3-642-22092-0_49

    Parameters
    ----------
    evoked : instance of Evoked
        Evoked data to invert.
    forward : dict
        Forward operator.
    noise_cov : instance of Covariance
        Noise covariance to compute whitener.
    alpha_space : float in [0, 100]
        Regularization parameter for spatial sparsity. If larger than 100,
        then no source will be active.
    alpha_time : float in [0, 100]
        Regularization parameter for temporal sparsity. It set to 0,
        no temporal regularization is applied. It this case, TF-MxNE is
        equivalent to MxNE with L21 norm.
    loose : float in [0, 1]
        Value that weights the source variances of the dipole components
        that are parallel (tangential) to the cortical surface. If loose
        is 0 or None then the solution is computed with fixed orientation.
        If loose is 1, it corresponds to free orientations.
    depth: None | float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed.
    maxit : int
        Maximum number of iterations.
    tol : float
        Tolerance parameter.
    weights: None | array | SourceEstimate
        Weight for penalty in mixed_norm. Can be None or
        1d array of length n_sources or a SourceEstimate e.g. obtained
        with wMNE or dSPM or fMRI.
    weights_min: float
        Do not consider in the estimation sources for which weights
        is less than weights_min.
    pca: bool
        If True the rank of the data is reduced to true dimension.
    debias: bool
        Remove coefficient amplitude bias due to L1 penalty.
    wsize: int
        Length of the STFT window in samples (must be a multiple of 4).
    tstep: int
        Step between successive windows in samples (must be a multiple of 2,
        a divider of wsize and smaller than wsize/2) (default: wsize/2).
    window : float or (float, float)
        Length of time window used to take care of edge artifacts in seconds.
        It can be one float or float if the values are different for left
        and right window length.
    return_residual : bool
        If True, the residual is returned as an Evoked instance.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : instance of SourceEstimate
        Source time courses.
    residual : instance of Evoked
        The residual a.k.a. data not explained by the sources.
        Only returned if return_residual is True.

    See Also
    --------
    mixed_norm
    """
    _check_reference(evoked)

    all_ch_names = evoked.ch_names
    info = evoked.info

    if (alpha_space < 0.) or (alpha_space > 100.):
        raise Exception('alpha_space must be in range [0, 100].'
                        ' Got alpha_space = %f' % alpha_space)

    if (alpha_time < 0.) or (alpha_time > 100.):
        raise Exception('alpha_time must be in range [0, 100].'
                        ' Got alpha_time = %f' % alpha_time)

    # put the forward solution in fixed orientation if it's not already
    if loose is None and not is_fixed_orient(forward):
        forward = deepcopy(forward)
        _to_fixed_ori(forward)

    n_dip_per_pos = 1 if is_fixed_orient(forward) else 3

    gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked.info, noise_cov, pca, depth, loose, weights,
        weights_min)

    if window is not None:
        evoked = _window_evoked(evoked, window)

    sel = [all_ch_names.index(name) for name in gain_info["ch_names"]]
    M = evoked.data[sel]

    # Whiten data
    logger.info('Whitening data matrix.')
    M = np.dot(whitener, M)

    # Scaling to make setting of alpha easy
    alpha_max = norm_l2inf(np.dot(gain.T, M), n_dip_per_pos, copy=False)
    alpha_max *= 0.01
    gain /= alpha_max
    source_weighting /= alpha_max

    X, active_set, E = tf_mixed_norm_solver(
        M, gain, alpha_space, alpha_time, wsize=wsize, tstep=tstep,
        maxit=maxit, tol=tol, verbose=verbose, n_orient=n_dip_per_pos,
        log_objective=False, debias=debias)

    if active_set.sum() == 0:
        raise Exception("No active dipoles found. "
                        "alpha_space/alpha_time are too big.")

    if mask is not None:
        active_set_tmp = np.zeros(len(mask), dtype=np.bool)
        active_set_tmp[mask] = active_set
        active_set = active_set_tmp
        del active_set_tmp

    X = _reapply_source_weighting(
        X, source_weighting, active_set, n_dip_per_pos)

    if return_residual:
        residual = _compute_residual(
            forward, evoked, X, active_set, gain_info)

    stc = _make_sparse_stc(
        X, active_set, forward, evoked.times[0], 1.0 / info['sfreq'])

    logger.info('[done]')

    if return_residual:
        out = stc, residual
    else:
        out = stc

    return out
