# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

from copy import deepcopy
import numpy as np
from scipy import linalg

from ..source_estimate import SourceEstimate
from ..minimum_norm.inverse import combine_xyz, _make_stc, _prepare_forward
from ..forward import compute_orient_prior, is_fixed_orient
from ..fiff.pick import pick_channels_evoked
from .optim import mixed_norm_solver, norm_l2inf


def _prepare_gain(gain, forward, whitener, depth, loose, weights, weights_min):
    print 'Whitening lead field matrix.'
    gain = np.dot(whitener, gain)

    # Handle depth prior scaling
    source_weighting = np.sum(gain ** 2, axis=0) ** depth

    # apply loose orientations
    orient_prior = compute_orient_prior(forward, loose)

    source_weighting /= orient_prior
    source_weighting = np.sqrt(source_weighting)
    gain /= source_weighting[None, :]

    # Handle weights
    mask = None
    if weights is not None:
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
        source_weighting /= weights
        gain *= weights[None, :]

        if weights_min is not None:
            mask = (weights > weights_min)
            gain = gain[:, mask]
            n_sources = np.sum(mask) / n_dip_per_pos
            print "Reducing source space to %d sources" % n_sources

    return gain, source_weighting, mask


def _make_sparse_stc(X, active_set, forward, tmin, tstep):
    if not is_fixed_orient(forward):
        print 'combining the current components...',
        X = combine_xyz(X)

    active_idx = np.where(active_set)[0]
    n_dip_per_pos = 1 if is_fixed_orient(forward) else 3
    if n_dip_per_pos > 1:
        active_idx = np.unique(active_idx // n_dip_per_pos)

    src = forward['src']

    n_lh_points = len(src[0]['vertno'])
    lh_vertno = src[0]['vertno'][active_idx[active_idx < n_lh_points]]
    rh_vertno = src[1]['vertno'][active_idx[active_idx >= n_lh_points]
                                             - n_lh_points]
    stc = _make_stc(X, tmin, tstep, [lh_vertno, rh_vertno])
    return stc


def mixed_norm(evoked, forward, noise_cov, alpha, loose=0.2, depth=0.8,
               maxit=3000, tol=1e-4, active_set_size=10, pca=True,
               debias=True, time_pca=True, weights=None, weights_min=None,
               return_residual=False):
    """Mixed-norm estimate (MxNE)

    Compute L1/L2 mixed-norm solution on evoked data.

    Reference:
    Gramfort A., Kowalski M. and Hamalainen, M,
    Mixed-norm estimates for the M/EEG inverse problem using accelerated
    gradient methods, Physics in Medicine and Biology, 2012
    http://dx.doi.org/10.1088/0031-9155/57/7/1937

    Parameters
    ----------
    evoked : instance of Evoked or list of instances of Evoked
        Evoked data to invert
    forward : dict
        Forward operator
    noise_cov : instance of Covariance
        Noise covariance to compute whitener
    alpha : float
        Regularization parameter
    loose : float in [0, 1]
        Value that weights the source variances of the dipole components
        defining the tangent space of the cortical surfaces. If loose
        is 0 or None then the solution is computed with fixed orientation.
    maxit : int
        Maximum number of iterations
    tol : float
        Tolerance parameter
    active_set_size : int | None
        Size of active set increment. If None, no active set strategy is used.
    pca: bool
        If True the rank of the data is reduced to true dimension.
    debias: bool
        Remove coefficient amplitude bias due to L1 penalty
    time_pca: bool or int
        If True the rank of the concatenated epochs is reduced to
        its true dimension. If is 'int' the rank is limited to this value.
    weights: None | array | SourceEstimate
        Weight for penalty in mixed_norm. Can be None or
        1d array of length n_sources or a SourceEstimate e.g. obtained
        with wMNE or dSPM or fMRI
    weights_min: float
        Do not consider in the estimation sources for which weights
        is less than weights_min.
    return_residual: bool
        If True, the residual is returned as an Evoked instance.

    Returns
    -------
    stc : SourceEstimate | list of SourceEstimate
        Source time courses for each evoked data passed as input.
    residual : instance of Evoked
        The residual a.k.a. data not explained by the sources.
        Only returned if return_residual is True.
    """
    if not isinstance(evoked, list):
        evoked = [evoked]

    all_ch_names = evoked[0].ch_names
    if not all(all_ch_names == evoked[i].ch_names
                                            for i in range(1, len(evoked))):
        raise Exception('All the datasets must have the same good channels.')

    info = evoked[0].info
    ch_names, gain, _, whitener, _ = _prepare_forward(forward,
                                                      info, noise_cov, pca)

    # Whiten lead field.
    gain, source_weighting, mask = _prepare_gain(gain, forward, whitener,
                                            depth, loose, weights, weights_min)

    sel = [all_ch_names.index(name) for name in ch_names]
    M = np.concatenate([e.data[sel] for e in evoked], axis=1)

    # Whiten data
    print 'Whitening data matrix.'
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
    source_weighting *= alpha_max

    X, active_set, E = mixed_norm_solver(M, gain, alpha,
                                         maxit=maxit, tol=tol, verbose=True,
                                         active_set_size=active_set_size,
                                         debias=debias,
                                         n_orient=n_dip_per_pos)

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
    X /= source_weighting[active_set][:, None]

    stcs = list()
    residual = list()
    cnt = 0
    for e in evoked:
        tmin = float(e.first) / e.info['sfreq']
        tstep = 1.0 / e.info['sfreq']
        stc = _make_sparse_stc(X[:, cnt:(cnt + len(e.times))], active_set,
                               forward, tmin, tstep)
        stcs.append(stc)

        if return_residual:
            sel = [forward['sol']['row_names'].index(c) for c in ch_names]
            r = deepcopy(e)
            r = pick_channels_evoked(r, include=ch_names)
            r.data -= np.dot(forward['sol']['data'][sel, :][:, active_set], X)
            residual.append(r)

    print '[done]'

    if len(stcs) == 1:
        out = stcs[0]
        if return_residual:
            residual = residual[0]
    else:
        out = stcs

    if return_residual:
        out = out, residual

    return out
