# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Martin Luessi <gramfort@nmr.mgh.harvard.edu>
# License: Simplified BSD
from copy import deepcopy

import numpy as np
from scipy import linalg

import logging
logger = logging.getLogger('mne')

from ..forward import is_fixed_orient, _to_fixed_ori
from ..mixed_norm.inverse import _make_sparse_stc, _prepare_gain
from ..minimum_norm.inverse import _prepare_forward
from .. import verbose


@verbose
def _gamma_map_opt(M, G, alpha, maxit=1000, tol=1e-6, update_mode=1,
                   group_size=1, gammas=None, verbose=None):
    """Hierarchical Bayes (Gamma-MAP)

    Parameters
    ----------
    M : array, /shape=(n_sensors, n_times)
        Observation.
    G : array, shape=(n_sensors, n_sources)
        Forward operator.
    alpha : float
        Regularization parameter (noise variance).
    maxit : int
        Maximum number of iterations.
    tol : float
        Tolerance parameter for convergence.
    group_size : int
        Number of consecutive sources which use the same gamma.
    gammas : array, shape=(n_sources,)
        Initial values for posterior variances (gammas). If None, a
        variance of 1.0 is used.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    X : array, shape=(n_active, n_times)
        Estimated source time courses.
    active_set : array, shape=(n_active,)
        Indices of in G of signals in X.

    References
    ----------
    Wipf et al. Analysis of Empirical Bayesian Methods for Neuroelectromagnetic
    Source Localization. Advances in Neural Information Processing Systems (2007)
    """
    G = G.copy()
    M = M.copy()

    if gammas is None:
        gammas = np.ones(G.shape[1], dtype=np.float)

    eps = np.finfo(float).eps

    n_sensors, n_times = M.shape
    Minit = M.copy()

    MMt = np.dot(M, M.T)
    normalize_constant = linalg.norm(MMt, ord='fro')
    M /= np.sqrt(normalize_constant)
    Minit /= np.sqrt(normalize_constant)
    MMt /= normalize_constant
    alpha /= normalize_constant

    G_normalize_constant = linalg.norm(G, ord=np.inf)
    G /= G_normalize_constant

    n_sources = G.shape[1]

    if n_sources % group_size != 0:
        raise ValueError('Number of sources has to be evenly dividable by the '
                         'group size')

    n_active = n_sources
    active_set = np.arange(n_sources)

    counter_n_active_fixed = 0
    gammas_full_old = gammas.copy()

    for itno in np.arange(maxit):
        counter_n_active_fixed += 1

        gammas[np.isnan(gammas)] = 0.0

        gidx = (np.abs(gammas) > eps)
        active_set = active_set[gidx]
        gammas = gammas[gidx]

        # update only active gammas (once set to zero it stays at zero)
        if n_active > len(active_set):
            n_active = active_set.size
            G = G[:, gidx]
            counter_n_active_fixed = 0

        CM = alpha * np.eye(n_sensors) + np.dot(G * gammas[np.newaxis, :], G.T)
        # Invert CM keeping symmetry
        U, S, V = linalg.svd(CM, full_matrices=False)
        S = S[np.newaxis, :]
        CM = np.dot(U * S, U.T)
        CMinv = np.dot(U / (S + eps), U.T)

        CMinvG = np.dot(CMinv, G)
        A = np.dot(CMinvG.T, M)  # mult. w. Diag(gamma) in gamma update

        if update_mode == 1:
            # M-SBL update
            numer = (gammas ** 2 * np.mean(np.abs(A) ** 2, axis=1)
                     + gammas * (1 - gammas * np.sum(G * CMinvG, axis=0)))
            denom = None
        elif update_mode == 2:
            # MacKay fixed point update (equivalent to Variational-Bayes Sato
            # update in hbi_inverse.m)
            numer = gammas ** 2 * np.mean(np.abs(A) ** 2, axis=1)
            denom = gammas * np.sum(G * CMinvG, axis=0)
        elif update_mode == 3:
            # modified MacKay fixed point update
            1 / 0  # XXX fix this
            gammas *= (np.sqrt(np.mean(np.abs(A) ** 2, axis=1)
                       / np.sum(G * CMinvG, axis=0)))

        if group_size == 1:
            if denom is None:
                gammas = numer
            else:
                gammas = numer / denom
        else:
            numer_comb = np.mean(numer.reshape(-1, group_size), axis=1)
            if denom is None:
                gammas_comb = numer_comb
            else:
                denom_comb = np.mean(denom.reshape(-1, group_size), axis=1)
                gammas_comb = numer_comb / denom_comb

            gammas = np.repeat(gammas_comb, group_size)

        # compute convergence criterion
        gammas_full = np.zeros(n_sources, dtype=np.float)
        gammas_full[active_set] = gammas

        err = (np.sum(np.abs(gammas_full - gammas_full_old))
               / np.sum(np.abs(gammas_full_old)))

        gammas_full_old = gammas_full

        logger.info('Iteration: %d\t active set size: %d\t convergence: %0.3e'
                    % (itno, len(gammas), err))

        if err < tol:
            break

        if n_active == 0:
            break

        #import pylab as pl
        #pl.figure(2)
        #pl.clf()
        #pl.plot(gammas)
        ##pl.ylim([0, 1])
        #pl.show()
        #print 'min max gamma: %e %e' % (np.min(gammas), np.max(gammas))
        #print gammas.dtype

    if itno < maxit - 1:
        print('\nConvergence reached !\n')
    else:
        print('\nConvergence NOT reached !\n')

    n_const = np.sqrt(normalize_constant) / G_normalize_constant
    x_active = n_const * gammas[:, None] * A

    return x_active, active_set


@verbose
def gamma_map_inverse(evoked, forward, noise_cov, alpha, loose=0.2, depth=0.8,
                      xyz_same_gamma=True, maxit=1000, tol=1e-6, update_mode=1,
                      gammas=None, pca=True, verbose=None):
    """Hierarchical Bayes (Gamma-MAP)

    Parameters
    ----------
    evoked : instance of Evoked
        Evoked data to invert.
    forward : dict
        Forward operator
    noise_cov : instance of Covariance
        Noise covariance to compute whitener.
    alpha : float
        Regularization parameter (noise variance).
    loose : float in [0, 1]
        Value that weights the source variances of the dipole components
        that are parallel (tangential) to the cortical surface. If loose
        is 0 or None then the solution is computed with fixed orientation.
        If loose is 1, it corresponds to free orientations.
    depth: None | float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed.
    xyz_same_gamma : bool
        Use same gamma for xyz current components at each source space point.
        Recommended for free-orientation forward solutions.
    maxit : int
        Maximum number of iterations.
    tol : float
        Tolerance parameter for convergence.
    gammas : array, shape=(n_sources,)
        Initial values for posterior variances (gammas). If None, a
        variance of 1.0 is used.
    pca : bool
        If True the rank of the data is reduced to true dimension.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : instance of SourceEstimate
        Source time courses.

    References
    ----------
    Wipf et al. Analysis of Empirical Bayesian Methods for Neuroelectromagnetic
    Source Localization. Advances in Neural Information Processing Systems (2007)
    """
    # make forward solution in fixed orientation if necessary
    if loose is None and not is_fixed_orient(forward):
        forward = deepcopy(forward)
        _to_fixed_ori(forward)

    if is_fixed_orient(forward) or not xyz_same_gamma:
        group_size = 1
    else:
        group_size = 3

    gain_info, gain, _, whitener, _ = _prepare_forward(forward, evoked.info,
                                                       noise_cov, pca)

    # get the data
    sel = [evoked.ch_names.index(name) for name in gain_info['ch_names']]
    M = evoked.data[sel]

    # whiten and prepare gain matrix
    gain, source_weighting, mask = _prepare_gain(gain, forward, whitener,
                                                 depth, loose, None,
                                                 None)
    # whiten the data
    M = np.dot(whitener, M)

    # run the optimization
    X, active_set = _gamma_map_opt(M, gain, alpha, maxit=maxit, tol=tol,
                                   update_mode=update_mode, gammas=gammas,
                                   group_size=group_size, verbose=verbose)

    # reapply weights to have correct unit
    X /= source_weighting[active_set][:, None]

    if group_size == 1 and not is_fixed_orient(forward):
        # make sure each source has 3 components
        active_src = np.unique(active_set // 3)
        in_pos = 0
        if len(X) < 3 * len(active_src):
            X_xyz = np.zeros((3 * len(active_src), X.shape[1]), dtype=X.dtype)
            for ii in xrange(len(active_src)):
                for jj in xrange(3):
                    if in_pos >= len(active_set):
                        break
                    if (active_set[in_pos] + jj) % 3 == 0:
                        X_xyz[3 * ii + jj] = X[in_pos]
                        in_pos += 1
            X = X_xyz

    tmin = evoked.times[0]
    tstep = 1.0 / evoked.info['sfreq']
    stc = _make_sparse_stc(X, active_set, forward, tmin, tstep,
                           active_is_idx=True, verbose=verbose)

    return stc
