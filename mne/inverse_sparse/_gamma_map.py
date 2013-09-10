# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Martin Luessi <gramfort@nmr.mgh.harvard.edu>
# License: Simplified BSD
from copy import deepcopy

import numpy as np
from scipy import linalg

from ..forward import is_fixed_orient, _to_fixed_ori
from ..fiff.pick import pick_channels_evoked
from ..minimum_norm.inverse import _prepare_forward
from ..utils import logger, verbose
from .mxne_inverse import _make_sparse_stc, _prepare_gain


@verbose
def _gamma_map_opt(M, G, alpha, maxit=10000, tol=1e-6, update_mode=1,
                   group_size=1, gammas=None, verbose=None):
    """Hierarchical Bayes (Gamma-MAP)

    Parameters
    ----------
    M : array, shape=(n_sensors, n_times)
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
    update_mode : int
        Update mode, 1: MacKay update (default), 3: Modified MacKay update.
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
        Indices of active sources.

    References
    ----------
    [1] Wipf et al. Analysis of Empirical Bayesian Methods for
    Neuroelectromagnetic Source Localization, Advances in Neural Information
    Processing Systems (2007).
    """
    G = G.copy()
    M = M.copy()

    if gammas is None:
        gammas = np.ones(G.shape[1], dtype=np.float)

    eps = np.finfo(float).eps

    n_sources = G.shape[1]
    n_sensors, n_times = M.shape

    # apply normalization so the numerical values are sane
    M_normalize_constant = linalg.norm(np.dot(M, M.T), ord='fro')
    M /= np.sqrt(M_normalize_constant)
    alpha /= M_normalize_constant
    G_normalize_constant = linalg.norm(G, ord=np.inf)
    G /= G_normalize_constant

    if n_sources % group_size != 0:
        raise ValueError('Number of sources has to be evenly dividable by the '
                         'group size')

    n_active = n_sources
    active_set = np.arange(n_sources)

    gammas_full_old = gammas.copy()

    if update_mode == 2:
        denom_fun = np.sqrt
    else:
        # do nothing
        denom_fun = lambda x: x

    for itno in np.arange(maxit):
        gammas[np.isnan(gammas)] = 0.0

        gidx = (np.abs(gammas) > eps)
        active_set = active_set[gidx]
        gammas = gammas[gidx]

        # update only active gammas (once set to zero it stays at zero)
        if n_active > len(active_set):
            n_active = active_set.size
            G = G[:, gidx]

        CM = alpha * np.eye(n_sensors) + np.dot(G * gammas[np.newaxis, :], G.T)
        # Invert CM keeping symmetry
        U, S, V = linalg.svd(CM, full_matrices=False)
        S = S[np.newaxis, :]
        CM = np.dot(U * S, U.T)
        CMinv = np.dot(U / (S + eps), U.T)

        CMinvG = np.dot(CMinv, G)
        A = np.dot(CMinvG.T, M)  # mult. w. Diag(gamma) in gamma update

        if update_mode == 1:
            # MacKay fixed point update (10) in [1]
            numer = gammas ** 2 * np.mean(np.abs(A) ** 2, axis=1)
            denom = gammas * np.sum(G * CMinvG, axis=0)
        elif update_mode == 2:
            # modified MacKay fixed point update (11) in [1]
            numer = gammas * np.sqrt(np.mean(np.abs(A) ** 2, axis=1))
            denom = np.sum(G * CMinvG, axis=0)  # sqrt is applied below
        else:
            raise ValueError('Invalid value for update_mode')

        if group_size == 1:
            if denom is None:
                gammas = numer
            else:
                gammas = numer / denom_fun(denom)
        else:
            numer_comb = np.sum(numer.reshape(-1, group_size), axis=1)
            if denom is None:
                gammas_comb = numer_comb
            else:
                denom_comb = np.sum(denom.reshape(-1, group_size), axis=1)
                gammas_comb = numer_comb / denom_fun(denom_comb)

            gammas = np.repeat(gammas_comb / group_size, group_size)

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

    if itno < maxit - 1:
        print('\nConvergence reached !\n')
    else:
        print('\nConvergence NOT reached !\n')

    # undo normalization and compute final posterior mean
    n_const = np.sqrt(M_normalize_constant) / G_normalize_constant
    x_active = n_const * gammas[:, None] * A

    return x_active, active_set


@verbose
def gamma_map(evoked, forward, noise_cov, alpha, loose=0.2, depth=0.8,
              xyz_same_gamma=True, maxit=10000, tol=1e-6, update_mode=1,
              gammas=None, pca=True, return_residual=False, verbose=None):
    """Hierarchical Bayes (Gamma-MAP) sparse source localization method

    Models each source time course using a zero-mean Gaussian prior with an
    unknown variance (gamma) parameter. During estimation, most gammas are
    driven to zero, resulting in a sparse source estimate.

    For fixed-orientation forward operators, a separate gamma is used for each
    source time course, while for free-orientation forward operators, the same
    gamma is used for the three source time courses at each source space point
    (separate gammas can be used in this case by using xyz_same_gamma=False).

    Parameters
    ----------
    evoked : instance of Evoked
        Evoked data to invert.
    forward : dict
        Forward operator.
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
    update_mode : int
        Update mode, 1: MacKay update (default), 2: Modified MacKay update.
    gammas : array, shape=(n_sources,)
        Initial values for posterior variances (gammas). If None, a
        variance of 1.0 is used.
    pca : bool
        If True the rank of the data is reduced to the true dimension.
    return_residual : bool
        If True, the residual is returned as an Evoked instance.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : instance of SourceEstimate
        Source time courses.
    residual : instance of Evoked
        The residual a.k.a. data not explained by the sources.
        Only returned if return_residual is True.

    References
    ----------
    Wipf et al. Analysis of Empirical Bayesian Methods for Neuroelectromagnetic
    Source Localization, Advances in Neural Information Process. Systems (2007)

    Wipf et al. A unified Bayesian framework for MEG/EEG source imaging,
    NeuroImage, vol. 44, no. 3, pp. 947-66, Mar. 2009.
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

    if len(active_set) == 0:
        raise Exception("No active dipoles found. alpha is too big.")

    # reapply weights to have correct unit
    X /= source_weighting[active_set][:, None]

    if return_residual:
        sel = [forward['sol']['row_names'].index(c)
               for c in gain_info['ch_names']]
        residual = evoked.copy()
        residual = pick_channels_evoked(residual,
                                        include=gain_info['ch_names'])
        residual.data -= np.dot(forward['sol']['data'][sel, :][:, active_set],
                                X)

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

    if return_residual:
        return stc, residual
    else:
        return stc
