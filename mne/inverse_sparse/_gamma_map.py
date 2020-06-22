# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
# License: Simplified BSD

import numpy as np
from scipy import linalg

from ..forward import is_fixed_orient

from ..minimum_norm.inverse import _check_reference
from ..utils import logger, verbose, warn
from .mxne_inverse import (_check_ori, _make_sparse_stc, _prepare_gain,
                           _reapply_source_weighting, _compute_residual,
                           _make_dipoles_sparse)


@verbose
def _gamma_map_opt(M, G, alpha, maxit=10000, tol=1e-6, update_mode=1,
                   group_size=1, gammas=None, verbose=None):
    """Hierarchical Bayes (Gamma-MAP).

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
    %(verbose)s

    Returns
    -------
    X : array, shape=(n_active, n_times)
        Estimated source time courses.
    active_set : array, shape=(n_active,)
        Indices of active sources.
    """
    G = G.copy()
    M = M.copy()

    if gammas is None:
        gammas = np.ones(G.shape[1], dtype=np.float64)

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
        def denom_fun(x):
            return x

    last_size = -1
    for itno in range(maxit):
        gammas[np.isnan(gammas)] = 0.0

        gidx = (np.abs(gammas) > eps)
        active_set = active_set[gidx]
        gammas = gammas[gidx]

        # update only active gammas (once set to zero it stays at zero)
        if n_active > len(active_set):
            n_active = active_set.size
            G = G[:, gidx]

        CM = np.dot(G * gammas[np.newaxis, :], G.T)
        CM.flat[::n_sensors + 1] += alpha
        # Invert CM keeping symmetry
        U, S, V = linalg.svd(CM, full_matrices=False)
        S = S[np.newaxis, :]
        del CM
        CMinv = np.dot(U / (S + eps), U.T)
        CMinvG = np.dot(CMinv, G)
        A = np.dot(CMinvG.T, M)  # mult. w. Diag(gamma) in gamma update

        if update_mode == 1:
            # MacKay fixed point update (10) in [1]
            numer = gammas ** 2 * np.mean((A * A.conj()).real, axis=1)
            denom = gammas * np.sum(G * CMinvG, axis=0)
        elif update_mode == 2:
            # modified MacKay fixed point update (11) in [1]
            numer = gammas * np.sqrt(np.mean((A * A.conj()).real, axis=1))
            denom = np.sum(G * CMinvG, axis=0)  # sqrt is applied below
        else:
            raise ValueError('Invalid value for update_mode')

        if group_size == 1:
            if denom is None:
                gammas = numer
            else:
                gammas = numer / np.maximum(denom_fun(denom),
                                            np.finfo('float').eps)
        else:
            numer_comb = np.sum(numer.reshape(-1, group_size), axis=1)
            if denom is None:
                gammas_comb = numer_comb
            else:
                denom_comb = np.sum(denom.reshape(-1, group_size), axis=1)
                gammas_comb = numer_comb / denom_fun(denom_comb)

            gammas = np.repeat(gammas_comb / group_size, group_size)

        # compute convergence criterion
        gammas_full = np.zeros(n_sources, dtype=np.float64)
        gammas_full[active_set] = gammas

        err = (np.sum(np.abs(gammas_full - gammas_full_old)) /
               np.sum(np.abs(gammas_full_old)))

        gammas_full_old = gammas_full

        breaking = (err < tol or n_active == 0)
        if len(gammas) != last_size or breaking:
            logger.info('Iteration: %d\t active set size: %d\t convergence: '
                        '%0.3e' % (itno, len(gammas), err))
            last_size = len(gammas)

        if breaking:
            break

    if itno < maxit - 1:
        logger.info('\nConvergence reached !\n')
    else:
        warn('\nConvergence NOT reached !\n')

    # undo normalization and compute final posterior mean
    n_const = np.sqrt(M_normalize_constant) / G_normalize_constant
    x_active = n_const * gammas[:, None] * A

    return x_active, active_set


@verbose
def gamma_map(evoked, forward, noise_cov, alpha, loose="auto", depth=0.8,
              xyz_same_gamma=True, maxit=10000, tol=1e-6, update_mode=1,
              gammas=None, pca=True, return_residual=False,
              return_as_dipoles=False, rank=None, pick_ori=None, verbose=None):
    """Hierarchical Bayes (Gamma-MAP) sparse source localization method.

    Models each source time course using a zero-mean Gaussian prior with an
    unknown variance (gamma) parameter. During estimation, most gammas are
    driven to zero, resulting in a sparse source estimate, as in
    [1]_ and [2]_.

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
    loose : float in [0, 1] | 'auto'
        Value that weights the source variances of the dipole components
        that are parallel (tangential) to the cortical surface. If loose
        is 0 then the solution is computed with fixed orientation.
        If loose is 1, it corresponds to free orientations.
        The default value ('auto') is set to 0.2 for surface-oriented source
        space and set to 1.0 for volumic or discrete source space.
    %(depth)s
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
    return_as_dipoles : bool
        If True, the sources are returned as a list of Dipole instances.
    %(rank_None)s

        .. versionadded:: 0.18
    %(pick_ori)s
    %(verbose)s

    Returns
    -------
    stc : instance of SourceEstimate
        Source time courses.
    residual : instance of Evoked
        The residual a.k.a. data not explained by the sources.
        Only returned if return_residual is True.

    References
    ----------
    .. [1] Wipf et al. Analysis of Empirical Bayesian Methods for
           Neuroelectromagnetic Source Localization, Advances in Neural
           Information Process. Systems (2007)

    .. [2] D. Wipf, S. Nagarajan
           "A unified Bayesian framework for MEG/EEG source imaging",
           Neuroimage, Volume 44, Number 3, pp. 947-966, Feb. 2009.
           DOI: 10.1016/j.neuroimage.2008.02.059
    """
    _check_reference(evoked)

    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked.info, noise_cov, pca, depth, loose, rank)
    _check_ori(pick_ori, forward)

    group_size = 1 if (is_fixed_orient(forward) or not xyz_same_gamma) else 3

    # get the data
    sel = [evoked.ch_names.index(name) for name in gain_info['ch_names']]
    M = evoked.data[sel]

    # whiten the data
    logger.info('Whitening data matrix.')
    M = np.dot(whitener, M)

    # run the optimization
    X, active_set = _gamma_map_opt(M, gain, alpha, maxit=maxit, tol=tol,
                                   update_mode=update_mode, gammas=gammas,
                                   group_size=group_size, verbose=verbose)

    if len(active_set) == 0:
        raise Exception("No active dipoles found. alpha is too big.")

    # Compute estimated whitened sensor data
    M_estimated = np.dot(gain[:, active_set], X)

    # Reapply weights to have correct unit
    X = _reapply_source_weighting(X, source_weighting, active_set)

    if return_residual:
        residual = _compute_residual(forward, evoked, X, active_set,
                                     gain_info)

    if group_size == 1 and not is_fixed_orient(forward):
        # make sure each source has 3 components
        active_src = np.unique(active_set // 3)
        in_pos = 0
        if len(X) < 3 * len(active_src):
            X_xyz = np.zeros((3 * len(active_src), X.shape[1]), dtype=X.dtype)
            for ii in range(len(active_src)):
                for jj in range(3):
                    if in_pos >= len(active_set):
                        break
                    if (active_set[in_pos] + jj) % 3 == 0:
                        X_xyz[3 * ii + jj] = X[in_pos]
                        in_pos += 1
            X = X_xyz

    tmin = evoked.times[0]
    tstep = 1.0 / evoked.info['sfreq']

    if return_as_dipoles:
        out = _make_dipoles_sparse(X, active_set, forward, tmin, tstep, M,
                                   M_estimated, active_is_idx=True)
    else:
        out = _make_sparse_stc(X, active_set, forward, tmin, tstep,
                               active_is_idx=True, pick_ori=pick_ori,
                               verbose=verbose)

    logger.info('[done]')

    if return_residual:
        out = out, residual

    return out
