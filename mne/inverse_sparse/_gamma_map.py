# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ..fixes import _safe_svd
from ..forward import is_fixed_orient
from ..minimum_norm.inverse import _check_reference, _log_exp_var
from ..utils import logger, verbose, warn
from .mxne_inverse import (
    _check_ori,
    _compute_residual,
    _make_dipoles_sparse,
    _make_sparse_stc,
    _prepare_gain,
    _reapply_source_weighting,
)


@verbose
def _gamma_map_opt(
    M,
    G,
    alpha,
    maxit=10000,
    tol=1e-6,
    update_mode=1,
    group_size=1,
    gammas=None,
    verbose=None,
):
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
    M_normalize_constant = np.linalg.norm(np.dot(M, M.T), ord="fro")
    M /= np.sqrt(M_normalize_constant)
    alpha /= M_normalize_constant
    G_normalize_constant = np.linalg.norm(G, ord=np.inf)
    G /= G_normalize_constant

    if n_sources % group_size != 0:
        raise ValueError(
            "Number of sources has to be evenly dividable by the group size"
        )

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

        gidx = np.abs(gammas) > eps
        active_set = active_set[gidx]
        gammas = gammas[gidx]

        # update only active gammas (once set to zero it stays at zero)
        if n_active > len(active_set):
            n_active = active_set.size
            G = G[:, gidx]

        CM = np.dot(G * gammas[np.newaxis, :], G.T)
        CM.flat[:: n_sensors + 1] += alpha
        # Invert CM keeping symmetry
        U, S, _ = _safe_svd(CM, full_matrices=False)
        S = S[np.newaxis, :]
        del CM
        CMinv = np.dot(U / (S + eps), U.T)
        CMinvG = np.dot(CMinv, G)
        A = np.dot(CMinvG.T, M)  # mult. w. Diag(gamma) in gamma update

        if update_mode == 1:
            # MacKay fixed point update (10) in [1]
            numer = gammas**2 * np.mean((A * A.conj()).real, axis=1)
            denom = gammas * np.sum(G * CMinvG, axis=0)
        elif update_mode == 2:
            # modified MacKay fixed point update (11) in [1]
            numer = gammas * np.sqrt(np.mean((A * A.conj()).real, axis=1))
            denom = np.sum(G * CMinvG, axis=0)  # sqrt is applied below
        else:
            raise ValueError("Invalid value for update_mode")

        if group_size == 1:
            if denom is None:
                gammas = numer
            else:
                gammas = numer / np.maximum(denom_fun(denom), np.finfo("float").eps)
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

        err = np.sum(np.abs(gammas_full - gammas_full_old)) / np.sum(
            np.abs(gammas_full_old)
        )

        gammas_full_old = gammas_full

        breaking = err < tol or n_active == 0
        if len(gammas) != last_size or breaking:
            logger.info(
                f"Iteration: {itno}\t active set size: {len(gammas)}\t convergence: "
                f"{err:.3e}"
            )
            last_size = len(gammas)

        if breaking:
            break

    if itno < maxit - 1:
        logger.info("\nConvergence reached !\n")
    else:
        warn("\nConvergence NOT reached !\n")

    # undo normalization and compute final posterior mean
    n_const = np.sqrt(M_normalize_constant) / G_normalize_constant
    x_active = n_const * gammas[:, None] * A

    return x_active, active_set


@verbose
def gamma_map(
    evoked,
    forward,
    noise_cov,
    alpha,
    loose="auto",
    depth=0.8,
    xyz_same_gamma=True,
    maxit=10000,
    tol=1e-6,
    update_mode=1,
    gammas=None,
    pca=True,
    return_residual=False,
    return_as_dipoles=False,
    rank=None,
    pick_ori=None,
    verbose=None,
):
    """Hierarchical Bayes (Gamma-MAP) sparse source localization method.

    Models each source time course using a zero-mean Gaussian prior with an
    unknown variance (gamma) parameter. During estimation, most gammas are
    driven to zero, resulting in a sparse source estimate, as in
    :footcite:`WipfEtAl2007` and :footcite:`WipfNagarajan2009`.

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
    %(loose)s
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
    %(rank_none)s

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
    .. footbibliography::
    """
    _check_reference(evoked)

    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked.info, noise_cov, pca, depth, loose, rank
    )
    _check_ori(pick_ori, forward)

    group_size = 1 if (is_fixed_orient(forward) or not xyz_same_gamma) else 3

    # get the data
    sel = [evoked.ch_names.index(name) for name in gain_info["ch_names"]]
    M = evoked.data[sel]

    # whiten the data
    logger.info("Whitening data matrix.")
    M = np.dot(whitener, M)

    # run the optimization
    X, active_set = _gamma_map_opt(
        M,
        gain,
        alpha,
        maxit=maxit,
        tol=tol,
        update_mode=update_mode,
        gammas=gammas,
        group_size=group_size,
        verbose=verbose,
    )

    if len(active_set) == 0:
        raise Exception("No active dipoles found. alpha is too big.")

    M_estimate = gain[:, active_set] @ X

    # Reapply weights to have correct unit
    X = _reapply_source_weighting(X, source_weighting, active_set)

    if return_residual:
        residual = _compute_residual(forward, evoked, X, active_set, gain_info)

    if group_size == 1 and not is_fixed_orient(forward):
        # make sure each source has 3 components
        idx, offset = divmod(active_set, 3)
        active_src = np.unique(idx)
        if len(X) < 3 * len(active_src):
            X_xyz = np.zeros((len(active_src), 3, X.shape[1]), dtype=X.dtype)
            idx = np.searchsorted(active_src, idx)
            X_xyz[idx, offset, :] = X
            X_xyz.shape = (len(active_src) * 3, X.shape[1])
            X = X_xyz
        active_set = (active_src[:, np.newaxis] * 3 + np.arange(3)).ravel()
    source_weighting[source_weighting == 0] = 1  # zeros
    gain_active = gain[:, active_set] / source_weighting[active_set]
    del source_weighting

    tmin = evoked.times[0]
    tstep = 1.0 / evoked.info["sfreq"]

    if return_as_dipoles:
        out = _make_dipoles_sparse(
            X, active_set, forward, tmin, tstep, M, gain_active, active_is_idx=True
        )
    else:
        out = _make_sparse_stc(
            X,
            active_set,
            forward,
            tmin,
            tstep,
            active_is_idx=True,
            pick_ori=pick_ori,
            verbose=verbose,
        )

    _log_exp_var(M, M_estimate, prefix="")
    logger.info("[done]")

    if return_residual:
        out = out, residual

    return out
