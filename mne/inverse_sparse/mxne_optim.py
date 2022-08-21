# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Daniel Strohmeier <daniel.strohmeier@gmail.com>
#         Mathurin Massias <mathurin.massias@gmail.com>
# License: Simplified BSD

import functools
from math import sqrt

import numpy as np

from .mxne_debiasing import compute_bias
from ..utils import (logger, verbose, sum_squared, warn, _get_blas_funcs,
                     _validate_type, _check_option)
from ..time_frequency._stft import stft_norm1, stft_norm2, stft, istft


@functools.lru_cache(None)
def _get_dgemm():
    return _get_blas_funcs(np.float64, 'gemm')


def groups_norm2(A, n_orient):
    """Compute squared L2 norms of groups inplace."""
    n_positions = A.shape[0] // n_orient
    return np.sum(np.power(A, 2, A).reshape(n_positions, -1), axis=1)


def norm_l2inf(A, n_orient, copy=True):
    """L2-inf norm."""
    if A.size == 0:
        return 0.0
    if copy:
        A = A.copy()
    return sqrt(np.max(groups_norm2(A, n_orient)))


def norm_l21(A, n_orient, copy=True):
    """L21 norm."""
    if A.size == 0:
        return 0.0
    if copy:
        A = A.copy()
    return np.sum(np.sqrt(groups_norm2(A, n_orient)))


def _primal_l21(M, G, X, active_set, alpha, n_orient):
    """Primal objective for the mixed-norm inverse problem.

    See :footcite:`GramfortEtAl2012`.

    Parameters
    ----------
    M : array, shape (n_sensors, n_times)
        The data.
    G : array, shape (n_sensors, n_active)
        The gain matrix a.k.a. lead field.
    X : array, shape (n_active, n_times)
        Sources.
    active_set : array of bool, shape (n_sources,)
        Mask of active sources.
    alpha : float
        The regularization parameter.
    n_orient : int
        Number of dipoles per locations (typically 1 or 3).

    Returns
    -------
    p_obj : float
        Primal objective.
    R : array, shape (n_sensors, n_times)
        Current residual (M - G * X).
    nR2 : float
        Data-fitting term.
    GX : array, shape (n_sensors, n_times)
        Forward prediction.
    """
    GX = np.dot(G[:, active_set], X)
    R = M - GX
    penalty = norm_l21(X, n_orient, copy=True)
    nR2 = sum_squared(R)
    p_obj = 0.5 * nR2 + alpha * penalty
    return p_obj, R, nR2, GX


def dgap_l21(M, G, X, active_set, alpha, n_orient):
    """Duality gap for the mixed norm inverse problem.

    See :footcite:`GramfortEtAl2012`.

    Parameters
    ----------
    M : array, shape (n_sensors, n_times)
        The data.
    G : array, shape (n_sensors, n_active)
        The gain matrix a.k.a. lead field.
    X : array, shape (n_active, n_times)
        Sources.
    active_set : array of bool, shape (n_sources, )
        Mask of active sources.
    alpha : float
        The regularization parameter.
    n_orient : int
        Number of dipoles per locations (typically 1 or 3).

    Returns
    -------
    gap : float
        Dual gap.
    p_obj : float
        Primal objective.
    d_obj : float
        Dual objective. gap = p_obj - d_obj.
    R : array, shape (n_sensors, n_times)
        Current residual (M - G * X).

    References
    ----------
    .. footbibilography::
    """
    p_obj, R, nR2, GX = _primal_l21(M, G, X, active_set, alpha, n_orient)
    dual_norm = norm_l2inf(np.dot(G.T, R), n_orient, copy=False)
    scaling = alpha / dual_norm
    scaling = min(scaling, 1.0)
    d_obj = (scaling - 0.5 * (scaling ** 2)) * nR2 + scaling * np.sum(R * GX)

    gap = p_obj - d_obj
    return gap, p_obj, d_obj, R


@verbose
def _mixed_norm_solver_cd(M, G, alpha, lipschitz_constant, maxit=10000,
                          tol=1e-8, verbose=None, init=None, n_orient=1,
                          dgap_freq=10):
    """Solve L21 inverse problem with coordinate descent."""
    from sklearn.linear_model import MultiTaskLasso

    assert M.ndim == G.ndim and M.shape[0] == G.shape[0]

    clf = MultiTaskLasso(alpha=alpha / len(M), tol=tol / sum_squared(M),
                         fit_intercept=False, max_iter=maxit, warm_start=True)
    if init is not None:
        clf.coef_ = init.T
    else:
        clf.coef_ = np.zeros((G.shape[1], M.shape[1])).T
    clf.fit(G, M)

    X = clf.coef_.T
    active_set = np.any(X, axis=1)
    X = X[active_set]
    gap, p_obj, d_obj, _ = dgap_l21(M, G, X, active_set, alpha, n_orient)
    return X, active_set, p_obj


@verbose
def _mixed_norm_solver_bcd(M, G, alpha, lipschitz_constant, maxit=200,
                           tol=1e-8, verbose=None, init=None, n_orient=1,
                           dgap_freq=10, use_accel=True, K=5):
    """Solve L21 inverse problem with block coordinate descent."""
    _, n_times = M.shape
    _, n_sources = G.shape
    n_positions = n_sources // n_orient

    if init is None:
        X = np.zeros((n_sources, n_times))
        R = M.copy()
    else:
        X = init
        R = M - np.dot(G, X)

    E = []  # track primal objective function
    highest_d_obj = - np.inf
    active_set = np.zeros(n_sources, dtype=bool)  # start with full AS

    alpha_lc = alpha / lipschitz_constant

    if use_accel:
        last_K_X = np.empty((K + 1, n_sources, n_times))
        U = np.zeros((K, n_sources * n_times))

    # First make G fortran for faster access to blocks of columns
    G = np.asfortranarray(G)
    # Ensure these are correct for dgemm
    assert R.dtype == np.float64
    assert G.dtype == np.float64
    one_ovr_lc = 1. / lipschitz_constant

    # assert that all the multiplied matrices are fortran contiguous
    assert X.T.flags.f_contiguous
    assert R.T.flags.f_contiguous
    assert G.flags.f_contiguous
    # storing list of contiguous arrays
    list_G_j_c = []
    for j in range(n_positions):
        idx = slice(j * n_orient, (j + 1) * n_orient)
        list_G_j_c.append(np.ascontiguousarray(G[:, idx]))

    for i in range(maxit):
        _bcd(G, X, R, active_set, one_ovr_lc, n_orient, alpha_lc, list_G_j_c)

        if (i + 1) % dgap_freq == 0:
            _, p_obj, d_obj, _ = dgap_l21(M, G, X[active_set], active_set,
                                          alpha, n_orient)
            highest_d_obj = max(d_obj, highest_d_obj)
            gap = p_obj - highest_d_obj
            E.append(p_obj)
            logger.debug("Iteration %d :: p_obj %f :: dgap %f :: n_active %d" %
                         (i + 1, p_obj, gap, np.sum(active_set) / n_orient))

            if gap < tol:
                logger.debug('Convergence reached ! (gap: %s < %s)'
                             % (gap, tol))
                break

        # using Anderson acceleration of the primal variable for faster
        # convergence
        if use_accel:
            last_K_X[i % (K + 1)] = X

            if i % (K + 1) == K:
                for k in range(K):
                    U[k] = last_K_X[k + 1].ravel() - last_K_X[k].ravel()
                C = U @ U.T
                # at least on ARM64 we can't rely on np.linalg.solve to
                # reliably raise LinAlgError here, so use SVD instead
                # equivalent to:
                # z = np.linalg.solve(C, np.ones(K))
                u, s, _ = np.linalg.svd(C, hermitian=True)
                if s[-1] <= 1e-6 * s[0]:
                    logger.debug("Iteration %d: LinAlg Error" % (i + 1))
                    continue
                z = ((u * 1 / s) @ u.T).sum(0)
                c = z / z.sum()
                X_acc = np.sum(
                    last_K_X[:-1] * c[:, None, None], axis=0
                )
                _grp_norm2_acc = groups_norm2(X_acc, n_orient)
                active_set_acc = _grp_norm2_acc != 0
                if n_orient > 1:
                    active_set_acc = np.kron(
                        active_set_acc, np.ones(n_orient, dtype=bool)
                    )
                p_obj = _primal_l21(M, G, X[active_set], active_set, alpha,
                                    n_orient)[0]
                p_obj_acc = _primal_l21(M, G, X_acc[active_set_acc],
                                        active_set_acc, alpha, n_orient)[0]
                if p_obj_acc < p_obj:
                    X = X_acc
                    active_set = active_set_acc
                    R = M - G[:, active_set] @ X[active_set]

    X = X[active_set]

    return X, active_set, E


def _bcd(G, X, R, active_set, one_ovr_lc, n_orient, alpha_lc, list_G_j_c):
    """Implement one full pass of BCD.

    BCD stands for Block Coordinate Descent.
    This function make use of scipy.linalg.get_blas_funcs to speed reasons.

    Parameters
    ----------
    G : array, shape (n_sensors, n_active)
        The gain matrix a.k.a. lead field.
    X : array, shape (n_sources, n_times)
        Sources, modified in place.
    R : array, shape (n_sensors, n_times)
        The residuals: R = M - G @ X, modified in place.
    active_set : array of bool, shape (n_sources, )
        Mask of active sources, modified in place.
    one_ovr_lc : array, shape (n_positions, )
        One over the lipschitz constants.
    n_orient : int
        Number of dipoles per positions (typically 1 or 3).
    n_positions : int
        Number of source positions.
    alpha_lc: array, shape (n_positions, )
        alpha * (Lipschitz constants).
    """
    X_j_new = np.zeros_like(X[:n_orient, :], order='C')
    dgemm = _get_dgemm()

    for j, G_j_c in enumerate(list_G_j_c):
        idx = slice(j * n_orient, (j + 1) * n_orient)
        G_j = G[:, idx]
        X_j = X[idx]
        dgemm(alpha=one_ovr_lc[j], beta=0., a=R.T, b=G_j, c=X_j_new.T,
              overwrite_c=True)
        # X_j_new = G_j.T @ R
        # Mathurin's trick to avoid checking all the entries
        was_non_zero = X_j[0, 0] != 0
        # was_non_zero = np.any(X_j)
        if was_non_zero:
            dgemm(alpha=1., beta=1., a=X_j.T, b=G_j_c.T, c=R.T,
                  overwrite_c=True)
            # R += np.dot(G_j, X_j)
            X_j_new += X_j
        block_norm = sqrt(sum_squared(X_j_new))
        if block_norm <= alpha_lc[j]:
            X_j.fill(0.)
            active_set[idx] = False
        else:
            shrink = max(1.0 - alpha_lc[j] / block_norm, 0.0)
            X_j_new *= shrink
            dgemm(alpha=-1., beta=1., a=X_j_new.T, b=G_j_c.T, c=R.T,
                  overwrite_c=True)
            # R -= np.dot(G_j, X_j_new)
            X_j[:] = X_j_new
            active_set[idx] = True


@verbose
def mixed_norm_solver(M, G, alpha, maxit=3000, tol=1e-8, verbose=None,
                      active_set_size=50, debias=True, n_orient=1,
                      solver='auto', return_gap=False, dgap_freq=10,
                      active_set_init=None, X_init=None):
    """Solve L1/L2 mixed-norm inverse problem with active set strategy.

    See references :footcite:`GramfortEtAl2012,StrohmeierEtAl2016,
    BertrandEtAl2020`.

    Parameters
    ----------
    M : array, shape (n_sensors, n_times)
        The data.
    G : array, shape (n_sensors, n_dipoles)
        The gain matrix a.k.a. lead field.
    alpha : float
        The regularization parameter. It should be between 0 and 100.
        A value of 100 will lead to an empty active set (no active source).
    maxit : int
        The number of iterations.
    tol : float
        Tolerance on dual gap for convergence checking.
    %(verbose)s
    active_set_size : int
        Size of active set increase at each iteration.
    debias : bool
        Debias source estimates.
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
    solver : 'cd' | 'bcd' | 'auto'
        The algorithm to use for the optimization. Block Coordinate Descent
        (BCD) uses Anderson acceleration for faster convergence.
    return_gap : bool
        Return final duality gap.
    dgap_freq : int
        The duality gap is computed every dgap_freq iterations of the solver on
        the active set.
    active_set_init : array, shape (n_dipoles,) or None
        The initial active set (boolean array) used at the first iteration.
        If None, the usual active set strategy is applied.
    X_init : array, shape (n_dipoles, n_times) or None
        The initial weight matrix used for warm starting the solver. If None,
        the weights are initialized at zero.

    Returns
    -------
    X : array, shape (n_active, n_times)
        The source estimates.
    active_set : array, shape (new_active_set_size,)
        The mask of active sources. Note that new_active_set_size is the size
        of the active set after convergence of the solver.
    E : list
        The value of the objective function over the iterations.
    gap : float
        Final duality gap. Returned only if return_gap is True.

    References
    ----------
    .. footbibliography::
    """
    n_dipoles = G.shape[1]
    n_positions = n_dipoles // n_orient
    _, n_times = M.shape
    alpha_max = norm_l2inf(np.dot(G.T, M), n_orient, copy=False)
    logger.info("-- ALPHA MAX : %s" % alpha_max)
    alpha = float(alpha)
    X = np.zeros((n_dipoles, n_times), dtype=G.dtype)

    has_sklearn = True
    try:
        from sklearn.linear_model import MultiTaskLasso  # noqa: F401
    except ImportError:
        has_sklearn = False

    _validate_type(solver, str, 'solver')
    _check_option('solver', solver, ('cd', 'bcd', 'auto'))
    if solver == 'auto':
        if has_sklearn and (n_orient == 1):
            solver = 'cd'
        else:
            solver = 'bcd'

    if solver == 'cd':
        if n_orient == 1 and not has_sklearn:
            warn('Scikit-learn >= 0.12 cannot be found. Using block coordinate'
                 ' descent instead of coordinate descent.')
            solver = 'bcd'
        if n_orient > 1:
            warn('Coordinate descent is only available for fixed orientation. '
                 'Using block coordinate descent instead of coordinate '
                 'descent')
            solver = 'bcd'

    if solver == 'cd':
        logger.info("Using coordinate descent")
        l21_solver = _mixed_norm_solver_cd
        lc = None
    else:
        assert solver == 'bcd'
        logger.info("Using block coordinate descent")
        l21_solver = _mixed_norm_solver_bcd
        G = np.asfortranarray(G)
        if n_orient == 1:
            lc = np.sum(G * G, axis=0)
        else:
            lc = np.empty(n_positions)
            for j in range(n_positions):
                G_tmp = G[:, (j * n_orient):((j + 1) * n_orient)]
                lc[j] = np.linalg.norm(np.dot(G_tmp.T, G_tmp), ord=2)

    if active_set_size is not None:
        E = list()
        highest_d_obj = - np.inf
        if X_init is not None and X_init.shape != (n_dipoles, n_times):
            raise ValueError('Wrong dim for initialized coefficients.')
        active_set = (active_set_init if active_set_init is not None else
                      np.zeros(n_dipoles, dtype=bool))
        idx_large_corr = np.argsort(groups_norm2(np.dot(G.T, M), n_orient))
        new_active_idx = idx_large_corr[-active_set_size:]
        if n_orient > 1:
            new_active_idx = (n_orient * new_active_idx[:, None] +
                              np.arange(n_orient)[None, :]).ravel()
        active_set[new_active_idx] = True
        as_size = np.sum(active_set)
        gap = np.inf
        for k in range(maxit):
            if solver == 'bcd':
                lc_tmp = lc[active_set[::n_orient]]
            elif solver == 'cd':
                lc_tmp = None
            else:
                lc_tmp = 1.01 * np.linalg.norm(G[:, active_set], ord=2) ** 2
            X, as_, _ = l21_solver(M, G[:, active_set], alpha, lc_tmp,
                                   maxit=maxit, tol=tol, init=X_init,
                                   n_orient=n_orient, dgap_freq=dgap_freq)
            active_set[active_set] = as_.copy()
            idx_old_active_set = np.where(active_set)[0]

            _, p_obj, d_obj, R = dgap_l21(M, G, X, active_set, alpha,
                                          n_orient)
            highest_d_obj = max(d_obj, highest_d_obj)
            gap = p_obj - highest_d_obj
            E.append(p_obj)
            logger.info("Iteration %d :: p_obj %f :: dgap %f :: "
                        "n_active_start %d :: n_active_end %d" % (
                            k + 1, p_obj, gap, as_size // n_orient,
                            np.sum(active_set) // n_orient))
            if gap < tol:
                logger.info('Convergence reached ! (gap: %s < %s)'
                            % (gap, tol))
                break

            # add sources if not last iteration
            if k < (maxit - 1):
                idx_large_corr = np.argsort(groups_norm2(np.dot(G.T, R),
                                                         n_orient))
                new_active_idx = idx_large_corr[-active_set_size:]
                if n_orient > 1:
                    new_active_idx = (n_orient * new_active_idx[:, None] +
                                      np.arange(n_orient)[None, :])
                    new_active_idx = new_active_idx.ravel()
                active_set[new_active_idx] = True
                idx_active_set = np.where(active_set)[0]
                as_size = np.sum(active_set)
                X_init = np.zeros((as_size, n_times), dtype=X.dtype)
                idx = np.searchsorted(idx_active_set, idx_old_active_set)
                X_init[idx] = X
        else:
            warn('Did NOT converge ! (gap: %s > %s)' % (gap, tol))
    else:
        X, active_set, E = l21_solver(M, G, alpha, lc, maxit=maxit,
                                      tol=tol, n_orient=n_orient, init=None)
        if return_gap:
            gap = dgap_l21(M, G, X, active_set, alpha, n_orient)[0]

    if np.any(active_set) and debias:
        bias = compute_bias(M, G[:, active_set], X, n_orient=n_orient)
        X *= bias[:, np.newaxis]

    logger.info('Final active set size: %s' % (np.sum(active_set) // n_orient))

    if return_gap:
        return X, active_set, E, gap
    else:
        return X, active_set, E


@verbose
def iterative_mixed_norm_solver(M, G, alpha, n_mxne_iter, maxit=3000,
                                tol=1e-8, verbose=None, active_set_size=50,
                                debias=True, n_orient=1, dgap_freq=10,
                                solver='auto', weight_init=None):
    """Solve L0.5/L2 mixed-norm inverse problem with active set strategy.

    See reference :footcite:`StrohmeierEtAl2016`.

    Parameters
    ----------
    M : array, shape (n_sensors, n_times)
        The data.
    G : array, shape (n_sensors, n_dipoles)
        The gain matrix a.k.a. lead field.
    alpha : float
        The regularization parameter. It should be between 0 and 100.
        A value of 100 will lead to an empty active set (no active source).
    n_mxne_iter : int
        The number of MxNE iterations. If > 1, iterative reweighting
        is applied.
    maxit : int
        The number of iterations.
    tol : float
        Tolerance on dual gap for convergence checking.
    %(verbose)s
    active_set_size : int
        Size of active set increase at each iteration.
    debias : bool
        Debias source estimates.
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
    dgap_freq : int or np.inf
        The duality gap is evaluated every dgap_freq iterations.
    solver : 'cd' | 'bcd' | 'auto'
        The algorithm to use for the optimization.
    weight_init : array, shape (n_dipoles,) or None
        The initial weight used for reweighting the gain matrix. If None, the
        weights are initialized with ones.

    Returns
    -------
    X : array, shape (n_active, n_times)
        The source estimates.
    active_set : array
        The mask of active sources.
    E : list
        The value of the objective function over the iterations.

    References
    ----------
    .. footbibliography::
    """
    def g(w):
        return np.sqrt(np.sqrt(groups_norm2(w.copy(), n_orient)))

    def gprime(w):
        return 2. * np.repeat(g(w), n_orient).ravel()

    E = list()

    if weight_init is not None and weight_init.shape != (G.shape[1],):
        raise ValueError('Wrong dimension for weight initialization. Got %s. '
                         'Expected %s.' % (weight_init.shape, (G.shape[1],)))

    weights = weight_init if weight_init is not None else np.ones(G.shape[1])
    active_set = (weights != 0)
    weights = weights[active_set]
    X = np.zeros((G.shape[1], M.shape[1]))

    for k in range(n_mxne_iter):
        X0 = X.copy()
        active_set_0 = active_set.copy()
        G_tmp = G[:, active_set] * weights[np.newaxis, :]

        if active_set_size is not None:
            if np.sum(active_set) > (active_set_size * n_orient):
                X, _active_set, _ = mixed_norm_solver(
                    M, G_tmp, alpha, debias=False, n_orient=n_orient,
                    maxit=maxit, tol=tol, active_set_size=active_set_size,
                    dgap_freq=dgap_freq, solver=solver, verbose=verbose)
            else:
                X, _active_set, _ = mixed_norm_solver(
                    M, G_tmp, alpha, debias=False, n_orient=n_orient,
                    maxit=maxit, tol=tol, active_set_size=None,
                    dgap_freq=dgap_freq, solver=solver, verbose=verbose)
        else:
            X, _active_set, _ = mixed_norm_solver(
                M, G_tmp, alpha, debias=False, n_orient=n_orient,
                maxit=maxit, tol=tol, active_set_size=None,
                dgap_freq=dgap_freq, solver=solver, verbose=verbose)

        logger.info('active set size %d' % (_active_set.sum() / n_orient))

        if _active_set.sum() > 0:
            active_set[active_set] = _active_set
            # Reapply weights to have correct unit
            X *= weights[_active_set][:, np.newaxis]
            weights = gprime(X)
            p_obj = 0.5 * np.linalg.norm(M - np.dot(G[:, active_set], X),
                                         'fro') ** 2. + alpha * np.sum(g(X))
            E.append(p_obj)

            # Check convergence
            if ((k >= 1) and np.all(active_set == active_set_0) and
                    np.all(np.abs(X - X0) < tol)):
                print('Convergence reached after %d reweightings!' % k)
                break
        else:
            active_set = np.zeros_like(active_set)
            p_obj = 0.5 * np.linalg.norm(M) ** 2.
            E.append(p_obj)
            break

    if np.any(active_set) and debias:
        bias = compute_bias(M, G[:, active_set], X, n_orient=n_orient)
        X *= bias[:, np.newaxis]

    return X, active_set, E


###############################################################################
# TF-MxNE

@verbose
def tf_lipschitz_constant(M, G, phi, phiT, tol=1e-3, verbose=None):
    """Compute lipschitz constant for FISTA.

    It uses a power iteration method.
    """
    n_times = M.shape[1]
    n_points = G.shape[1]
    iv = np.ones((n_points, n_times), dtype=np.float64)
    v = phi(iv)
    L = 1e100
    for it in range(100):
        L_old = L
        logger.info('Lipschitz estimation: iteration = %d' % it)
        iv = np.real(phiT(v))
        Gv = np.dot(G, iv)
        GtGv = np.dot(G.T, Gv)
        w = phi(GtGv)
        L = np.max(np.abs(w))  # l_inf norm
        v = w / L
        if abs((L - L_old) / L_old) < tol:
            break
    return L


def safe_max_abs(A, ia):
    """Compute np.max(np.abs(A[ia])) possible with empty A."""
    if np.sum(ia):  # ia is not empty
        return np.max(np.abs(A[ia]))
    else:
        return 0.


def safe_max_abs_diff(A, ia, B, ib):
    """Compute np.max(np.abs(A)) possible with empty A."""
    A = A[ia] if np.sum(ia) else 0.0
    B = B[ib] if np.sum(ia) else 0.0
    return np.max(np.abs(A - B))


class _Phi(object):
    """Have phi stft as callable w/o using a lambda that does not pickle."""

    def __init__(self, wsize, tstep, n_coefs, n_times):  # noqa: D102
        self.wsize = np.atleast_1d(wsize)
        self.tstep = np.atleast_1d(tstep)
        self.n_coefs = np.atleast_1d(n_coefs)
        self.n_dicts = len(tstep)
        self.n_freqs = wsize // 2 + 1
        self.n_steps = self.n_coefs // self.n_freqs
        self.n_times = n_times
        # ravel freq+time here
        self.ops = list()
        for ws, ts in zip(self.wsize, self.tstep):
            self.ops.append(
                stft(np.eye(n_times), ws, ts,
                     verbose=False).reshape(n_times, -1))

    def __call__(self, x):  # noqa: D105
        if self.n_dicts == 1:
            return x @ self.ops[0]
        else:
            return np.hstack(
                [x @ op for op in self.ops]) / np.sqrt(self.n_dicts)

    def norm(self, z, ord=2):
        """Squared L2 norm if ord == 2 and L1 norm if order == 1."""
        if ord not in (1, 2):
            raise ValueError('Only supported norm order are 1 and 2. '
                             'Got ord = %s' % ord)
        stft_norm = stft_norm1 if ord == 1 else stft_norm2
        norm = 0.
        if len(self.n_coefs) > 1:
            z_ = np.array_split(np.atleast_2d(z), np.cumsum(self.n_coefs)[:-1],
                                axis=1)
        else:
            z_ = [np.atleast_2d(z)]
        for i in range(len(z_)):
            norm += stft_norm(
                z_[i].reshape(-1, self.n_freqs[i], self.n_steps[i]))
        return norm


class _PhiT(object):
    """Have phi.T istft as callable w/o using a lambda that does not pickle."""

    def __init__(self, tstep, n_freqs, n_steps, n_times):  # noqa: D102
        self.tstep = tstep
        self.n_freqs = n_freqs
        self.n_steps = n_steps
        self.n_times = n_times
        self.n_dicts = len(tstep) if isinstance(tstep, np.ndarray) else 1
        self.n_coefs = list()
        self.op_re = list()
        self.op_im = list()
        for nf, ns, ts in zip(self.n_freqs, self.n_steps, self.tstep):
            nc = nf * ns
            self.n_coefs.append(nc)
            eye = np.eye(nc).reshape(nf, ns, nf, ns)
            self.op_re.append(istft(
                eye, ts, n_times).reshape(nc, n_times))
            self.op_im.append(istft(
                eye * 1j, ts, n_times).reshape(nc, n_times))

    def __call__(self, z):  # noqa: D105
        if self.n_dicts == 1:
            return z.real @ self.op_re[0] + z.imag @ self.op_im[0]
        else:
            x_out = np.zeros((z.shape[0], self.n_times))
            z_ = np.array_split(z, np.cumsum(self.n_coefs)[:-1], axis=1)
            for this_z, op_re, op_im in zip(z_, self.op_re, self.op_im):
                x_out += this_z.real @ op_re + this_z.imag @ op_im
            return x_out / np.sqrt(self.n_dicts)


def norm_l21_tf(Z, phi, n_orient, w_space=None):
    """L21 norm for TF."""
    if Z.shape[0]:
        l21_norm = np.sqrt(
            phi.norm(Z, ord=2).reshape(-1, n_orient).sum(axis=1))
        if w_space is not None:
            l21_norm *= w_space
        l21_norm = l21_norm.sum()
    else:
        l21_norm = 0.
    return l21_norm


def norm_l1_tf(Z, phi, n_orient, w_time):
    """L1 norm for TF."""
    if Z.shape[0]:
        n_positions = Z.shape[0] // n_orient
        Z_ = np.sqrt(np.sum(
            (np.abs(Z) ** 2.).reshape((n_orient, -1), order='F'), axis=0))
        Z_ = Z_.reshape((n_positions, -1), order='F')
        if w_time is not None:
            Z_ *= w_time
        l1_norm = phi.norm(Z_, ord=1).sum()
    else:
        l1_norm = 0.
    return l1_norm


def norm_epsilon(Y, l1_ratio, phi, w_space=1., w_time=None):
    """Weighted epsilon norm.

    The weighted epsilon norm is the dual norm of::

    w_{space} * (1. - l1_ratio) * ||Y||_2 + l1_ratio * ||Y||_{1, w_{time}}.

    where `||Y||_{1, w_{time}} = (np.abs(Y) * w_time).sum()`

    Warning: it takes into account the fact that Y only contains coefficients
    corresponding to the positive frequencies (see `stft_norm2()`): some
    entries will be counted twice. It is also assumed that all entries of both
    Y and w_time are non-negative. See
    :footcite:`NdiayeEtAl2016,BurdakovMerkulov2001`.

    Parameters
    ----------
    Y : array, shape (n_coefs,)
        The input data.
    l1_ratio : float between 0 and 1
        Tradeoff between L2 and L1 regularization. When it is 0, no temporal
        regularization is applied.
    phi : instance of _Phi
        The TF operator.
    w_space : float
        Scalar weight of the L2 norm. By default, it is taken equal to 1.
    w_time : array, shape (n_coefs, ) | None
        Weights of each TF coefficient in the L1 norm. If None, weights equal
        to 1 are used.


    Returns
    -------
    nu : float
        The value of the dual norm evaluated at Y.

    References
    ----------
    .. footbibliography::
    """
    # since the solution is invariant to flipped signs in Y, all entries
    # of Y are assumed positive

    # Add negative freqs: count all freqs twice except first and last:
    freqs_count = np.full(len(Y), 2)
    for i, fc in enumerate(np.array_split(freqs_count,
                                          np.cumsum(phi.n_coefs)[:-1])):
        fc[:phi.n_steps[i]] = 1
        fc[-phi.n_steps[i]:] = 1

    # exclude 0 weights:
    if w_time is not None:
        nonzero_weights = (w_time != 0.0)
        Y = Y[nonzero_weights]
        freqs_count = freqs_count[nonzero_weights]
        w_time = w_time[nonzero_weights]

    norm_inf_Y = np.max(Y / w_time) if w_time is not None else np.max(Y)
    if l1_ratio == 1.:
        # dual norm of L1 weighted is Linf with inverse weights
        return norm_inf_Y
    elif l1_ratio == 0.:
        # dual norm of L2 is L2
        return np.sqrt(phi.norm(Y[None, :], ord=2).sum())

    if norm_inf_Y == 0.:
        return 0.

    # ignore some values of Y by lower bound on dual norm:
    if w_time is None:
        idx = Y > l1_ratio * norm_inf_Y
    else:
        idx = Y > l1_ratio * np.max(Y / (w_space * (1. - l1_ratio) +
                                    l1_ratio * w_time))

    if idx.sum() == 1:
        return norm_inf_Y

    # sort both Y / w_time and freqs_count at the same time
    if w_time is not None:
        idx_sort = np.argsort(Y[idx] / w_time[idx])[::-1]
        w_time = w_time[idx][idx_sort]
    else:
        idx_sort = np.argsort(Y[idx])[::-1]

    Y = Y[idx][idx_sort]
    freqs_count = freqs_count[idx][idx_sort]

    Y = np.repeat(Y, freqs_count)
    if w_time is not None:
        w_time = np.repeat(w_time, freqs_count)

    K = Y.shape[0]
    if w_time is None:
        p_sum_Y2 = np.cumsum(Y ** 2)
        p_sum_w2 = np.arange(1, K + 1)
        p_sum_Yw = np.cumsum(Y)
        upper = p_sum_Y2 / Y ** 2 - 2. * p_sum_Yw / Y + p_sum_w2
    else:
        p_sum_Y2 = np.cumsum(Y ** 2)
        p_sum_w2 = np.cumsum(w_time ** 2)
        p_sum_Yw = np.cumsum(Y * w_time)
        upper = (p_sum_Y2 / (Y / w_time) ** 2 -
                 2. * p_sum_Yw / (Y / w_time) + p_sum_w2)
    upper_greater = np.where(upper > w_space ** 2 * (1. - l1_ratio) ** 2 /
                             l1_ratio ** 2)[0]

    i0 = upper_greater[0] - 1 if upper_greater.size else K - 1

    p_sum_Y2 = p_sum_Y2[i0]
    p_sum_w2 = p_sum_w2[i0]
    p_sum_Yw = p_sum_Yw[i0]

    denom = l1_ratio ** 2 * p_sum_w2 - w_space ** 2 * (1. - l1_ratio) ** 2
    if np.abs(denom) < 1e-10:
        return p_sum_Y2 / (2. * l1_ratio * p_sum_Yw)
    else:
        delta = (l1_ratio * p_sum_Yw) ** 2 - p_sum_Y2 * denom
        return (l1_ratio * p_sum_Yw - np.sqrt(delta)) / denom


def norm_epsilon_inf(G, R, phi, l1_ratio, n_orient, w_space=None, w_time=None):
    """Weighted epsilon-inf norm of phi(np.dot(G.T, R)).

    Parameters
    ----------
    G : array, shape (n_sensors, n_sources)
        Gain matrix a.k.a. lead field.
    R : array, shape (n_sensors, n_times)
        Residual.
    phi : instance of _Phi
        The TF operator.
    l1_ratio : float between 0 and 1
        Parameter controlling the tradeoff between L21 and L1 regularization.
        0 corresponds to an absence of temporal regularization, ie MxNE.
    n_orient : int
        Number of dipoles per location (typically 1 or 3).
    w_space : array, shape (n_positions,) or None.
        Weights for the L2 term of the epsilon norm. If None, weights are
        all equal to 1.
    w_time : array, shape (n_positions, n_coefs) or None
        Weights for the L1 term of the epsilon norm. If None, weights are
        all equal to 1.

    Returns
    -------
    nu : float
        The maximum value of the epsilon norms over groups of n_orient dipoles
        (consecutive rows of phi(np.dot(G.T, R))).
    """
    n_positions = G.shape[1] // n_orient
    GTRPhi = np.abs(phi(np.dot(G.T, R)))
    # norm over orientations:
    GTRPhi = GTRPhi.reshape((n_orient, -1), order='F')
    GTRPhi = np.linalg.norm(GTRPhi, axis=0)
    GTRPhi = GTRPhi.reshape((n_positions, -1), order='F')
    nu = 0.
    for idx in range(n_positions):
        GTRPhi_ = GTRPhi[idx]
        w_t = w_time[idx] if w_time is not None else None
        w_s = w_space[idx] if w_space is not None else 1.
        norm_eps = norm_epsilon(GTRPhi_, l1_ratio, phi, w_space=w_s,
                                w_time=w_t)
        if norm_eps > nu:
            nu = norm_eps

    return nu


def dgap_l21l1(M, G, Z, active_set, alpha_space, alpha_time, phi, phiT,
               n_orient, highest_d_obj, w_space=None, w_time=None):
    """Duality gap for the time-frequency mixed norm inverse problem.

    See :footcite:`GramfortEtAl2012,NdiayeEtAl2016`

    Parameters
    ----------
    M : array, shape (n_sensors, n_times)
        The data.
    G : array, shape (n_sensors, n_sources)
        Gain matrix a.k.a. lead field.
    Z : array, shape (n_active, n_coefs)
        Sources in TF domain.
    active_set : array of bool, shape (n_sources, )
        Mask of active sources.
    alpha_space : float
        The spatial regularization parameter.
    alpha_time : float
        The temporal regularization parameter. The higher it is the smoother
        will be the estimated time series.
    phi : instance of _Phi
        The TF operator.
    phiT : instance of _PhiT
        The transpose of the TF operator.
    n_orient : int
        Number of dipoles per locations (typically 1 or 3).
    highest_d_obj : float
        The highest value of the dual objective so far.
    w_space : array, shape (n_positions, )
        Array of spatial weights.
    w_time : array, shape (n_positions, n_coefs)
        Array of TF weights.

    Returns
    -------
    gap : float
        Dual gap
    p_obj : float
        Primal objective
    d_obj : float
        Dual objective. gap = p_obj - d_obj
    R : array, shape (n_sensors, n_times)
        Current residual (M - G * X)

    References
    ----------
    .. footbibliography::
    """
    X = phiT(Z)
    GX = np.dot(G[:, active_set], X)
    R = M - GX

    # some functions need w_time only on active_set, other need it completely
    if w_time is not None:
        w_time_as = w_time[active_set[::n_orient]]
    else:
        w_time_as = None
    if w_space is not None:
        w_space_as = w_space[active_set[::n_orient]]
    else:
        w_space_as = None

    penaltyl1 = norm_l1_tf(Z, phi, n_orient, w_time_as)
    penaltyl21 = norm_l21_tf(Z, phi, n_orient, w_space_as)
    nR2 = sum_squared(R)
    p_obj = 0.5 * nR2 + alpha_space * penaltyl21 + alpha_time * penaltyl1

    l1_ratio = alpha_time / (alpha_space + alpha_time)
    dual_norm = norm_epsilon_inf(G, R, phi, l1_ratio, n_orient,
                                 w_space=w_space, w_time=w_time)
    scaling = min(1., (alpha_space + alpha_time) / dual_norm)

    d_obj = (scaling - 0.5 * (scaling ** 2)) * nR2 + scaling * np.sum(R * GX)
    d_obj = max(d_obj, highest_d_obj)

    gap = p_obj - d_obj
    return gap, p_obj, d_obj, R


def _tf_mixed_norm_solver_bcd_(M, G, Z, active_set, candidates, alpha_space,
                               alpha_time, lipschitz_constant, phi, phiT,
                               w_space=None, w_time=None, n_orient=1,
                               maxit=200, tol=1e-8, dgap_freq=10, perc=None,
                               timeit=True, verbose=None):
    n_sources = G.shape[1]
    n_positions = n_sources // n_orient

    # First make G fortran for faster access to blocks of columns
    Gd = np.asfortranarray(G)
    G = np.ascontiguousarray(
        Gd.T.reshape(n_positions, n_orient, -1).transpose(0, 2, 1))

    R = M.copy()  # residual
    active = np.where(active_set[::n_orient])[0]
    for idx in active:
        R -= np.dot(G[idx], phiT(Z[idx]))

    E = []  # track primal objective function

    if w_time is None:
        alpha_time_lc = alpha_time / lipschitz_constant
    else:
        alpha_time_lc = alpha_time * w_time / lipschitz_constant[:, None]
    if w_space is None:
        alpha_space_lc = alpha_space / lipschitz_constant
    else:
        alpha_space_lc = alpha_space * w_space / lipschitz_constant

    converged = False
    d_obj = - np.inf

    for i in range(maxit):
        for jj in candidates:
            ids = jj * n_orient
            ide = ids + n_orient

            G_j = G[jj]
            Z_j = Z[jj]
            active_set_j = active_set[ids:ide]

            was_active = np.any(active_set_j)

            # gradient step
            GTR = np.dot(G_j.T, R) / lipschitz_constant[jj]
            X_j_new = GTR.copy()

            if was_active:
                X_j = phiT(Z_j)
                R += np.dot(G_j, X_j)
                X_j_new += X_j

            rows_norm = np.linalg.norm(X_j_new, 'fro')
            if rows_norm <= alpha_space_lc[jj]:
                if was_active:
                    Z[jj] = 0.0
                    active_set_j[:] = False
            else:
                GTR_phi = phi(GTR)
                if was_active:
                    Z_j_new = Z_j + GTR_phi
                else:
                    Z_j_new = GTR_phi
                col_norm = np.linalg.norm(Z_j_new, axis=0)

                if np.all(col_norm <= alpha_time_lc[jj]):
                    Z[jj] = 0.0
                    active_set_j[:] = False
                else:
                    # l1
                    shrink = np.maximum(1.0 - alpha_time_lc[jj] / np.maximum(
                                        col_norm, alpha_time_lc[jj]), 0.0)
                    if w_time is not None:
                        shrink[w_time[jj] == 0.0] = 0.0
                    Z_j_new *= shrink[np.newaxis, :]

                    # l21
                    shape_init = Z_j_new.shape
                    row_norm = np.sqrt(phi.norm(Z_j_new, ord=2).sum())
                    if row_norm <= alpha_space_lc[jj]:
                        Z[jj] = 0.0
                        active_set_j[:] = False
                    else:
                        shrink = np.maximum(
                            1.0 - alpha_space_lc[jj] /
                            np.maximum(row_norm, alpha_space_lc[jj]), 0.0)
                        Z_j_new *= shrink
                        Z[jj] = Z_j_new.reshape(-1, *shape_init[1:]).copy()
                        active_set_j[:] = True
                        Z_j_phi_T = phiT(Z[jj])
                        R -= np.dot(G_j, Z_j_phi_T)

        if (i + 1) % dgap_freq == 0:
            Zd = np.vstack([Z[pos] for pos in range(n_positions)
                            if np.any(Z[pos])])
            gap, p_obj, d_obj, _ = dgap_l21l1(
                M, Gd, Zd, active_set, alpha_space, alpha_time, phi, phiT,
                n_orient, d_obj, w_space=w_space, w_time=w_time)
            converged = (gap < tol)
            E.append(p_obj)
            logger.info("\n    Iteration %d :: n_active %d" % (
                        i + 1, np.sum(active_set) / n_orient))
            logger.info("    dgap %.2e :: p_obj %f :: d_obj %f" % (
                        gap, p_obj, d_obj))

        if converged:
            break

        if perc is not None:
            if np.sum(active_set) / float(n_orient) <= perc * n_positions:
                break

    return Z, active_set, E, converged


@verbose
def _tf_mixed_norm_solver_bcd_active_set(M, G, alpha_space, alpha_time,
                                         lipschitz_constant, phi, phiT,
                                         Z_init=None, w_space=None,
                                         w_time=None, n_orient=1, maxit=200,
                                         tol=1e-8, dgap_freq=10,
                                         verbose=None):

    n_sensors, n_times = M.shape
    n_sources = G.shape[1]
    n_positions = n_sources // n_orient

    Z = dict.fromkeys(np.arange(n_positions), 0.0)
    active_set = np.zeros(n_sources, dtype=bool)
    active = []
    if Z_init is not None:
        if Z_init.shape != (n_sources, phi.n_coefs.sum()):
            raise Exception('Z_init must be None or an array with shape '
                            '(n_sources, n_coefs).')
        for ii in range(n_positions):
            if np.any(Z_init[ii * n_orient:(ii + 1) * n_orient]):
                active_set[ii * n_orient:(ii + 1) * n_orient] = True
                active.append(ii)
        if len(active):
            Z.update(dict(zip(active,
                              np.vsplit(Z_init[active_set], len(active)))))

    E = []
    candidates = range(n_positions)
    d_obj = -np.inf

    while True:
        # single BCD pass on all positions:
        Z_init = dict.fromkeys(np.arange(n_positions), 0.0)
        Z_init.update(dict(zip(active, Z.values())))
        Z, active_set, E_tmp, _ = _tf_mixed_norm_solver_bcd_(
            M, G, Z_init, active_set, candidates, alpha_space, alpha_time,
            lipschitz_constant, phi, phiT, w_space=w_space, w_time=w_time,
            n_orient=n_orient, maxit=1, tol=tol, perc=None, verbose=verbose)

        E += E_tmp

        # multiple BCD pass on active positions:
        active = np.where(active_set[::n_orient])[0]
        Z_init = dict(zip(range(len(active)), [Z[idx] for idx in active]))
        candidates_ = range(len(active))
        if w_space is not None:
            w_space_as = w_space[active_set[::n_orient]]
        else:
            w_space_as = None
        if w_time is not None:
            w_time_as = w_time[active_set[::n_orient]]
        else:
            w_time_as = None

        Z, as_, E_tmp, converged = _tf_mixed_norm_solver_bcd_(
            M, G[:, active_set], Z_init,
            np.ones(len(active) * n_orient, dtype=bool),
            candidates_, alpha_space, alpha_time,
            lipschitz_constant[active_set[::n_orient]], phi, phiT,
            w_space=w_space_as, w_time=w_time_as,
            n_orient=n_orient, maxit=maxit, tol=tol,
            dgap_freq=dgap_freq, perc=0.5,
            verbose=verbose)
        active = np.where(active_set[::n_orient])[0]
        active_set[active_set] = as_.copy()
        E += E_tmp

        converged = True
        if converged:
            Zd = np.vstack([Z[pos] for pos in range(len(Z)) if np.any(Z[pos])])
            gap, p_obj, d_obj, _ = dgap_l21l1(
                M, G, Zd, active_set, alpha_space, alpha_time,
                phi, phiT, n_orient, d_obj, w_space, w_time)
            logger.info("\ndgap %.2e :: p_obj %f :: d_obj %f :: n_active %d"
                        % (gap, p_obj, d_obj, np.sum(active_set) / n_orient))
            if gap < tol:
                logger.info("\nConvergence reached!\n")
                break

    if active_set.sum():
        Z = np.vstack([Z[pos] for pos in range(len(Z)) if np.any(Z[pos])])
        X = phiT(Z)
    else:
        Z = np.zeros((0, phi.n_coefs.sum()), dtype=np.complex128)
        X = np.zeros((0, n_times))

    return X, Z, active_set, E, gap


@verbose
def tf_mixed_norm_solver(M, G, alpha_space, alpha_time, wsize=64, tstep=4,
                         n_orient=1, maxit=200, tol=1e-8,
                         active_set_size=None, debias=True, return_gap=False,
                         dgap_freq=10, verbose=None):
    """Solve TF L21+L1 inverse solver with BCD and active set approach.

    See :footcite:`GramfortEtAl2013b,GramfortEtAl2011,BekhtiEtAl2016`.

    Parameters
    ----------
    M : array, shape (n_sensors, n_times)
        The data.
    G : array, shape (n_sensors, n_dipoles)
        The gain matrix a.k.a. lead field.
    alpha_space : float
        The spatial regularization parameter.
    alpha_time : float
        The temporal regularization parameter. The higher it is the smoother
        will be the estimated time series.
    wsize: int or array-like
        Length of the STFT window in samples (must be a multiple of 4).
        If an array is passed, multiple TF dictionaries are used (each having
        its own wsize and tstep) and each entry of wsize must be a multiple
        of 4.
    tstep: int or array-like
        Step between successive windows in samples (must be a multiple of 2,
        a divider of wsize and smaller than wsize/2) (default: wsize/2).
        If an array is passed, multiple TF dictionaries are used (each having
        its own wsize and tstep), and each entry of tstep must be a multiple
        of 2 and divide the corresponding entry of wsize.
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
    maxit : int
        The number of iterations.
    tol : float
        If absolute difference between estimates at 2 successive iterations
        is lower than tol, the convergence is reached.
    debias : bool
        Debias source estimates.
    return_gap : bool
        Return final duality gap.
    dgap_freq : int or np.inf
        The duality gap is evaluated every dgap_freq iterations.
    %(verbose)s

    Returns
    -------
    X : array, shape (n_active, n_times)
        The source estimates.
    active_set : array
        The mask of active sources.
    E : list
        The value of the objective function every dgap_freq iteration. If
        log_objective is False or dgap_freq is np.inf, it will be empty.
    gap : float
        Final duality gap. Returned only if return_gap is True.

    References
    ----------
    .. footbibliography::
    """
    n_sensors, n_times = M.shape
    n_sensors, n_sources = G.shape
    n_positions = n_sources // n_orient

    tstep = np.atleast_1d(tstep)
    wsize = np.atleast_1d(wsize)
    if len(tstep) != len(wsize):
        raise ValueError('The same number of window sizes and steps must be '
                         'passed. Got tstep = %s and wsize = %s' %
                         (tstep, wsize))

    n_steps = np.ceil(M.shape[1] / tstep.astype(float)).astype(int)
    n_freqs = wsize // 2 + 1
    n_coefs = n_steps * n_freqs
    phi = _Phi(wsize, tstep, n_coefs, n_times)
    phiT = _PhiT(tstep, n_freqs, n_steps, n_times)

    if n_orient == 1:
        lc = np.sum(G * G, axis=0)
    else:
        lc = np.empty(n_positions)
        for j in range(n_positions):
            G_tmp = G[:, (j * n_orient):((j + 1) * n_orient)]
            lc[j] = np.linalg.norm(np.dot(G_tmp.T, G_tmp), ord=2)

    logger.info("Using block coordinate descent with active set approach")
    X, Z, active_set, E, gap = _tf_mixed_norm_solver_bcd_active_set(
        M, G, alpha_space, alpha_time, lc, phi, phiT,
        Z_init=None, n_orient=n_orient, maxit=maxit, tol=tol,
        dgap_freq=dgap_freq, verbose=None)

    if np.any(active_set) and debias:
        bias = compute_bias(M, G[:, active_set], X, n_orient=n_orient)
        X *= bias[:, np.newaxis]

    if return_gap:
        return X, active_set, E, gap
    else:
        return X, active_set, E


@verbose
def iterative_tf_mixed_norm_solver(M, G, alpha_space, alpha_time,
                                   n_tfmxne_iter, wsize=64, tstep=4,
                                   maxit=3000, tol=1e-8, debias=True,
                                   n_orient=1, dgap_freq=10, verbose=None):
    """Solve TF L0.5/L1 + L0.5 inverse problem with BCD + active set approach.

    Parameters
    ----------
    M: array, shape (n_sensors, n_times)
        The data.
    G: array, shape (n_sensors, n_dipoles)
        The gain matrix a.k.a. lead field.
    alpha_space: float
        The spatial regularization parameter. The higher it is the less there
        will be active sources.
    alpha_time : float
        The temporal regularization parameter. The higher it is the smoother
        will be the estimated time series. 0 means no temporal regularization,
        a.k.a. irMxNE.
    n_tfmxne_iter : int
        Number of TF-MxNE iterations. If > 1, iterative reweighting is applied.
    wsize : int or array-like
        Length of the STFT window in samples (must be a multiple of 4).
        If an array is passed, multiple TF dictionaries are used (each having
        its own wsize and tstep) and each entry of wsize must be a multiple
        of 4.
    tstep : int or array-like
        Step between successive windows in samples (must be a multiple of 2,
        a divider of wsize and smaller than wsize/2) (default: wsize/2).
        If an array is passed, multiple TF dictionaries are used (each having
        its own wsize and tstep), and each entry of tstep must be a multiple
        of 2 and divide the corresponding entry of wsize.
    maxit : int
        The maximum number of iterations for each TF-MxNE problem.
    tol : float
        If absolute difference between estimates at 2 successive iterations
        is lower than tol, the convergence is reached. Also used as criterion
        on duality gap for each TF-MxNE problem.
    debias : bool
        Debias source estimates.
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
    dgap_freq : int or np.inf
        The duality gap is evaluated every dgap_freq iterations.
    %(verbose)s

    Returns
    -------
    X : array, shape (n_active, n_times)
        The source estimates.
    active_set : array
        The mask of active sources.
    E : list
        The value of the objective function over iterations.
    """
    n_sensors, n_times = M.shape
    n_sources = G.shape[1]
    n_positions = n_sources // n_orient

    tstep = np.atleast_1d(tstep)
    wsize = np.atleast_1d(wsize)
    if len(tstep) != len(wsize):
        raise ValueError('The same number of window sizes and steps must be '
                         'passed. Got tstep = %s and wsize = %s' %
                         (tstep, wsize))

    n_steps = np.ceil(n_times / tstep.astype(float)).astype(int)
    n_freqs = wsize // 2 + 1
    n_coefs = n_steps * n_freqs
    phi = _Phi(wsize, tstep, n_coefs, n_times)
    phiT = _PhiT(tstep, n_freqs, n_steps, n_times)

    if n_orient == 1:
        lc = np.sum(G * G, axis=0)
    else:
        lc = np.empty(n_positions)
        for j in range(n_positions):
            G_tmp = G[:, (j * n_orient):((j + 1) * n_orient)]
            lc[j] = np.linalg.norm(np.dot(G_tmp.T, G_tmp), ord=2)

    # space and time penalties, and inverse of their derivatives:
    def g_space(Z):
        return np.sqrt(np.sqrt(phi.norm(Z, ord=2).reshape(
            -1, n_orient).sum(axis=1)))

    def g_space_prime_inv(Z):
        return 2. * g_space(Z)

    def g_time(Z):
        return np.sqrt(np.sqrt(np.sum((np.abs(Z) ** 2.).reshape(
            (n_orient, -1), order='F'), axis=0)).reshape(
            (-1, Z.shape[1]), order='F'))

    def g_time_prime_inv(Z):
        return 2. * g_time(Z)

    E = list()

    active_set = np.ones(n_sources, dtype=bool)
    Z = np.zeros((n_sources, phi.n_coefs.sum()), dtype=np.complex128)

    for k in range(n_tfmxne_iter):
        active_set_0 = active_set.copy()
        Z0 = Z.copy()

        if k == 0:
            w_space = None
            w_time = None
        else:
            w_space = 1. / g_space_prime_inv(Z)
            w_time = g_time_prime_inv(Z)
            w_time[w_time == 0.0] = -1.
            w_time = 1. / w_time
            w_time[w_time < 0.0] = 0.0

        X, Z, active_set_, E_, _ = _tf_mixed_norm_solver_bcd_active_set(
            M, G[:, active_set], alpha_space, alpha_time,
            lc[active_set[::n_orient]], phi, phiT,
            Z_init=Z, w_space=w_space, w_time=w_time, n_orient=n_orient,
            maxit=maxit, tol=tol, dgap_freq=dgap_freq, verbose=None)

        active_set[active_set] = active_set_

        if active_set.sum() > 0:
            l21_penalty = np.sum(g_space(Z.copy()))
            l1_penalty = phi.norm(g_time(Z.copy()), ord=1).sum()

            p_obj = (0.5 * np.linalg.norm(M - np.dot(G[:, active_set], X),
                     'fro') ** 2. + alpha_space * l21_penalty +
                     alpha_time * l1_penalty)
            E.append(p_obj)

            logger.info('Iteration %d: active set size=%d, E=%f' % (
                        k + 1, active_set.sum() / n_orient, p_obj))

            # Check convergence
            if np.array_equal(active_set, active_set_0):
                max_diff = np.amax(np.abs(Z - Z0))
                if (max_diff < tol):
                    print('Convergence reached after %d reweightings!' % k)
                    break
        else:
            p_obj = 0.5 * np.linalg.norm(M) ** 2.
            E.append(p_obj)
            logger.info('Iteration %d: as_size=%d, E=%f' % (
                        k + 1, active_set.sum() / n_orient, p_obj))
            break

    if debias:
        if active_set.sum() > 0:
            bias = compute_bias(M, G[:, active_set], X, n_orient=n_orient)
            X *= bias[:, np.newaxis]

    return X, active_set, E
