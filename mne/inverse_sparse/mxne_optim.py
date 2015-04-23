from __future__ import print_function
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Daniel Strohmeier <daniel.strohmeier@gmail.com>
#
# License: Simplified BSD

import warnings
from math import sqrt, ceil
import numpy as np
from scipy import linalg

from .mxne_debiasing import compute_bias
from ..utils import logger, verbose, sum_squared
from ..time_frequency.stft import stft_norm2, stft, istft
from ..externals.six.moves import xrange as range


def groups_norm2(A, n_orient):
    """compute squared L2 norms of groups inplace"""
    n_positions = A.shape[0] // n_orient
    return np.sum(np.power(A, 2, A).reshape(n_positions, -1), axis=1)


def norm_l2inf(A, n_orient, copy=True):
    """L2-inf norm"""
    if A.size == 0:
        return 0.0
    if copy:
        A = A.copy()
    return sqrt(np.max(groups_norm2(A, n_orient)))


def norm_l21(A, n_orient, copy=True):
    """L21 norm"""
    if A.size == 0:
        return 0.0
    if copy:
        A = A.copy()
    return np.sum(np.sqrt(groups_norm2(A, n_orient)))


def prox_l21(Y, alpha, n_orient, shape=None, is_stft=False):
    """proximity operator for l21 norm

    L2 over columns and L1 over rows => groups contain n_orient rows.

    It can eventually take into account the negative frequencies
    when a complex value is passed and is_stft=True.

    Example
    -------
    >>> Y = np.tile(np.array([0, 4, 3, 0, 0], dtype=np.float), (2, 1))
    >>> Y = np.r_[Y, np.zeros_like(Y)]
    >>> print(Y)
    [[ 0.  4.  3.  0.  0.]
     [ 0.  4.  3.  0.  0.]
     [ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]]
    >>> Yp, active_set = prox_l21(Y, 2, 2)
    >>> print(Yp)
    [[ 0.          2.86862915  2.15147186  0.          0.        ]
     [ 0.          2.86862915  2.15147186  0.          0.        ]]
    >>> print(active_set)
    [ True  True False False]
    """
    if len(Y) == 0:
        return np.zeros_like(Y), np.zeros((0,), dtype=np.bool)
    if shape is not None:
        shape_init = Y.shape
        Y = Y.reshape(*shape)
    n_positions = Y.shape[0] // n_orient

    if is_stft:
        rows_norm = np.sqrt(stft_norm2(Y).reshape(n_positions, -1).sum(axis=1))
    else:
        rows_norm = np.sqrt((Y * Y.conj()).real.reshape(n_positions,
                                                        -1).sum(axis=1))
    # Ensure shrink is >= 0 while avoiding any division by zero
    shrink = np.maximum(1.0 - alpha / np.maximum(rows_norm, alpha), 0.0)
    active_set = shrink > 0.0
    if n_orient > 1:
        active_set = np.tile(active_set[:, None], [1, n_orient]).ravel()
        shrink = np.tile(shrink[:, None], [1, n_orient]).ravel()
    Y = Y[active_set]
    if shape is None:
        Y *= shrink[active_set][:, np.newaxis]
    else:
        Y *= shrink[active_set][:, np.newaxis, np.newaxis]
        Y = Y.reshape(-1, *shape_init[1:])
    return Y, active_set


def prox_l1(Y, alpha, n_orient):
    """proximity operator for l1 norm with multiple orientation support

    L2 over orientation and L1 over position (space + time)

    Example
    -------
    >>> Y = np.tile(np.array([1, 2, 3, 2, 0], dtype=np.float), (2, 1))
    >>> Y = np.r_[Y, np.zeros_like(Y)]
    >>> print(Y)
    [[ 1.  2.  3.  2.  0.]
     [ 1.  2.  3.  2.  0.]
     [ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]]
    >>> Yp, active_set = prox_l1(Y, 2, 2)
    >>> print(Yp)
    [[ 0.          0.58578644  1.58578644  0.58578644  0.        ]
     [ 0.          0.58578644  1.58578644  0.58578644  0.        ]]
    >>> print(active_set)
    [ True  True False False]
    """
    n_positions = Y.shape[0] // n_orient
    norms = np.sqrt((Y * Y.conj()).real.T.reshape(-1, n_orient).sum(axis=1))
    # Ensure shrink is >= 0 while avoiding any division by zero
    shrink = np.maximum(1.0 - alpha / np.maximum(norms, alpha), 0.0)
    shrink = shrink.reshape(-1, n_positions).T
    active_set = np.any(shrink > 0.0, axis=1)
    shrink = shrink[active_set]
    if n_orient > 1:
        active_set = np.tile(active_set[:, None], [1, n_orient]).ravel()
    Y = Y[active_set]
    if len(Y) > 0:
        for o in range(n_orient):
            Y[o::n_orient] *= shrink
    return Y, active_set


def dgap_l21(M, G, X, active_set, alpha, n_orient):
    """Duality gaps for the mixed norm inverse problem

    For details see:
    Gramfort A., Kowalski M. and Hamalainen, M,
    Mixed-norm estimates for the M/EEG inverse problem using accelerated
    gradient methods, Physics in Medicine and Biology, 2012
    http://dx.doi.org/10.1088/0031-9155/57/7/1937

    Parameters
    ----------
    M : array, shape (n_sensors, n_times)
        The data.
    G : array, shape (n_sensors, n_active)
        The gain matrix a.k.a. lead field.
    X : array, shape (n_active, n_times)
        Sources
    active_set : array of bool
        Mask of active sources
    alpha : float
        Regularization parameter
    n_orient : int
        Number of dipoles per locations (typically 1 or 3)

    Returns
    -------
    gap : float
        Dual gap
    pobj : float
        Primal cost
    dobj : float
        Dual cost. gap = pobj - dobj
    R : array, shape (n_sensors, n_times)
        Current residual of M - G * X
    """
    GX = np.dot(G[:, active_set], X)
    R = M - GX
    penalty = norm_l21(X, n_orient, copy=True)
    nR2 = sum_squared(R)
    pobj = 0.5 * nR2 + alpha * penalty
    dual_norm = norm_l2inf(np.dot(G.T, R), n_orient, copy=False)
    scaling = alpha / dual_norm
    scaling = min(scaling, 1.0)
    dobj = 0.5 * (scaling ** 2) * nR2 + scaling * np.sum(R * GX)
    gap = pobj - dobj
    return gap, pobj, dobj, R


@verbose
def _mixed_norm_solver_prox(M, G, alpha, lipschitz_constant, maxit=200,
                            tol=1e-8, verbose=None, init=None, n_orient=1):
    """Solves L21 inverse problem with proximal iterations and FISTA"""
    n_sensors, n_times = M.shape
    n_sensors, n_sources = G.shape

    if n_sources < n_sensors:
        gram = np.dot(G.T, G)
        GTM = np.dot(G.T, M)
    else:
        gram = None

    if init is None:
        X = 0.0
        R = M.copy()
        if gram is not None:
            R = np.dot(G.T, R)
    else:
        X = init
        if gram is None:
            R = M - np.dot(G, X)
        else:
            R = GTM - np.dot(gram, X)

    t = 1.0
    Y = np.zeros((n_sources, n_times))  # FISTA aux variable
    E = []  # track cost function

    active_set = np.ones(n_sources, dtype=np.bool)  # start with full AS

    for i in range(maxit):
        X0, active_set_0 = X, active_set  # store previous values
        if gram is None:
            Y += np.dot(G.T, R) / lipschitz_constant  # ISTA step
        else:
            Y += R / lipschitz_constant  # ISTA step
        X, active_set = prox_l21(Y, alpha / lipschitz_constant, n_orient)

        t0 = t
        t = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t ** 2))
        Y.fill(0.0)
        dt = ((t0 - 1.0) / t)
        Y[active_set] = (1.0 + dt) * X
        Y[active_set_0] -= dt * X0
        Y_as = active_set_0 | active_set

        if gram is None:
            R = M - np.dot(G[:, Y_as], Y[Y_as])
        else:
            R = GTM - np.dot(gram[:, Y_as], Y[Y_as])

        gap, pobj, dobj, _ = dgap_l21(M, G, X, active_set, alpha, n_orient)
        E.append(pobj)
        logger.debug("pobj : %s -- gap : %s" % (pobj, gap))
        if gap < tol:
            logger.debug('Convergence reached ! (gap: %s < %s)' % (gap, tol))
            break
    return X, active_set, E


@verbose
def _mixed_norm_solver_cd(M, G, alpha, lipschitz_constant, maxit=10000,
                          tol=1e-8, verbose=None, init=None, n_orient=1):
    """Solves L21 inverse problem with coordinate descent"""
    from sklearn.linear_model.coordinate_descent import MultiTaskLasso

    n_sensors, n_times = M.shape
    n_sensors, n_sources = G.shape

    if init is not None:
        init = init.T

    clf = MultiTaskLasso(alpha=alpha / len(M), tol=tol, normalize=False,
                         fit_intercept=False, max_iter=maxit,
                         warm_start=True)
    clf.coef_ = init
    clf.fit(G, M)

    X = clf.coef_.T
    active_set = np.any(X, axis=1)
    X = X[active_set]
    gap, pobj, dobj, _ = dgap_l21(M, G, X, active_set, alpha, n_orient)
    return X, active_set, pobj


@verbose
def _mixed_norm_solver_bcd(M, G, alpha, lipschitz_constant, maxit=200,
                           tol=1e-8, verbose=None, init=None, n_orient=1):
    """Solves L21 inverse problem with block coordinate descent"""
    # First make G fortran for faster access to blocks of columns
    G = np.asfortranarray(G)

    n_sensors, n_times = M.shape
    n_sensors, n_sources = G.shape
    n_positions = n_sources // n_orient

    if init is None:
        X = np.zeros((n_sources, n_times))
        R = M.copy()
    else:
        X = init
        R = M - np.dot(G, X)

    E = []  # track cost function

    active_set = np.zeros(n_sources, dtype=np.bool)  # start with full AS

    alpha_lc = alpha / lipschitz_constant

    for i in range(maxit):
        for j in range(n_positions):
            idx = slice(j * n_orient, (j + 1) * n_orient)

            G_j = G[:, idx]
            X_j = X[idx]

            X_j_new = np.dot(G_j.T, R) / lipschitz_constant[j]

            was_non_zero = np.any(X_j)
            if was_non_zero:
                R += np.dot(G_j, X_j)
                X_j_new += X_j

            block_norm = linalg.norm(X_j_new, 'fro')
            if block_norm <= alpha_lc[j]:
                X_j.fill(0.)
                active_set[idx] = False
            else:
                shrink = np.maximum(1.0 - alpha_lc[j] / block_norm, 0.0)
                X_j_new *= shrink
                R -= np.dot(G_j, X_j_new)
                X_j[:] = X_j_new
                active_set[idx] = True

        gap, pobj, dobj, _ = dgap_l21(M, G, X[active_set], active_set, alpha,
                                      n_orient)
        E.append(pobj)
        logger.debug("Iteration %d :: pobj %f :: dgap %f :: n_active %d" % (
                     i + 1, pobj, gap, np.sum(active_set) / n_orient))

        if gap < tol:
            logger.debug('Convergence reached ! (gap: %s < %s)' % (gap, tol))
            break

    X = X[active_set]

    return X, active_set, E


@verbose
def mixed_norm_solver(M, G, alpha, maxit=3000, tol=1e-8, verbose=None,
                      active_set_size=50, debias=True, n_orient=1,
                      solver='auto'):
    """Solves L1/L2 mixed-norm inverse problem with active set strategy

    Algorithm is detailed in:
    Gramfort A., Kowalski M. and Hamalainen, M,
    Mixed-norm estimates for the M/EEG inverse problem using accelerated
    gradient methods, Physics in Medicine and Biology, 2012
    http://dx.doi.org/10.1088/0031-9155/57/7/1937

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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    active_set_size : int
        Size of active set increase at each iteration.
    debias : bool
        Debias source estimates.
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
    solver : 'prox' | 'cd' | 'bcd' | 'auto'
        The algorithm to use for the optimization.

    Returns
    -------
    X : array, shape (n_active, n_times)
        The source estimates.
    active_set : array
        The mask of active sources.
    E : list
        The value of the objective function over the iterations.
    """
    n_dipoles = G.shape[1]
    n_positions = n_dipoles // n_orient
    n_sensors, n_times = M.shape
    alpha_max = norm_l2inf(np.dot(G.T, M), n_orient, copy=False)
    logger.info("-- ALPHA MAX : %s" % alpha_max)
    alpha = float(alpha)

    has_sklearn = True
    try:
        from sklearn.linear_model.coordinate_descent import MultiTaskLasso  # noqa
    except ImportError:
        has_sklearn = False

    if solver == 'auto':
        if has_sklearn and (n_orient == 1):
            solver = 'cd'
        else:
            solver = 'bcd'

    if solver == 'cd':
        if n_orient == 1 and not has_sklearn:
            warnings.warn("Scikit-learn >= 0.12 cannot be found. "
                          "Using block coordinate descent instead of "
                          "coordinate descent.")
            solver = 'bcd'
        if n_orient > 1:
            warnings.warn("Coordinate descent is only available for fixed "
                          "orientation. Using block coordinate descent "
                          "instead of coordinate descent")
            solver = 'bcd'

    if solver == 'cd':
        logger.info("Using coordinate descent")
        l21_solver = _mixed_norm_solver_cd
        lc = None
    elif solver == 'bcd':
        logger.info("Using block coordinate descent")
        l21_solver = _mixed_norm_solver_bcd
        G = np.asfortranarray(G)
        if n_orient == 1:
            lc = np.sum(G * G, axis=0)
        else:
            lc = np.empty(n_positions)
            for j in range(n_positions):
                G_tmp = G[:, (j * n_orient):((j + 1) * n_orient)]
                lc[j] = linalg.norm(np.dot(G_tmp.T, G_tmp), ord=2)
    else:
        logger.info("Using proximal iterations")
        l21_solver = _mixed_norm_solver_prox
        lc = 1.01 * linalg.norm(G, ord=2) ** 2

    if active_set_size is not None:
        E = list()
        X_init = None
        active_set = np.zeros(n_dipoles, dtype=np.bool)
        idx_large_corr = np.argsort(groups_norm2(np.dot(G.T, M), n_orient))
        new_active_idx = idx_large_corr[-active_set_size:]
        if n_orient > 1:
            new_active_idx = (n_orient * new_active_idx[:, None] +
                              np.arange(n_orient)[None, :]).ravel()
        active_set[new_active_idx] = True
        as_size = np.sum(active_set)
        for k in range(maxit):
            if solver == 'bcd':
                lc_tmp = lc[active_set[::n_orient]]
            elif solver == 'cd':
                lc_tmp = None
            else:
                lc_tmp = 1.01 * linalg.norm(G[:, active_set], ord=2) ** 2
            X, as_, _ = l21_solver(M, G[:, active_set], alpha, lc_tmp,
                                   maxit=maxit, tol=tol, init=X_init,
                                   n_orient=n_orient)
            active_set[active_set] = as_.copy()
            idx_old_active_set = np.where(active_set)[0]

            gap, pobj, dobj, R = dgap_l21(M, G, X, active_set, alpha,
                                          n_orient)
            E.append(pobj)
            logger.info("Iteration %d :: pobj %f :: dgap %f ::"
                        "n_active_start %d :: n_active_end %d" % (
                            k + 1, pobj, gap, as_size // n_orient,
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
            logger.warning('Did NOT converge ! (gap: %s > %s)' % (gap, tol))
    else:
        X, active_set, E = l21_solver(M, G, alpha, lc, maxit=maxit,
                                      tol=tol, n_orient=n_orient, init=None)

    if np.any(active_set) and debias:
        bias = compute_bias(M, G[:, active_set], X, n_orient=n_orient)
        X *= bias[:, np.newaxis]

    logger.info('Final active set size: %s' % (np.sum(active_set) // n_orient))

    return X, active_set, E


@verbose
def iterative_mixed_norm_solver(M, G, alpha, n_mxne_iter, maxit=3000,
                                tol=1e-8, verbose=None, active_set_size=50,
                                debias=True, n_orient=1, solver='auto'):
    """Solves L0.5/L2 mixed-norm inverse problem with active set strategy

    Algorithm is detailed in:
    Strohmeier D., Haueisen J., and Gramfort A.:
    Improved MEG/EEG source localization with reweighted mixed-norms,
    4th International Workshop on Pattern Recognition in Neuroimaging,
    Tuebingen, 2014

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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    active_set_size : int
        Size of active set increase at each iteration.
    debias : bool
        Debias source estimates.
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
    solver : 'prox' | 'cd' | 'bcd' | 'auto'
        The algorithm to use for the optimization.

    Returns
    -------
    X : array, shape (n_active, n_times)
        The source estimates.
    active_set : array
        The mask of active sources.
    E : list
        The value of the objective function over the iterations.
    """
    def g(w):
        return np.sqrt(np.sqrt(groups_norm2(w.copy(), n_orient)))

    def gprime(w):
        return 2. * np.repeat(g(w), n_orient).ravel()

    E = list()

    active_set = np.ones(G.shape[1], dtype=np.bool)
    weights = np.ones(G.shape[1])
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
                    solver=solver, verbose=verbose)
            else:
                X, _active_set, _ = mixed_norm_solver(
                    M, G_tmp, alpha, debias=False, n_orient=n_orient,
                    maxit=maxit, tol=tol, active_set_size=None, solver=solver,
                    verbose=verbose)
        else:
            X, _active_set, _ = mixed_norm_solver(
                M, G_tmp, alpha, debias=False, n_orient=n_orient,
                maxit=maxit, tol=tol, active_set_size=None, solver=solver,
                verbose=verbose)

        logger.info('active set size %d' % (_active_set.sum() / n_orient))

        if _active_set.sum() > 0:
            active_set[active_set] = _active_set

            # Reapply weights to have correct unit
            X *= weights[_active_set][:, np.newaxis]
            weights = gprime(X)
            p_obj = 0.5 * linalg.norm(M - np.dot(G[:, active_set],  X),
                                      'fro') ** 2. + alpha * np.sum(g(X))
            E.append(p_obj)

            # Check convergence
            if ((k >= 1) and np.all(active_set == active_set_0) and
                    np.all(np.abs(X - X0) < tol)):
                print('Convergence reached after %d reweightings!' % k)
                break
        else:
            active_set = np.zeros_like(active_set)
            p_obj = 0.5 * linalg.norm(M) ** 2.
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
    """Compute lipschitz constant for FISTA

    It uses a power iteration method.
    """
    n_times = M.shape[1]
    n_points = G.shape[1]
    iv = np.ones((n_points, n_times), dtype=np.float)
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
    """Compute np.max(np.abs(A[ia])) possible with empty A"""
    if np.sum(ia):  # ia is not empty
        return np.max(np.abs(A[ia]))
    else:
        return 0.


def safe_max_abs_diff(A, ia, B, ib):
    """Compute np.max(np.abs(A)) possible with empty A"""
    A = A[ia] if np.sum(ia) else 0.0
    B = B[ib] if np.sum(ia) else 0.0
    return np.max(np.abs(A - B))


class _Phi(object):
    """Util class to have phi stft as callable without using
    a lambda that does not pickle"""
    def __init__(self, wsize, tstep, n_coefs):
        self.wsize = wsize
        self.tstep = tstep
        self.n_coefs = n_coefs

    def __call__(self, x):
        return stft(x, self.wsize, self.tstep,
                    verbose=False).reshape(-1, self.n_coefs)


class _PhiT(object):
    """Util class to have phi.T istft as callable without using
    a lambda that does not pickle"""
    def __init__(self, tstep, n_freq, n_step, n_times):
        self.tstep = tstep
        self.n_freq = n_freq
        self.n_step = n_step
        self.n_times = n_times

    def __call__(self, z):
        return istft(z.reshape(-1, self.n_freq, self.n_step), self.tstep,
                     self.n_times)


@verbose
def tf_mixed_norm_solver(M, G, alpha_space, alpha_time, wsize=64, tstep=4,
                         n_orient=1, maxit=200, tol=1e-8, log_objective=True,
                         lipschitz_constant=None, debias=True, verbose=None):
    """Solves TF L21+L1 inverse solver

    Algorithm is detailed in:

    A. Gramfort, D. Strohmeier, J. Haueisen, M. Hamalainen, M. Kowalski
    Time-Frequency Mixed-Norm Estimates: Sparse M/EEG imaging with
    non-stationary source activations
    Neuroimage, Volume 70, 15 April 2013, Pages 410-422, ISSN 1053-8119,
    DOI: 10.1016/j.neuroimage.2012.12.051.

    Functional Brain Imaging with M/EEG Using Structured Sparsity in
    Time-Frequency Dictionaries
    Gramfort A., Strohmeier D., Haueisen J., Hamalainen M. and Kowalski M.
    INFORMATION PROCESSING IN MEDICAL IMAGING
    Lecture Notes in Computer Science, 2011, Volume 6801/2011,
    600-611, DOI: 10.1007/978-3-642-22092-0_49
    http://dx.doi.org/10.1007/978-3-642-22092-0_49

    Parameters
    ----------
    M : array, shape (n_sensors, n_times)
        The data.
    G : array, shape (n_sensors, n_dipoles)
        The gain matrix a.k.a. lead field.
    alpha_space : float
        The spatial regularization parameter. It should be between 0 and 100.
    alpha_time : float
        The temporal regularization parameter. The higher it is the smoother
        will be the estimated time series.
    wsize: int
        length of the STFT window in samples (must be a multiple of 4).
    tstep: int
        step between successive windows in samples (must be a multiple of 2,
        a divider of wsize and smaller than wsize/2) (default: wsize/2).
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
    maxit : int
        The number of iterations.
    tol : float
        If absolute difference between estimates at 2 successive iterations
        is lower than tol, the convergence is reached.
    log_objective : bool
        If True, the value of the minimized objective function is computed
        and stored at every iteration.
    lipschitz_constant : float | None
        The lipschitz constant of the spatio temporal linear operator.
        If None it is estimated.
    debias : bool
        Debias source estimates.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    X : array, shape (n_active, n_times)
        The source estimates.
    active_set : array
        The mask of active sources.
    E : list
        The value of the objective function at each iteration. If log_objective
        is False, it will be empty.
    """
    n_sensors, n_times = M.shape
    n_dipoles = G.shape[1]

    n_step = int(ceil(n_times / float(tstep)))
    n_freq = wsize // 2 + 1
    n_coefs = n_step * n_freq
    phi = _Phi(wsize, tstep, n_coefs)
    phiT = _PhiT(tstep, n_freq, n_step, n_times)

    Z = np.zeros((0, n_coefs), dtype=np.complex)
    active_set = np.zeros(n_dipoles, dtype=np.bool)
    R = M.copy()  # residual

    if lipschitz_constant is None:
        lipschitz_constant = 1.1 * tf_lipschitz_constant(M, G, phi, phiT)

    logger.info("lipschitz_constant : %s" % lipschitz_constant)

    t = 1.0
    Y = np.zeros((n_dipoles, n_coefs), dtype=np.complex)  # FISTA aux variable
    Y[active_set] = Z
    E = []  # track cost function
    Y_time_as = None
    Y_as = None

    alpha_time_lc = alpha_time / lipschitz_constant
    alpha_space_lc = alpha_space / lipschitz_constant
    for i in range(maxit):
        Z0, active_set_0 = Z, active_set  # store previous values

        if active_set.sum() < len(R) and Y_time_as is not None:
            # trick when using tight frame to do a first screen based on
            # L21 prox (L21 norms are not changed by phi)
            GTR = np.dot(G.T, R) / lipschitz_constant
            A = GTR.copy()
            A[Y_as] += Y_time_as
            _, active_set_l21 = prox_l21(A, alpha_space_lc, n_orient)
            # just compute prox_l1 on rows that won't be zeroed by prox_l21
            B = Y[active_set_l21] + phi(GTR[active_set_l21])
            Z, active_set_l1 = prox_l1(B, alpha_time_lc, n_orient)
            active_set_l21[active_set_l21] = active_set_l1
            active_set_l1 = active_set_l21
        else:
            Y += np.dot(G.T, phi(R)) / lipschitz_constant  # ISTA step
            Z, active_set_l1 = prox_l1(Y, alpha_time_lc, n_orient)

        Z, active_set_l21 = prox_l21(Z, alpha_space_lc, n_orient,
                                     shape=(-1, n_freq, n_step), is_stft=True)
        active_set = active_set_l1
        active_set[active_set_l1] = active_set_l21

        # Check convergence : max(abs(Z - Z0)) < tol
        stop = (safe_max_abs(Z, ~active_set_0[active_set]) < tol and
                safe_max_abs(Z0, ~active_set[active_set_0]) < tol and
                safe_max_abs_diff(Z, active_set_0[active_set],
                                  Z0, active_set[active_set_0]) < tol)
        if stop:
            print('Convergence reached !')
            break

        # FISTA 2 steps
        # compute efficiently : Y = Z + ((t0 - 1.0) / t) * (Z - Z0)
        t0 = t
        t = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t ** 2))
        Y.fill(0.0)
        dt = ((t0 - 1.0) / t)
        Y[active_set] = (1.0 + dt) * Z
        if len(Z0):
            Y[active_set_0] -= dt * Z0
        Y_as = active_set_0 | active_set

        Y_time_as = phiT(Y[Y_as])
        R = M - np.dot(G[:, Y_as], Y_time_as)

        if log_objective:  # log cost function value
            Z2 = np.abs(Z)
            Z2 **= 2
            X = phiT(Z)
            RZ = M - np.dot(G[:, active_set], X)
            pobj = (0.5 * linalg.norm(RZ, ord='fro') ** 2 +
                    alpha_space * norm_l21(X, n_orient) +
                    alpha_time * np.sqrt(np.sum(Z2.T.reshape(-1, n_orient),
                                                axis=1)).sum())
            E.append(pobj)
            logger.info("Iteration %d :: pobj %f :: n_active %d" % (i + 1,
                        pobj, np.sum(active_set)))
        else:
            logger.info("Iteration %d" % i + 1)

    X = phiT(Z)

    if np.any(active_set) and debias:
        bias = compute_bias(M, G[:, active_set], X, n_orient=n_orient)
        X *= bias[:, np.newaxis]

    return X, active_set, E
