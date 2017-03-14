from __future__ import print_function
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Daniel Strohmeier <daniel.strohmeier@gmail.com>
#
# License: Simplified BSD

from copy import deepcopy
from math import sqrt, ceil

import numpy as np
from scipy import linalg

from .mxne_debiasing import compute_bias
from ..utils import logger, verbose, sum_squared, warn
from ..time_frequency.stft import stft_norm2, stft, istft
from ..externals.six.moves import xrange as range


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


def prox_l21(Y, alpha, n_orient, shape=None, is_stft=False):
    """Proximity operator for l21 norm.

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
    """Proximity operator for l1 norm with multiple orientation support.

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
    """Duality gaps for the mixed norm inverse problem.

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
    """Solve L21 inverse problem with proximal iterations and FISTA."""
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
    """Solve L21 inverse problem with coordinate descent."""
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
    """Solve L21 inverse problem with block coordinate descent."""
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
    """Solve L1/L2 mixed-norm inverse problem with active set strategy.

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
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
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
        from sklearn.linear_model.coordinate_descent import MultiTaskLasso  # noqa: F401,E501
    except ImportError:
        has_sklearn = False

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
            warn('Did NOT converge ! (gap: %s > %s)' % (gap, tol))
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
    """Solve L0.5/L2 mixed-norm inverse problem with active set strategy.

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
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
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
    """Compute lipschitz constant for FISTA.

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

    def __init__(self, wsize, tstep, n_coefs):  # noqa: D102
        self.wsize = wsize
        self.tstep = tstep
        self.n_coefs = n_coefs

    def __call__(self, x):  # noqa: D105
        return stft(x, self.wsize, self.tstep,
                    verbose=False).reshape(-1, self.n_coefs)


class _PhiT(object):
    """Have phi.T istft as callable w/o using a lambda that does not pickle."""

    def __init__(self, tstep, n_freq, n_step, n_times):  # noqa: D102
        self.tstep = tstep
        self.n_freq = n_freq
        self.n_step = n_step
        self.n_times = n_times

    def __call__(self, z):  # noqa: D105
        return istft(z.reshape(-1, self.n_freq, self.n_step), self.tstep,
                     self.n_times)


def norm_l21_tf(Z, shape, n_orient):
    """L21 norm for TF."""
    if Z.shape[0]:
        Z2 = Z.reshape(*shape)
        l21_norm = np.sqrt(stft_norm2(Z2).reshape(-1, n_orient).sum(axis=1))
        l21_norm = l21_norm.sum()
    else:
        l21_norm = 0.
    return l21_norm


def norm_l1_tf(Z, shape, n_orient):
    """L1 norm for TF."""
    if Z.shape[0]:
        n_positions = Z.shape[0] // n_orient
        Z_ = np.sqrt(np.sum((np.abs(Z) ** 2.).reshape((n_orient, -1),
                     order='F'), axis=0))
        Z_ = Z_.reshape((n_positions, -1), order='F').reshape(*shape)
        l1_norm = (2. * Z_.sum(axis=2).sum(axis=1) - np.sum(Z_[:, 0, :],
                   axis=1) - np.sum(Z_[:, -1, :], axis=1))
        l1_norm = l1_norm.sum()
    else:
        l1_norm = 0.
    return l1_norm


@verbose
def _tf_mixed_norm_solver_bcd_(M, G, Z, active_set, alpha_space, alpha_time,
                               lipschitz_constant, phi, phiT,
                               wsize=64, tstep=4, n_orient=1,
                               maxit=200, tol=1e-8, log_objective=True,
                               perc=None, verbose=None):
    # First make G fortran for faster access to blocks of columns
    G = np.asfortranarray(G)

    n_sensors, n_times = M.shape
    n_sources = G.shape[1]
    n_positions = n_sources // n_orient

    n_step = int(ceil(n_times / float(tstep)))
    n_freq = wsize // 2 + 1
    shape = (-1, n_freq, n_step)

    G = dict(zip(np.arange(n_positions), np.hsplit(G, n_positions)))
    R = M.copy()  # residual
    active = np.where(active_set)[0][::n_orient] // n_orient
    for idx in active:
        R -= np.dot(G[idx], phiT(Z[idx]))

    E = []  # track cost function

    alpha_time_lc = alpha_time / lipschitz_constant
    alpha_space_lc = alpha_space / lipschitz_constant

    converged = False

    for i in range(maxit):
        val_norm_l21_tf = 0.0
        val_norm_l1_tf = 0.0
        max_diff = 0.0
        active_set_0 = active_set.copy()
        for j in range(n_positions):
            ids = j * n_orient
            ide = ids + n_orient

            G_j = G[j]
            Z_j = Z[j]
            active_set_j = active_set[ids:ide]

            Z0 = deepcopy(Z_j)

            was_active = np.any(active_set_j)

            # gradient step
            GTR = np.dot(G_j.T, R) / lipschitz_constant[j]
            X_j_new = GTR.copy()

            if was_active:
                X_j = phiT(Z_j)
                R += np.dot(G_j, X_j)
                X_j_new += X_j

            rows_norm = linalg.norm(X_j_new, 'fro')
            if rows_norm <= alpha_space_lc[j]:
                if was_active:
                    Z[j] = 0.0
                    active_set_j[:] = False
            else:
                if was_active:
                    Z_j_new = Z_j + phi(GTR)
                else:
                    Z_j_new = phi(GTR)

                col_norm = np.sqrt(np.sum(np.abs(Z_j_new) ** 2, axis=0))

                if np.all(col_norm <= alpha_time_lc[j]):
                    Z[j] = 0.0
                    active_set_j[:] = False
                else:
                    # l1
                    shrink = np.maximum(1.0 - alpha_time_lc[j] / np.maximum(
                                        col_norm, alpha_time_lc[j]), 0.0)
                    Z_j_new *= shrink[np.newaxis, :]

                    # l21
                    shape_init = Z_j_new.shape
                    Z_j_new = Z_j_new.reshape(*shape)
                    row_norm = np.sqrt(stft_norm2(Z_j_new).sum())
                    if row_norm <= alpha_space_lc[j]:
                        Z[j] = 0.0
                        active_set_j[:] = False
                    else:
                        shrink = np.maximum(1.0 - alpha_space_lc[j] /
                                            np.maximum(row_norm,
                                            alpha_space_lc[j]), 0.0)
                        Z_j_new *= shrink
                        Z[j] = Z_j_new.reshape(-1, *shape_init[1:]).copy()
                        active_set_j[:] = True
                        R -= np.dot(G_j, phiT(Z[j]))

                        if log_objective:
                            val_norm_l21_tf += norm_l21_tf(
                                Z[j], shape, n_orient)
                            val_norm_l1_tf += norm_l1_tf(
                                Z[j], shape, n_orient)

            max_diff = np.maximum(max_diff, np.max(np.abs(Z[j] - Z0)))

        if log_objective:  # log cost function value
            pobj = (0.5 * (R ** 2.).sum() + alpha_space * val_norm_l21_tf +
                    alpha_time * val_norm_l1_tf)
            E.append(pobj)
            logger.info("Iteration %d :: pobj %f :: n_active %d" % (i + 1,
                        pobj, np.sum(active_set) / n_orient))
        else:
            logger.info("Iteration %d" % (i + 1))

        if perc is not None:
            if np.sum(active_set) / float(n_orient) <= perc * n_positions:
                break

        if np.array_equal(active_set, active_set_0):
            if max_diff < tol:
                logger.info("Convergence reached !")
                converged = True
                break

    return Z, active_set, E, converged


@verbose
def _tf_mixed_norm_solver_bcd_active_set(
        M, G, alpha_space, alpha_time, lipschitz_constant, phi, phiT,
        Z_init=None, wsize=64, tstep=4, n_orient=1, maxit=200, tol=1e-8,
        log_objective=True, perc=None, verbose=None):
    """Solve TF L21+L1 inverse solver with BCD and active set approach.

    Algorithm is detailed in:

    Strohmeier D., Gramfort A., and Haueisen J.:
    MEG/EEG source imaging with a non-convex penalty in the time-
    frequency domain,
    5th International Workshop on Pattern Recognition in Neuroimaging,
    Stanford University, 2015

    Parameters
    ----------
    M : array
        The data.
    G : array
        The forward operator.
    alpha_space : float in [0, 100]
        Regularization parameter for spatial sparsity. If larger than 100,
        then no source will be active.
    alpha_time : float in [0, 100]
        Regularization parameter for temporal sparsity. It set to 0,
        no temporal regularization is applied. It this case, TF-MxNE is
        equivalent to MxNE with L21 norm.
    lipschitz_constant : float
        The lipschitz constant of the spatio temporal linear operator.
    phi : instance of _Phi
        The TF operator.
    phiT : instance of _PhiT
        The transpose of the TF operator.
    Z_init : None | array
        The initialization of the TF coefficient matrix. If None, zeros
        will be used for all coefficients.
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
    perc : None | float in [0, 1]
        The early stopping parameter used for BCD with active set approach.
        If the active set size is smaller than perc * n_sources, the
        subproblem limited to the active set is stopped. If None, full
        convergence will be achieved.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    X : array
        The source estimates.
    active_set : array
        The mask of active sources.
    E : list
        The value of the objective function at each iteration. If log_objective
        is False, it will be empty.
    """
    n_sources = G.shape[1]
    n_positions = n_sources // n_orient

    if Z_init is None:
        Z = dict.fromkeys(range(n_positions), 0.0)
        active_set = np.zeros(n_sources, dtype=np.bool)
    else:
        active_set = np.zeros(n_sources, dtype=np.bool)
        active = list()
        for i in range(n_positions):
            if np.any(Z_init[i * n_orient:(i + 1) * n_orient]):
                active_set[i * n_orient:(i + 1) * n_orient] = True
                active.append(i)
        Z = dict.fromkeys(range(n_positions), 0.0)
        if len(active):
            Z.update(dict(zip(active, np.vsplit(Z_init[active_set],
                     len(active)))))

    Z, active_set, E, _ = _tf_mixed_norm_solver_bcd_(
        M, G, Z, active_set, alpha_space, alpha_time, lipschitz_constant,
        phi, phiT, wsize=wsize, tstep=tstep, n_orient=n_orient, maxit=1,
        tol=tol, log_objective=log_objective, perc=None, verbose=verbose)

    while active_set.sum():
        active = np.where(active_set)[0][::n_orient] // n_orient
        Z_init = dict(zip(range(len(active)), [Z[idx] for idx in active]))
        Z, as_, E_tmp, converged = _tf_mixed_norm_solver_bcd_(
            M, G[:, active_set], Z_init,
            np.ones(len(active) * n_orient, dtype=np.bool),
            alpha_space, alpha_time,
            lipschitz_constant[active_set[::n_orient]],
            phi, phiT, wsize=wsize, tstep=tstep, n_orient=n_orient,
            maxit=maxit, tol=tol, log_objective=log_objective,
            perc=0.5, verbose=verbose)
        E += E_tmp
        active = np.where(active_set)[0][::n_orient] // n_orient
        Z_init = dict.fromkeys(range(n_positions), 0.0)
        Z_init.update(dict(zip(active, Z.values())))
        active_set[active_set] = as_
        active_set_0 = active_set.copy()
        Z, active_set, E_tmp, _ = _tf_mixed_norm_solver_bcd_(
            M, G, Z_init, active_set, alpha_space, alpha_time,
            lipschitz_constant, phi, phiT, wsize=wsize, tstep=tstep,
            n_orient=n_orient, maxit=1, tol=tol, log_objective=log_objective,
            perc=None, verbose=verbose)
        E += E_tmp
        if converged:
            if np.array_equal(active_set_0, active_set):
                break

    if active_set.sum():
        Z = np.vstack([Z_ for Z_ in list(Z.values()) if np.any(Z_)])
        X = phiT(Z)
    else:
        n_sensors, n_times = M.shape
        n_step = int(ceil(n_times / float(tstep)))
        n_freq = wsize // 2 + 1
        Z = np.zeros((0, n_step * n_freq), dtype=np.complex)
        X = np.zeros((0, n_times))

    return X, Z, active_set, E


@verbose
def tf_mixed_norm_solver(M, G, alpha_space, alpha_time, wsize=64, tstep=4,
                         n_orient=1, maxit=200, tol=1e-8, log_objective=True,
                         debias=True, verbose=None):
    """Solve TF L21+L1 inverse solver with BCD and active set approach.

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
    debias : bool
        Debias source estimates.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

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
    n_sensors, n_sources = G.shape
    n_positions = n_sources // n_orient

    n_step = int(ceil(n_times / float(tstep)))
    n_freq = wsize // 2 + 1
    n_coefs = n_step * n_freq
    phi = _Phi(wsize, tstep, n_coefs)
    phiT = _PhiT(tstep, n_freq, n_step, n_times)

    if n_orient == 1:
        lc = np.sum(G * G, axis=0)
    else:
        lc = np.empty(n_positions)
        for j in range(n_positions):
            G_tmp = G[:, (j * n_orient):((j + 1) * n_orient)]
            lc[j] = linalg.norm(np.dot(G_tmp.T, G_tmp), ord=2)

    logger.info("Using block coordinate descent and active set approach")
    X, Z, active_set, E = _tf_mixed_norm_solver_bcd_active_set(
        M, G, alpha_space, alpha_time, lc, phi, phiT, Z_init=None,
        wsize=wsize, tstep=tstep, n_orient=n_orient, maxit=maxit, tol=tol,
        log_objective=log_objective, verbose=None)

    if np.any(active_set) and debias:
        bias = compute_bias(M, G[:, active_set], X, n_orient=n_orient)
        X *= bias[:, np.newaxis]

    return X, active_set, E
