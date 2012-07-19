# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

from math import sqrt
import numpy as np
from scipy import linalg

from .debiasing import compute_bias


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


def prox_l21(Y, alpha, n_orient):
    """proximity operator for l21 norm

    (L2 over columns and L1 over rows => groups contain n_orient rows)

    Example
    -------
    >>> Y = np.tile(np.array([0, 4, 3, 0, 0], dtype=np.float), (2, 1))
    >>> Y = np.r_[Y, np.zeros_like(Y)]
    >>> print Y
    [[ 0.  4.  3.  0.  0.]
     [ 0.  4.  3.  0.  0.]
     [ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]]
    >>> Yp, active_set = prox_l21(Y, 2, 2)
    >>> print Yp
    [[ 0.          2.86862915  2.15147186  0.          0.        ]
     [ 0.          2.86862915  2.15147186  0.          0.        ]]
    >>> print active_set
    [ True  True False False]
    """
    if len(Y) == 0:
        return np.zeros((0, Y.shape[1]), dtype=Y.dtype), \
                    np.zeros((0,), dtype=np.bool)
    n_positions = Y.shape[0] // n_orient
    rows_norm = np.sqrt(np.sum((np.abs(Y) ** 2).reshape(n_positions, -1),
                                axis=1))
    shrink = np.maximum(1.0 - alpha / rows_norm, 0.0)
    active_set = shrink > 0.0
    if n_orient > 1:
        active_set = np.tile(active_set[:, None], [1, n_orient]).ravel()
        shrink = np.tile(shrink[:, None], [1, n_orient]).ravel()
    Y = Y[active_set]
    Y *= shrink[active_set][:, np.newaxis]
    return Y, active_set


def prox_l1(Y, alpha, n_orient):
    """proximity operator for l1 norm with multiple orientation support

    L2 over orientation and L1 over position (space + time)

    Example
    -------
    >>> Y = np.tile(np.array([1, 2, 3, 2, 0], dtype=np.float), (2, 1))
    >>> Y = np.r_[Y, np.zeros_like(Y)]
    >>> print Y
    [[ 1.  2.  3.  2.  0.]
     [ 1.  2.  3.  2.  0.]
     [ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]]
    >>> Yp, active_set = prox_l1(Y, 2, 2)
    >>> print Yp
    [[ 0.          0.58578644  1.58578644  0.58578644  0.        ]
     [ 0.          0.58578644  1.58578644  0.58578644  0.        ]]
    >>> print active_set
    [ True  True False False]
    """
    n_positions = Y.shape[0] // n_orient
    norms = np.sqrt(np.sum((np.abs(Y) ** 2).T.reshape(-1, n_orient), axis=1))
    shrink = np.maximum(1.0 - alpha / norms, 0.0).reshape(-1, n_positions).T
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
    M : array of shape [n_sensors, n_times]
        data
    G : array of shape [n_sensors, n_active]
        Gain matrix a.k.a. lead field
    X : array of shape [n_active, n_times]
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
    R : array of shape [n_sensors, n_times]
        Current residual of M - G * X
    """
    GX = np.dot(G[:, active_set], X)
    R = M - GX
    penalty = norm_l21(X, n_orient, copy=True)
    nR2 = np.sum(R ** 2)
    pobj = 0.5 * nR2 + alpha * penalty
    dual_norm = norm_l2inf(np.dot(G.T, R), n_orient, copy=False)
    scaling = alpha / dual_norm
    scaling = min(scaling, 1.0)
    dobj = 0.5 * (scaling ** 2) * nR2 + scaling * np.sum(R * GX)
    gap = pobj - dobj
    return gap, pobj, dobj, R


def _mixed_norm_solver(M, G, alpha, maxit=200, tol=1e-8, verbose=True,
                       init=None, n_orient=1):
    """Solves L21 inverse solver"""
    n_sensors, n_times = M.shape
    n_sensors, n_sources = G.shape

    lipschitz_constant = 1.1 * linalg.norm(G, ord=2) ** 2

    if n_sources < n_sensors:
        gram = np.dot(G.T, G)
        GTM = np.dot(G.T, M)
    else:
        gram = None

    if init is None:
        X = 0.0
        active_set = np.zeros(n_sources, dtype=np.bool)
        R = M.copy()
        if gram is not None:
            R = np.dot(G.T, R)
    else:
        X, active_set = init
        if gram is None:
            R = M - np.dot(G[:, active_set], X)
        else:
            R = GTM - np.dot(gram[:, active_set], X)

    t = 1.0
    Y = np.zeros((n_sources, n_times))  # FISTA aux variable
    E = []  # track cost function

    active_set = np.ones(n_sources, dtype=np.bool)  # HACK

    for i in xrange(maxit):
        X0, active_set_0 = X, active_set  # store previous values
        if gram is None:
            Y += np.dot(G.T, R) / lipschitz_constant  # ISTA step
        else:
            Y += R / lipschitz_constant  # ISTA step
        X, active_set = prox_l21(Y, alpha / lipschitz_constant, n_orient)

        t0 = t
        t = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t ** 2))
        Y[:] = 0.0
        dt = ((t0 - 1.0) / t)
        Y[active_set] = (1.0 + dt) * X
        Y[active_set_0] -= dt * X0
        Y_as = active_set_0 | active_set

        if gram is None:
            R = M - np.dot(G[:, Y_as], Y[Y_as])
        else:
            R = GTM - np.dot(gram[:, Y_as], Y[Y_as])

        if verbose:  # log cost function value
            gap, pobj, dobj, _ = dgap_l21(M, G, X, active_set, alpha, n_orient)
            E.append(pobj)
            print "pobj : %s -- gap : %s" % (pobj, gap)
            if gap < tol:
                print 'Convergence reached ! (gap: %s < %s)' % (gap, tol)
                break
    return X, active_set, E


def mixed_norm_solver(M, G, alpha, maxit=200, tol=1e-8, verbose=True,
                      active_set_size=50, debias=True, n_orient=1):
    """Solves L21 inverse solver with active set strategy

    Algorithm is detailed in:
    Gramfort A., Kowalski M. and Hamalainen, M,
    Mixed-norm estimates for the M/EEG inverse problem using accelerated
    gradient methods, Physics in Medicine and Biology, 2012
    http://dx.doi.org/10.1088/0031-9155/57/7/1937

    Parameters
    ----------
    M : array
        The data
    G : array
        The forward operator
    maxit : int
        The number of iterations
    tol : float
        Tolerance on dual gap for convergence checking
    verbose : bool
        Use verbose output
    active_set_size : int
        Size of active set increase at each iteration.
    debias : bool
        Debias source estimates
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).

    Returns
    -------
    X: array
        The source estimates
    active_set: array
        The mask of active sources
    E: array
        The cost function over the iterations
    """
    n_dipoles = G.shape[1]
    n_positions = n_dipoles // n_orient
    alpha_max = norm_l2inf(np.dot(G.T, M), n_orient, copy=False)
    print "-- ALPHA MAX : %s" % alpha_max
    if active_set_size is not None:
        n_sensors, n_times = M.shape
        idx_large_corr = np.argsort(groups_norm2(np.dot(G.T, M), n_orient))
        active_set = np.zeros(n_positions, dtype=np.bool)
        active_set[idx_large_corr[-active_set_size:]] = True
        if n_orient > 1:
            active_set = np.tile(active_set[:, None], [1, n_orient]).ravel()
        init = None
        for k in xrange(maxit):
            X, as_, E = _mixed_norm_solver(M, G[:, active_set], alpha,
                                           maxit=maxit, tol=tol, verbose=False,
                                           init=init, n_orient=n_orient)
            as_ = np.where(active_set)[0][as_]
            gap, pobj, dobj, R = dgap_l21(M, G, X, as_, alpha, n_orient)
            print 'gap = %s, pobj = %s' % (gap, pobj)
            if gap < tol:
                print 'Convergence reached ! (gap: %s < %s)' % (gap, tol)
                break
            else:  # add sources
                idx_large_corr = np.argsort(groups_norm2(np.dot(G.T, R),
                                                         n_orient))
                new_active_idx = idx_large_corr[-active_set_size:]
                if n_orient > 1:
                    new_active_idx = n_orient * new_active_idx[:, None] + \
                                                np.arange(n_orient)[None, :]
                    new_active_idx = new_active_idx.ravel()
                idx_old_active_set = as_
                active_set_old = active_set.copy()
                active_set[new_active_idx] = True
                as_size = np.sum(active_set)
                print 'active set size %s' % as_size
                X_init = np.zeros((as_size, n_times), dtype=X.dtype)
                idx_active_set = np.where(active_set)[0]
                idx = np.searchsorted(idx_active_set, idx_old_active_set)
                X_init[idx] = X
                init = X_init, active_set[active_set == True]
                if np.all(active_set_old == active_set):
                    print 'Convergence stopped (AS did not change) !'
                    break
        else:
            print 'Did NOT converge ! (gap: %s > %s)' % (gap, tol)

        active_set = np.zeros_like(active_set)
        active_set[as_] = True
    else:
        X, active_set, E = _mixed_norm_solver(M, G, alpha, maxit=maxit,
                                              tol=tol, verbose=verbose,
                                              n_orient=n_orient)

    if (active_set.sum() > 0) and debias:
        bias = compute_bias(M, G[:, active_set], X, n_orient=n_orient)
        X *= bias[:, np.newaxis]

    return X, active_set, E
