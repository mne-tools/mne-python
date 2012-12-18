# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import numpy as np
import warnings
from numpy.testing import assert_array_equal, assert_array_almost_equal

from mne.mixed_norm.optim import mixed_norm_solver


def test_l21_MxNE():
    """Test convergence of MxNE"""
    n, p, t, alpha = 30, 40, 20, 1
    rng = np.random.RandomState(0)
    G = rng.randn(n, p)
    G /= np.std(G, axis=0)[None, :]
    X = np.zeros((p, t))
    X[0] = 3
    X[4] = -2
    M = np.dot(G, X)

    # suppress a coordinate-descent warning here
    X_hat_prox, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=None, debias=True,
                            solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat_cd, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=None, debias=True,
                            solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    assert_array_almost_equal(X_hat_prox, X_hat_cd, 5)

    X_hat_prox, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=2, debias=True,
                            solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat_cd, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=2, debias=True,
                            solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    assert_array_almost_equal(X_hat_prox, X_hat_cd, 5)

    X_hat_prox, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=2, debias=True,
                            n_orient=2, solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    with warnings.catch_warnings(True) as w:
        X_hat_cd, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=2, debias=True,
                            n_orient=2, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    assert_array_equal(X_hat_prox, X_hat_cd)

    X_hat_prox, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=2, debias=True,
                            n_orient=5)
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    X_hat_cd, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=2, debias=True,
                            n_orient=5, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    assert_array_equal(X_hat_prox, X_hat_cd)
