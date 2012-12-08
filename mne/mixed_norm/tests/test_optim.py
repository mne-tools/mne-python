# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import time

import numpy as np
from numpy.testing import assert_array_equal

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
    X_hat_FISTA, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=None, debias=True,
                            solver='FISTA')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat_CD, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=None, debias=True,
                            solver='CD')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    assert_array_equal(np.around(X_hat_FISTA, decimals=3),
                       np.around(X_hat_CD, decimals=3))

    X_hat, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=2, debias=True,
                            solver='FISTA')
    X_hat, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=2, debias=True,
                            solver='CD')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    assert_array_equal(np.around(X_hat_FISTA, decimals=3),
                       np.around(X_hat_CD, decimals=3))

    X_hat, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=1, debias=True, n_orient=2)
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    X_hat, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=1, debias=True, n_orient=5)
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    assert_array_equal(np.around(X_hat_FISTA, decimals=3),
                       np.around(X_hat_CD, decimals=3))
