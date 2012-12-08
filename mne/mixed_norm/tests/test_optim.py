# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import time

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_true

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

    tic_FISTA = time.time()
    X_hat_FISTA, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=None, debias=True,
                            solver='FISTA')
    toc_FISTA = time.time()
    assert_array_equal(np.where(active_set)[0], [0, 4])
    tic_CD = time.time()
    X_hat_CD, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=None, debias=True,
                            solver='CD')
    toc_CD = time.time()
    assert_array_equal(np.where(active_set)[0], [0, 4])
    assert_array_almost_equal(X_hat_FISTA, X_hat_CD, 5)
    assert_true(toc_FISTA - tic_FISTA > toc_CD - tic_CD)

    tic_FISTA = time.time()
    X_hat_FISTA, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=2, debias=True,
                            solver='FISTA')
    toc_FISTA = time.time()
    assert_array_equal(np.where(active_set)[0], [0, 4])
    tic_CD = time.time()
    X_hat_CD, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=2, debias=True,
                            solver='CD')
    toc_CD = time.time()
    assert_array_equal(np.where(active_set)[0], [0, 4])
    assert_array_almost_equal(X_hat_FISTA, X_hat_CD, 5)
    assert_true(toc_FISTA - tic_FISTA > toc_CD - tic_CD)

    X_hat_FISTA, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=2, debias=True,
                            n_orient=2, solver='FISTA')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    X_hat_CD, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=2, debias=True,
                            n_orient=2, solver='CD')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    assert_array_equal(X_hat_FISTA, X_hat_CD)

    X_hat_FISTA, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=2, debias=True,
                            n_orient=5)
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    X_hat_CD, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8,
                            active_set_size=2, debias=True,
                            n_orient=5, solver='CD')
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    assert_array_equal(X_hat_FISTA, X_hat_CD)
