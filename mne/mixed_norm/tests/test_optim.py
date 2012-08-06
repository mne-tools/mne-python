# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

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
    X_hat, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8, verbose=True,
                            active_set_size=None, debias=True)
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8, verbose=True,
                            active_set_size=1, debias=True)
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8, verbose=True,
                            active_set_size=1, debias=True, n_orient=2)
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    X_hat, active_set, _ = mixed_norm_solver(M,
                            G, alpha, maxit=1000, tol=1e-8, verbose=True,
                            active_set_size=1, debias=True, n_orient=5)
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
