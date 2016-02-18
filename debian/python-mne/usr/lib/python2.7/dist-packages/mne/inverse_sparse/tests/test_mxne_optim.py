# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Daniel Strohmeier <daniel.strohmeier@gmail.com>
#
# License: Simplified BSD

import numpy as np
import warnings
from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.testing import assert_allclose

from mne.inverse_sparse.mxne_optim import (mixed_norm_solver,
                                           tf_mixed_norm_solver,
                                           iterative_mixed_norm_solver)

warnings.simplefilter('always')  # enable b/c these tests throw warnings


def _generate_tf_data():
    n, p, t = 30, 40, 64
    rng = np.random.RandomState(0)
    G = rng.randn(n, p)
    G /= np.std(G, axis=0)[None, :]
    X = np.zeros((p, t))
    active_set = [0, 4]
    times = np.linspace(0, 2 * np.pi, t)
    X[0] = np.sin(times)
    X[4] = -2 * np.sin(4 * times)
    X[4, times <= np.pi / 2] = 0
    X[4, times >= np.pi] = 0
    M = np.dot(G, X)
    M += 1 * rng.randn(*M.shape)
    return M, G, active_set


def test_l21_mxne():
    """Test convergence of MxNE solver"""
    n, p, t, alpha = 30, 40, 20, 1.
    rng = np.random.RandomState(0)
    G = rng.randn(n, p)
    G /= np.std(G, axis=0)[None, :]
    X = np.zeros((p, t))
    X[0] = 3
    X[4] = -2
    M = np.dot(G, X)

    args = (M, G, alpha, 1000, 1e-8)
    X_hat_prox, active_set, _ = mixed_norm_solver(
        *args, active_set_size=None,
        debias=True, solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat_cd, active_set, _ = mixed_norm_solver(
        *args, active_set_size=None,
        debias=True, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat_bcd, active_set, _ = mixed_norm_solver(
        M, G, alpha, maxit=1000, tol=1e-8, active_set_size=None,
        debias=True, solver='bcd')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    assert_allclose(X_hat_prox, X_hat_cd, rtol=1e-2)
    assert_allclose(X_hat_prox, X_hat_bcd, rtol=1e-2)
    assert_allclose(X_hat_bcd, X_hat_cd, rtol=1e-2)

    X_hat_prox, active_set, _ = mixed_norm_solver(
        *args, active_set_size=2, debias=True, solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat_cd, active_set, _ = mixed_norm_solver(
        *args, active_set_size=2, debias=True, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat_bcd, active_set, _ = mixed_norm_solver(
        *args, active_set_size=2, debias=True, solver='bcd')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    assert_allclose(X_hat_bcd, X_hat_cd, rtol=1e-2)
    assert_allclose(X_hat_bcd, X_hat_prox, rtol=1e-2)

    X_hat_prox, active_set, _ = mixed_norm_solver(
        *args, active_set_size=2, debias=True, n_orient=2, solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    X_hat_bcd, active_set, _ = mixed_norm_solver(
        *args, active_set_size=2, debias=True, n_orient=2, solver='bcd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])

    # suppress a coordinate-descent warning here
    with warnings.catch_warnings(record=True):
        X_hat_cd, active_set, _ = mixed_norm_solver(
            *args, active_set_size=2, debias=True, n_orient=2, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    assert_allclose(X_hat_bcd, X_hat_prox, rtol=1e-2)
    assert_allclose(X_hat_bcd, X_hat_cd, rtol=1e-2)

    X_hat_bcd, active_set, _ = mixed_norm_solver(
        *args, active_set_size=2, debias=True, n_orient=5, solver='bcd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    X_hat_prox, active_set, _ = mixed_norm_solver(
        *args, active_set_size=2, debias=True, n_orient=5, solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    with warnings.catch_warnings(record=True):  # coordinate-ascent warning
        X_hat_cd, active_set, _ = mixed_norm_solver(
            *args, active_set_size=2, debias=True, n_orient=5, solver='cd')

    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    assert_array_equal(X_hat_bcd, X_hat_cd)
    assert_allclose(X_hat_bcd, X_hat_prox, rtol=1e-2)


def test_tf_mxne():
    """Test convergence of TF-MxNE solver"""
    alpha_space = 10.
    alpha_time = 5.

    M, G, active_set = _generate_tf_data()

    X_hat_tf, active_set_hat_tf, E = tf_mixed_norm_solver(
        M, G, alpha_space, alpha_time, maxit=200, tol=1e-8, verbose=True,
        n_orient=1, tstep=4, wsize=32)
    assert_array_equal(np.where(active_set_hat_tf)[0], active_set)


def test_tf_mxne_vs_mxne():
    """Test equivalence of TF-MxNE (with alpha_time=0) and MxNE"""
    alpha_space = 60.
    alpha_time = 0.

    M, G, active_set = _generate_tf_data()

    X_hat_tf, active_set_hat_tf, E = tf_mixed_norm_solver(
        M, G, alpha_space, alpha_time, maxit=200, tol=1e-8, verbose=True,
        debias=False, n_orient=1, tstep=4, wsize=32)

    # Also run L21 and check that we get the same
    X_hat_l21, _, _ = mixed_norm_solver(
        M, G, alpha_space, maxit=200, tol=1e-8, verbose=False, n_orient=1,
        active_set_size=None, debias=False)

    assert_allclose(X_hat_tf, X_hat_l21, rtol=1e-1)


def test_iterative_reweighted_mxne():
    """Test convergence of irMxNE solver"""
    n, p, t, alpha = 30, 40, 20, 1
    rng = np.random.RandomState(0)
    G = rng.randn(n, p)
    G /= np.std(G, axis=0)[None, :]
    X = np.zeros((p, t))
    X[0] = 3
    X[4] = -2
    M = np.dot(G, X)

    X_hat_l21, _, _ = mixed_norm_solver(
        M, G, alpha, maxit=1000, tol=1e-8, verbose=False, n_orient=1,
        active_set_size=None, debias=False, solver='bcd')
    X_hat_bcd, active_set, _ = iterative_mixed_norm_solver(
        M, G, alpha, 1, maxit=1000, tol=1e-8, active_set_size=None,
        debias=False, solver='bcd')
    X_hat_prox, active_set, _ = iterative_mixed_norm_solver(
        M, G, alpha, 1, maxit=1000, tol=1e-8, active_set_size=None,
        debias=False, solver='prox')
    assert_allclose(X_hat_bcd, X_hat_l21, rtol=1e-3)
    assert_allclose(X_hat_prox, X_hat_l21, rtol=1e-3)

    X_hat_prox, active_set, _ = iterative_mixed_norm_solver(
        M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=None,
        debias=True, solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat_bcd, active_set, _ = iterative_mixed_norm_solver(
        M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=2,
        debias=True, solver='bcd')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat_cd, active_set, _ = iterative_mixed_norm_solver(
        M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=None,
        debias=True, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    assert_array_almost_equal(X_hat_prox, X_hat_cd, 5)
    assert_array_almost_equal(X_hat_bcd, X_hat_cd, 5)

    X_hat_bcd, active_set, _ = iterative_mixed_norm_solver(
        M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=2,
        debias=True, n_orient=2, solver='bcd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    # suppress a coordinate-descent warning here
    with warnings.catch_warnings(record=True):
        X_hat_cd, active_set, _ = iterative_mixed_norm_solver(
            M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=2,
            debias=True, n_orient=2, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    assert_array_equal(X_hat_bcd, X_hat_cd, 5)

    X_hat_bcd, active_set, _ = iterative_mixed_norm_solver(
        M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=2, debias=True,
        n_orient=5)
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    with warnings.catch_warnings(record=True):  # coordinate-ascent warning
        X_hat_cd, active_set, _ = iterative_mixed_norm_solver(
            M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=2,
            debias=True, n_orient=5, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    assert_array_equal(X_hat_bcd, X_hat_cd, 5)
