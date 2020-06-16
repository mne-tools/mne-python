# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Daniel Strohmeier <daniel.strohmeier@gmail.com>
#
# License: Simplified BSD

import pytest
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_allclose, assert_array_less)

from mne.inverse_sparse.mxne_optim import (mixed_norm_solver,
                                           tf_mixed_norm_solver,
                                           iterative_mixed_norm_solver,
                                           iterative_tf_mixed_norm_solver,
                                           norm_epsilon_inf, norm_epsilon,
                                           _Phi, _PhiT, dgap_l21l1)
from mne.time_frequency._stft import stft_norm2


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
    """Test convergence of MxNE solver."""
    n, p, t, alpha = 30, 40, 20, 1.
    rng = np.random.RandomState(0)
    G = rng.randn(n, p)
    G /= np.std(G, axis=0)[None, :]
    X = np.zeros((p, t))
    X[0] = 3
    X[4] = -2
    M = np.dot(G, X)

    args = (M, G, alpha, 1000, 1e-8)
    with pytest.warns(None):  # CD
        X_hat_prox, active_set, _ = mixed_norm_solver(
            *args, active_set_size=None,
            debias=True, solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    with pytest.warns(None):  # CD
        X_hat_cd, active_set, _, gap_cd = mixed_norm_solver(
            *args, active_set_size=None,
            debias=True, solver='cd', return_gap=True)
    assert_array_less(gap_cd, 1e-8)
    assert_array_equal(np.where(active_set)[0], [0, 4])
    with pytest.warns(None):  # CD
        X_hat_bcd, active_set, E, gap_bcd = mixed_norm_solver(
            M, G, alpha, maxit=1000, tol=1e-8, active_set_size=None,
            debias=True, solver='bcd', return_gap=True)
    assert_array_less(gap_bcd, 9.6e-9)
    assert_array_equal(np.where(active_set)[0], [0, 4])
    assert_allclose(X_hat_prox, X_hat_cd, rtol=1e-2)
    assert_allclose(X_hat_prox, X_hat_bcd, rtol=1e-2)
    assert_allclose(X_hat_bcd, X_hat_cd, rtol=1e-2)

    with pytest.warns(None):  # CD
        X_hat_prox, active_set, _ = mixed_norm_solver(
            *args, active_set_size=2, debias=True, solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    with pytest.warns(None):  # CD
        X_hat_cd, active_set, _ = mixed_norm_solver(
            *args, active_set_size=2, debias=True, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    with pytest.warns(None):  # CD
        X_hat_bcd, active_set, _ = mixed_norm_solver(
            *args, active_set_size=2, debias=True, solver='bcd')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    assert_allclose(X_hat_bcd, X_hat_cd, rtol=1e-2)
    assert_allclose(X_hat_bcd, X_hat_prox, rtol=1e-2)

    with pytest.warns(None):  # CD
        X_hat_prox, active_set, _ = mixed_norm_solver(
            *args, active_set_size=2, debias=True, n_orient=2, solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    with pytest.warns(None):  # CD
        X_hat_bcd, active_set, _ = mixed_norm_solver(
            *args, active_set_size=2, debias=True, n_orient=2, solver='bcd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])

    # suppress a coordinate-descent warning here
    with pytest.warns(RuntimeWarning, match='descent'):
        X_hat_cd, active_set, _ = mixed_norm_solver(
            *args, active_set_size=2, debias=True, n_orient=2, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    assert_allclose(X_hat_bcd, X_hat_prox, rtol=1e-2)
    assert_allclose(X_hat_bcd, X_hat_cd, rtol=1e-2)

    with pytest.warns(None):  # CD
        X_hat_bcd, active_set, _ = mixed_norm_solver(
            *args, active_set_size=2, debias=True, n_orient=5, solver='bcd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    with pytest.warns(None):  # CD
        X_hat_prox, active_set, _ = mixed_norm_solver(
            *args, active_set_size=2, debias=True, n_orient=5, solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    with pytest.warns(RuntimeWarning, match='descent'):
        X_hat_cd, active_set, _ = mixed_norm_solver(
            *args, active_set_size=2, debias=True, n_orient=5, solver='cd')

    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    assert_array_equal(X_hat_bcd, X_hat_cd)
    assert_allclose(X_hat_bcd, X_hat_prox, rtol=1e-2)


def test_tf_mxne():
    """Test convergence of TF-MxNE solver."""
    alpha_space = 10.
    alpha_time = 5.

    M, G, active_set = _generate_tf_data()

    with pytest.warns(None):  # CD
        X_hat_tf, active_set_hat_tf, E, gap_tfmxne = tf_mixed_norm_solver(
            M, G, alpha_space, alpha_time, maxit=200, tol=1e-8, verbose=True,
            n_orient=1, tstep=4, wsize=32, return_gap=True)
    assert_array_less(gap_tfmxne, 1e-8)
    assert_array_equal(np.where(active_set_hat_tf)[0], active_set)


def test_norm_epsilon():
    """Test computation of espilon norm on TF coefficients."""
    tstep = np.array([2])
    wsize = np.array([4])
    n_times = 10
    n_steps = np.ceil(n_times / tstep.astype(float)).astype(int)
    n_freqs = wsize // 2 + 1
    n_coefs = n_steps * n_freqs
    phi = _Phi(wsize, tstep, n_coefs)

    Y = np.zeros(n_steps * n_freqs)
    l1_ratio = 0.03
    assert_allclose(norm_epsilon(Y, l1_ratio, phi), 0.)

    Y[0] = 2.
    assert_allclose(norm_epsilon(Y, l1_ratio, phi), np.max(Y))

    l1_ratio = 1.
    assert_allclose(norm_epsilon(Y, l1_ratio, phi), np.max(Y))
    # dummy value without random:
    Y = np.arange(n_steps * n_freqs).reshape(-1, )
    l1_ratio = 0.0
    assert_allclose(norm_epsilon(Y, l1_ratio, phi) ** 2,
                    stft_norm2(Y.reshape(-1, n_freqs[0], n_steps[0])))

    l1_ratio = 0.03
    # test that vanilla epsilon norm = weights equal to 1
    w_time = np.ones(n_coefs[0])
    Y = np.abs(np.random.randn(n_coefs[0]))
    assert_allclose(norm_epsilon(Y, l1_ratio, phi),
                    norm_epsilon(Y, l1_ratio, phi, w_time=w_time))

    # scaling w_time and w_space by the same amount should divide
    # epsilon norm by the same amount
    Y = np.arange(n_coefs) + 1
    mult = 2.
    assert_allclose(
        norm_epsilon(Y, l1_ratio, phi, w_space=1,
                     w_time=np.ones(n_coefs)) / mult,
        norm_epsilon(Y, l1_ratio, phi, w_space=mult,
                     w_time=mult * np.ones(n_coefs)))


@pytest.mark.slowtest  # slow-ish on Travis OSX
@pytest.mark.timeout(60)  # ~30 sec on Travis OSX and Linux OpenBLAS
def test_dgapl21l1():
    """Test duality gap for L21 + L1 regularization."""
    n_orient = 2
    M, G, active_set = _generate_tf_data()
    n_times = M.shape[1]
    n_sources = G.shape[1]
    tstep, wsize = np.array([4, 2]), np.array([64, 16])
    n_steps = np.ceil(n_times / tstep.astype(float)).astype(int)
    n_freqs = wsize // 2 + 1
    n_coefs = n_steps * n_freqs
    phi = _Phi(wsize, tstep, n_coefs)
    phiT = _PhiT(tstep, n_freqs, n_steps, n_times)

    for l1_ratio in [0.05, 0.1]:
        alpha_max = norm_epsilon_inf(G, M, phi, l1_ratio, n_orient)
        alpha_space = (1. - l1_ratio) * alpha_max
        alpha_time = l1_ratio * alpha_max

        Z = np.zeros([n_sources, phi.n_coefs.sum()])
        # for alpha = alpha_max, Z = 0 is the solution so the dgap is 0
        gap = dgap_l21l1(M, G, Z, np.ones(n_sources, dtype=bool),
                         alpha_space, alpha_time, phi, phiT,
                         n_orient, -np.inf)[0]

        assert_allclose(0., gap)
        # check that solution for alpha smaller than alpha_max is non 0:
        X_hat_tf, active_set_hat_tf, E, gap = tf_mixed_norm_solver(
            M, G, alpha_space / 1.01, alpha_time / 1.01, maxit=200, tol=1e-8,
            verbose=True, debias=False, n_orient=n_orient, tstep=tstep,
            wsize=wsize, return_gap=True)
        # allow possible small numerical errors (negative gap)
        assert_array_less(-1e-10, gap)
        assert_array_less(gap, 1e-8)
        assert_array_less(1, len(active_set_hat_tf))

        X_hat_tf, active_set_hat_tf, E, gap = tf_mixed_norm_solver(
            M, G, alpha_space / 5., alpha_time / 5., maxit=200, tol=1e-8,
            verbose=True, debias=False, n_orient=n_orient, tstep=tstep,
            wsize=wsize, return_gap=True)
        assert_array_less(-1e-10, gap)
        assert_array_less(gap, 1e-8)
        assert_array_less(1, len(active_set_hat_tf))


def test_tf_mxne_vs_mxne():
    """Test equivalence of TF-MxNE (with alpha_time=0) and MxNE."""
    alpha_space = 60.
    alpha_time = 0.

    M, G, active_set = _generate_tf_data()

    X_hat_tf, active_set_hat_tf, E = tf_mixed_norm_solver(
        M, G, alpha_space, alpha_time, maxit=200, tol=1e-8,
        verbose=True, debias=False, n_orient=1, tstep=4, wsize=32)

    # Also run L21 and check that we get the same
    X_hat_l21, _, _ = mixed_norm_solver(
        M, G, alpha_space, maxit=200, tol=1e-8, verbose=False, n_orient=1,
        active_set_size=None, debias=False)

    assert_allclose(X_hat_tf, X_hat_l21, rtol=1e-1)


@pytest.mark.slowtest  # slow-ish on Travis OSX
def test_iterative_reweighted_mxne():
    """Test convergence of irMxNE solver."""
    n, p, t, alpha = 30, 40, 20, 1
    rng = np.random.RandomState(0)
    G = rng.randn(n, p)
    G /= np.std(G, axis=0)[None, :]
    X = np.zeros((p, t))
    X[0] = 3
    X[4] = -2
    M = np.dot(G, X)

    with pytest.warns(None):  # CD
        X_hat_l21, _, _ = mixed_norm_solver(
            M, G, alpha, maxit=1000, tol=1e-8, verbose=False, n_orient=1,
            active_set_size=None, debias=False, solver='bcd')
    with pytest.warns(None):  # CD
        X_hat_bcd, active_set, _ = iterative_mixed_norm_solver(
            M, G, alpha, 1, maxit=1000, tol=1e-8, active_set_size=None,
            debias=False, solver='bcd')
    with pytest.warns(None):  # CD
        X_hat_prox, active_set, _ = iterative_mixed_norm_solver(
            M, G, alpha, 1, maxit=1000, tol=1e-8, active_set_size=None,
            debias=False, solver='prox')
    assert_allclose(X_hat_bcd, X_hat_l21, rtol=1e-3)
    assert_allclose(X_hat_prox, X_hat_l21, rtol=1e-3)

    with pytest.warns(None):  # CD
        X_hat_prox, active_set, _ = iterative_mixed_norm_solver(
            M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=None,
            debias=True, solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    with pytest.warns(None):  # CD
        X_hat_bcd, active_set, _ = iterative_mixed_norm_solver(
            M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=2,
            debias=True, solver='bcd')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    with pytest.warns(None):  # CD
        X_hat_cd, active_set, _ = iterative_mixed_norm_solver(
            M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=None,
            debias=True, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    assert_array_almost_equal(X_hat_prox, X_hat_cd, 5)
    assert_array_almost_equal(X_hat_bcd, X_hat_cd, 5)

    with pytest.warns(None):  # CD
        X_hat_bcd, active_set, _ = iterative_mixed_norm_solver(
            M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=2,
            debias=True, n_orient=2, solver='bcd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    # suppress a coordinate-descent warning here
    with pytest.warns(RuntimeWarning, match='descent'):
        X_hat_cd, active_set, _ = iterative_mixed_norm_solver(
            M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=2,
            debias=True, n_orient=2, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    assert_array_equal(X_hat_bcd, X_hat_cd, 5)

    X_hat_bcd, active_set, _ = iterative_mixed_norm_solver(
        M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=2, debias=True,
        n_orient=5)
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    with pytest.warns(RuntimeWarning, match='descent'):
        X_hat_cd, active_set, _ = iterative_mixed_norm_solver(
            M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=2,
            debias=True, n_orient=5, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    assert_array_equal(X_hat_bcd, X_hat_cd, 5)


@pytest.mark.slowtest
def test_iterative_reweighted_tfmxne():
    """Test convergence of irTF-MxNE solver."""
    M, G, true_active_set = _generate_tf_data()
    alpha_space = 38.
    alpha_time = 0.5
    tstep, wsize = [4, 2], [64, 16]

    X_hat_tf, _, _ = tf_mixed_norm_solver(
        M, G, alpha_space, alpha_time, maxit=1000, tol=1e-4, wsize=wsize,
        tstep=tstep, verbose=False, n_orient=1, debias=False)
    X_hat_bcd, active_set, _ = iterative_tf_mixed_norm_solver(
        M, G, alpha_space, alpha_time, 1, wsize=wsize, tstep=tstep,
        maxit=1000, tol=1e-4, debias=False, verbose=False)
    assert_allclose(X_hat_tf, X_hat_bcd, rtol=1e-3)
    assert_array_equal(np.where(active_set)[0], true_active_set)

    alpha_space = 50.
    X_hat_bcd, active_set, _ = iterative_tf_mixed_norm_solver(
        M, G, alpha_space, alpha_time, 3, wsize=wsize, tstep=tstep,
        n_orient=5, maxit=1000, tol=1e-4, debias=False, verbose=False)
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])

    alpha_space = 40.
    X_hat_bcd, active_set, _ = iterative_tf_mixed_norm_solver(
        M, G, alpha_space, alpha_time, 2, wsize=wsize, tstep=tstep,
        n_orient=2, maxit=1000, tol=1e-4, debias=False, verbose=False)
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
