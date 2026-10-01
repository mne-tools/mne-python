# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# Parts of this code are taken from scikit-learn

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy import signal, stats

from mne.preprocessing.infomax_ import infomax
from mne.utils import check_random_state, pinv

pytest.importorskip("sklearn")


def center_and_norm(x, axis=-1):
    """Center and norm x in place.

    Parameters
    ----------
    x: ndarray
        Array with an axis of observations (statistical units) measured on
        random variables.
    axis : int
        Axis along which the mean and variance are calculated.
    """
    x = np.rollaxis(x, axis)
    x -= x.mean(axis=0)
    x /= x.std(axis=0)


def test_infomax_blowup():
    """Test the infomax algorithm blowup condition."""
    n_samples = 100
    # Generate two sources:
    s1 = (2 * np.sin(np.linspace(0, 100, n_samples)) > 0) - 1
    s2 = stats.t.rvs(1, size=n_samples, random_state=0)
    s = np.c_[s1, s2].T
    center_and_norm(s)
    s1, s2 = s

    # Mixing angle
    phi = 0.6
    mixing = np.array(
        [[np.cos(phi), np.sin(phi)], [np.sin(phi), -np.cos(phi)]]  # noqa: E241
    )
    m = np.dot(mixing, s)

    center_and_norm(m)

    X = _get_pca(0).fit_transform(m.T)
    k_ = infomax(X, extended=True, l_rate=0.1, rng=0)
    s_ = np.dot(k_, X.T)

    center_and_norm(s_)
    s1_, s2_ = s_
    # Check to see if the sources have been estimated
    # in the wrong order
    if abs(np.dot(s1_, s2)) > abs(np.dot(s1_, s1)):
        s2_, s1_ = s_
    s1_ *= np.sign(np.dot(s1_, s1))
    s2_ *= np.sign(np.dot(s2_, s2))

    # Check that we have estimated the original sources
    assert_almost_equal(np.dot(s1_, s1) / n_samples, 1, decimal=2)
    assert_almost_equal(np.dot(s2_, s2) / n_samples, 1, decimal=2)


def test_infomax_simple():
    """Test the infomax algorithm on very simple data."""
    rng = np.random.default_rng(0)
    n_samples = 500
    # Generate two sources:
    s1 = (2 * np.sin(np.linspace(0, 100, n_samples)) > 0) - 1
    s2 = stats.t.rvs(1, size=n_samples, random_state=0)
    s = np.c_[s1, s2].T
    center_and_norm(s)
    s1, s2 = s

    # Mixing angle
    phi = 0.6
    mixing = np.array(
        [[np.cos(phi), np.sin(phi)], [np.sin(phi), -np.cos(phi)]]  # noqa: E241
    )
    for add_noise in (False, True):
        m = np.dot(mixing, s)
        if add_noise:
            m += rng.normal(scale=0.1, size=(2, n_samples))
        center_and_norm(m)

        algos = [True, False]
        for algo in algos:
            X = _get_pca(0).fit_transform(m.T)
            k_ = infomax(X, extended=algo, rng=0)
            s_ = np.dot(k_, X.T)

            center_and_norm(s_)
            s1_, s2_ = s_
            # Check to see if the sources have been estimated
            # in the wrong order
            if abs(np.dot(s1_, s2)) > abs(np.dot(s1_, s1)):
                s2_, s1_ = s_
            s1_ *= np.sign(np.dot(s1_, s1))
            s2_ *= np.sign(np.dot(s2_, s2))

            # Check that we have estimated the original sources
            if not add_noise:
                assert_almost_equal(np.dot(s1_, s1) / n_samples, 1, decimal=2)
                assert_almost_equal(np.dot(s2_, s2) / n_samples, 1, decimal=2)
            else:
                assert_almost_equal(np.dot(s1_, s1) / n_samples, 1, decimal=1)
                assert_almost_equal(np.dot(s2_, s2) / n_samples, 1, decimal=1)


def test_infomax_weights_ini():
    """Test the infomax algorithm w/initial weights matrix."""
    rng = np.random.default_rng(0)
    X = rng.random((3, 100))
    weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)

    w1 = infomax(X, max_iter=0, weights=weights, extended=True, rng=0)
    w2 = infomax(X, max_iter=0, weights=weights, extended=False, rng=0)

    assert_almost_equal(w1, weights)
    assert_almost_equal(w2, weights)


def test_non_square_infomax():
    """Test non-square infomax."""
    rng = np.random.default_rng(0)

    n_samples = 200
    # Generate two sources:
    t = np.linspace(0, 100, n_samples)
    s1 = np.sin(t)
    s2 = np.ceil(np.sin(np.pi * t))
    s = np.c_[s1, s2].T
    center_and_norm(s)
    s1, s2 = s

    # Mixing matrix
    n_observed = 6
    mixing = rng.standard_normal((n_observed, 2))
    for add_noise in (False, True):
        m = np.dot(mixing, s)

        if add_noise:
            m += rng.normal(scale=0.1, size=(n_observed, n_samples))

        center_and_norm(m)
        m = m.T
        m = _get_pca(0).fit_transform(m)
        # we need extended since input signals are sub-gaussian
        unmixing_ = infomax(m, rng=0, extended=True)
        s_ = np.dot(unmixing_, m.T)
        # Check that the mixing model described in the docstring holds:
        mixing_ = pinv(unmixing_.T)

        assert_almost_equal(m, s_.T.dot(mixing_))

        center_and_norm(s_)
        s1_, s2_ = s_
        # Check to see if the sources have been estimated
        # in the wrong order
        if abs(np.dot(s1_, s2)) > abs(np.dot(s1_, s1)):
            s2_, s1_ = s_
        s1_ *= np.sign(np.dot(s1_, s1))
        s2_ *= np.sign(np.dot(s2_, s2))

        # Check that we have estimated the original sources
        if not add_noise:
            assert_almost_equal(np.dot(s1_, s1) / n_samples, 1, decimal=2)
            assert_almost_equal(np.dot(s2_, s2) / n_samples, 1, decimal=2)


@pytest.mark.parametrize("return_n_iter", [True, False])
def test_infomax_n_iter(return_n_iter):
    """Test the return_n_iter kwarg."""
    rng = np.random.default_rng(0)
    X = rng.random((3, 100))
    max_iter = 1
    r = infomax(X, max_iter=max_iter, extended=True, return_n_iter=return_n_iter, rng=0)

    if return_n_iter:
        assert isinstance(r, tuple)
        assert r[1] == max_iter
    else:
        assert isinstance(r, np.ndarray)


def test_infomax_legacy_rng_nested():
    """Test legacy RNGs survive Infomax's nested permutation path."""
    X = np.random.default_rng(0).standard_normal((20, 2))
    results = []
    for random_state in (0, check_random_state(0)):
        results.append(
            infomax(
                X,
                block=5,
                extended=False,
                max_iter=1,
                random_state=random_state,
            )
        )
    assert_array_equal(results[0], results[1])


def _get_pca(rng=None):
    from sklearn.decomposition import PCA

    return PCA(n_components=2, whiten=True, svd_solver="randomized", random_state=rng)


def test_infomax_n_iter_reports_actual_iterations():
    """Test that infomax reports iterations performed, not the budget.

    Convergence was previously signalled by assigning ``step = max_iter`` to
    leave the training loop, so a converged fit returned ``max_iter`` however
    few iterations it had actually run. The value was therefore the budget,
    not the work, and carried no information.

    The second exit -- the small-angle branch -- assigned ``max_iter = step``
    instead and did report the true count, so which of the two fired decided
    whether ``n_iter`` was meaningful. That is why the effect shows on some
    data and not others.
    """
    rng = np.random.RandomState(0)
    n_samples = 2000
    t = np.linspace(0, 8, n_samples)
    sources = np.c_[
        np.sin(2 * t),
        np.sign(np.sin(3 * t)),
        signal.sawtooth(2 * np.pi * t),
    ]
    sources += 0.2 * rng.standard_normal(sources.shape)
    sources /= sources.std(axis=0)
    mixing = np.array([[1.0, 1.0, 1.0], [0.5, 2.0, 1.0], [1.5, 1.0, 2.0]])
    data = sources @ mixing.T
    data = (data - data.mean(0)) / data.std(0)

    for extended in (False, True):
        # A budget far above what the fit needs. The reported count must not
        # grow with the budget once the fit has converged.
        counts = [
            infomax(
                data,
                rng=np.random.RandomState(0),
                max_iter=max_iter,
                extended=extended,
                return_n_iter=True,
            )[1]
            for max_iter in (500, 2000)
        ]
        assert counts[0] == counts[1], (
            f"extended={extended}: n_iter tracked max_iter ({counts[0]} vs {counts[1]})"
        )
        assert counts[0] < 500, f"extended={extended}: n_iter == the budget"


def test_infomax_unmixing_unchanged_by_the_n_iter_fix():
    """Test that reporting the true count does not alter the solution.

    ``step += 1`` happens before the stopping rule, so replacing the
    assignments with ``break`` skips no work. A budget above the convergence
    point must give the same unmixing matrix as one far above it.
    """
    rng = np.random.RandomState(0)
    t = np.linspace(0, 8, 2000)
    sources = np.c_[
        np.sin(2 * t), np.sign(np.sin(3 * t)), signal.sawtooth(2 * np.pi * t)
    ]
    sources += 0.2 * rng.standard_normal(sources.shape)
    sources /= sources.std(axis=0)
    data = sources @ np.array([[1.0, 1.0, 1.0], [0.5, 2.0, 1.0], [1.5, 1.0, 2.0]]).T
    data = (data - data.mean(0)) / data.std(0)

    for extended in (False, True):
        weights = [
            infomax(
                data, rng=np.random.RandomState(0), max_iter=max_iter, extended=extended
            )
            for max_iter in (500, 2000)
        ]
        assert_array_equal(weights[0], weights[1])
