# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse, stats

from mne.stats import permutation_cluster_1samp_test
from mne.stats.permutations import (
    _ci,
    _extreme_stat,
    bootstrap_confidence_interval,
    permutation_t_test,
)


def test_permutation_t_test():
    """Test T-test based on permutations."""
    # 1 sample t-test
    # seed chosen so the injected effects are detected at the asserted rate
    rng = np.random.default_rng(5)
    n_samples, n_tests = 30, 5
    X = rng.standard_normal((n_samples, n_tests))
    X[:, :2] += 1

    t_obs, p_values, H0 = permutation_t_test(X, n_permutations=999, tail=0, rng=0)
    assert (p_values > 0).all()
    assert len(H0) == 999
    is_significant = p_values < 0.05
    assert_array_equal(is_significant, [True, True, False, False, False])

    t_obs, p_values, H0 = permutation_t_test(X, n_permutations=999, tail=1, rng=0)
    assert (p_values > 0).all()
    assert len(H0) == 999
    is_significant = p_values < 0.05
    assert_array_equal(is_significant, [True, True, False, False, False])

    t_obs, p_values, H0 = permutation_t_test(X, n_permutations=999, tail=-1, rng=0)
    is_significant = p_values < 0.05
    assert_array_equal(is_significant, [False, False, False, False, False])

    X *= -1
    t_obs, p_values, H0 = permutation_t_test(X, n_permutations=999, tail=-1, rng=0)
    assert (p_values > 0).all()
    assert len(H0) == 999
    is_significant = p_values < 0.05
    assert_array_equal(is_significant, [True, True, False, False, False])

    # test chunked calculation in _extreme_stat
    X2 = np.mean(X**2, axis=0)
    perms = rng.choice([-1, 1], size=(100, n_samples))
    dof_scaling = np.sqrt(n_samples / (n_samples - 1.0))
    res_full = _extreme_stat(X, X2, perms, dof_scaling, 0, step=100)
    res_chunked = _extreme_stat(X, X2, perms, dof_scaling, 0, step=37)
    assert_allclose(res_full, res_chunked)

    # check equivalence with spatio_temporal_cluster_test
    for adjacency, tail in ((sparse.eye_array(n_tests), 0), (False, -1)):
        t_obs, p_values, _ = permutation_t_test(X, 999, tail=tail, rng=0)
        t_obs_clust, clusters, p_values_clust, _ = permutation_cluster_1samp_test(
            X, n_permutations=999, tail=tail, rng=0, adjacency=adjacency
        )
        # the cluster tests drop any variables that don't get thresholded
        keep = [cluster[0][0] for cluster in clusters]  # one variable each
        assert_allclose(t_obs_clust, t_obs)
        assert_allclose(p_values_clust, p_values[keep], atol=1e-2)


@pytest.mark.parametrize(
    "tail_name,tail_code",
    [
        ("two-sided", 0),
        ("less", -1),
        ("greater", 1),
    ],
)
def test_permutation_t_test_tail(tail_name, tail_code):
    """Test that tails work properly."""
    # seed chosen so the injected effects are detected at the asserted rate
    rng = np.random.default_rng(1)
    X = rng.standard_normal((18, 1))

    t_obs, p_values, H0 = permutation_t_test(X, n_permutations="all", tail=tail_code)
    assert len(H0) == 2 ** (len(X) - (tail_code == 0))  # exact test
    t_obs_scipy, p_values_scipy = stats.ttest_1samp(X[:, 0], 0, alternative=tail_name)
    assert_allclose(t_obs[0], t_obs_scipy, 8)
    assert_allclose(p_values[0], p_values_scipy, rtol=1e-2)


def test_ci():
    """Test confidence intervals."""
    # isolated test of CI functions
    arr = np.linspace(0, 1, 1000)[..., np.newaxis]
    assert_allclose(
        _ci(arr, method="parametric"), _ci(arr, method="bootstrap"), rtol=0.005
    )
    assert_allclose(
        bootstrap_confidence_interval(arr, stat_fun="median", rng=0),
        bootstrap_confidence_interval(arr, stat_fun="mean", rng=0),
        rtol=0.1,
    )
