# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

from mne.fixes import _eye_array
from mne.stats import permutation_cluster_1samp_test
from mne.stats.permutations import (
    _ci,
    bootstrap_confidence_interval,
    permutation_t_test,
)


def test_permutation_t_test():
    """Test T-test based on permutations."""
    # 1 sample t-test
    np.random.seed(10)
    n_samples, n_tests = 30, 5
    X = np.random.randn(n_samples, n_tests)
    X[:, :2] += 1

    t_obs, p_values, H0 = permutation_t_test(X, n_permutations=999, tail=0, seed=0)
    assert (p_values > 0).all()
    assert len(H0) == 999
    is_significant = p_values < 0.05
    assert_array_equal(is_significant, [True, True, False, False, False])

    t_obs, p_values, H0 = permutation_t_test(X, n_permutations=999, tail=1, seed=0)
    assert (p_values > 0).all()
    assert len(H0) == 999
    is_significant = p_values < 0.05
    assert_array_equal(is_significant, [True, True, False, False, False])

    t_obs, p_values, H0 = permutation_t_test(X, n_permutations=999, tail=-1, seed=0)
    is_significant = p_values < 0.05
    assert_array_equal(is_significant, [False, False, False, False, False])

    X *= -1
    t_obs, p_values, H0 = permutation_t_test(X, n_permutations=999, tail=-1, seed=0)
    assert (p_values > 0).all()
    assert len(H0) == 999
    is_significant = p_values < 0.05
    assert_array_equal(is_significant, [True, True, False, False, False])

    # check equivalence with spatio_temporal_cluster_test
    for adjacency in (_eye_array(n_tests), False):
        t_obs_clust, _, p_values_clust, _ = permutation_cluster_1samp_test(
            X, n_permutations=999, seed=0, adjacency=adjacency, out_type="mask"
        )
        # the cluster tests drop any clusters that don't get thresholded
        keep = p_values < 1
        assert_allclose(t_obs_clust, t_obs)
        assert_allclose(p_values_clust, p_values[keep], atol=1e-2)


@pytest.mark.parametrize(
    "tail_name,tail_code",
    [
        ("two-sided", 0),
        pytest.param(
            "less", -1, marks=pytest.mark.xfail(reason="Bug in permutation function")
        ),
        pytest.param(
            "greater", 1, marks=pytest.mark.xfail(reason="Bug in permutation function")
        ),
    ],
)
def test_permutation_t_test_tail(tail_name, tail_code):
    """Test that tails work properly."""
    X = np.random.randn(18, 1)

    t_obs, p_values, _ = permutation_t_test(X, n_permutations="all", tail=tail_code)
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
        bootstrap_confidence_interval(arr, stat_fun="median", random_state=0),
        bootstrap_confidence_interval(arr, stat_fun="mean", random_state=0),
        rtol=0.1,
    )
