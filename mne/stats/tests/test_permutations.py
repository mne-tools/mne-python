# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

from numpy.testing import assert_array_equal, assert_allclose
import numpy as np
from scipy import stats, sparse

from mne.stats import permutation_cluster_1samp_test
from mne.stats.permutations import (permutation_t_test, _ci,
                                    bootstrap_confidence_interval)
from mne.utils import check_version


def test_permutation_t_test():
    """Test T-test based on permutations."""
    # 1 sample t-test
    np.random.seed(10)
    n_samples, n_tests = 30, 5
    X = np.random.randn(n_samples, n_tests)
    X[:, :2] += 1

    t_obs, p_values, H0 = permutation_t_test(
        X, n_permutations=999, tail=0, seed=0)
    assert (p_values > 0).all()
    assert len(H0) == 999
    is_significant = p_values < 0.05
    assert_array_equal(is_significant, [True, True, False, False, False])

    t_obs, p_values, H0 = permutation_t_test(
        X, n_permutations=999, tail=1, seed=0)
    assert (p_values > 0).all()
    assert len(H0) == 999
    is_significant = p_values < 0.05
    assert_array_equal(is_significant, [True, True, False, False, False])

    t_obs, p_values, H0 = permutation_t_test(
        X, n_permutations=999, tail=-1, seed=0)
    is_significant = p_values < 0.05
    assert_array_equal(is_significant, [False, False, False, False, False])

    X *= -1
    t_obs, p_values, H0 = permutation_t_test(
        X, n_permutations=999, tail=-1, seed=0)
    assert (p_values > 0).all()
    assert len(H0) == 999
    is_significant = p_values < 0.05
    assert_array_equal(is_significant, [True, True, False, False, False])

    # check equivalence with spatio_temporal_cluster_test
    for adjacency in (sparse.eye(n_tests), False):
        t_obs_clust, _, p_values_clust, _ = permutation_cluster_1samp_test(
            X, n_permutations=999, seed=0, adjacency=adjacency,
            out_type='mask')
        # the cluster tests drop any clusters that don't get thresholded
        keep = p_values < 1
        assert_allclose(t_obs_clust, t_obs)
        assert_allclose(p_values_clust, p_values[keep], atol=1e-2)

    X = np.random.randn(18, 1)
    t_obs, p_values, H0 = permutation_t_test(X, n_permutations='all')
    t_obs_scipy, p_values_scipy = stats.ttest_1samp(X[:, 0], 0)
    assert_allclose(t_obs[0], t_obs_scipy, 8)
    assert_allclose(p_values[0], p_values_scipy, rtol=1e-2)


def test_ci():
    """Test confidence intervals."""
    # isolated test of CI functions
    arr = np.linspace(0, 1, 1000)[..., np.newaxis]
    assert_allclose(_ci(arr, method="parametric"),
                    _ci(arr, method="bootstrap"), rtol=.005)
    assert_allclose(bootstrap_confidence_interval(arr, stat_fun="median",
                                                  random_state=0),
                    bootstrap_confidence_interval(arr, stat_fun="mean",
                                                  random_state=0),
                    rtol=.1)
    # smoke test for new API
    if check_version('numpy', '1.17'):
        random_state = np.random.default_rng(0)
        bootstrap_confidence_interval(arr, random_state=random_state)
