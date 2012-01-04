import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from nose.tools import assert_true

from ..cluster_level import permutation_cluster_test, \
                            permutation_cluster_1samp_test

noiselevel = 20

normfactor = np.hanning(20).sum()

rng = np.random.RandomState(42)
condition1 = rng.randn(40, 350) * noiselevel
for c in condition1:
    c[:] = np.convolve(c, np.hanning(20), mode="same") / normfactor

condition2 = rng.randn(33, 350) * noiselevel
for c in condition2:
    c[:] = np.convolve(c, np.hanning(20), mode="same") / normfactor

pseudoekp = 5 * np.hanning(150)[None, :]
condition1[:, 100:250] += pseudoekp
condition2[:, 100:250] -= pseudoekp


def test_cluster_permutation_test():
    """Test cluster level permutations tests."""
    T_obs, clusters, cluster_p_values, hist = permutation_cluster_test(
                                [condition1, condition2], n_permutations=500,
                                tail=1)
    assert_equal(np.sum(cluster_p_values < 0.05), 1)

    T_obs, clusters, cluster_p_values, hist = permutation_cluster_test(
                                [condition1, condition2], n_permutations=500,
                                tail=0)
    assert_equal(np.sum(cluster_p_values < 0.05), 1)


def test_cluster_permutation_t_test():
    """Test cluster level permutations T-test."""
    my_condition1 = condition1[:, :, None]  # to test 2D also
    T_obs, clusters, cluster_p_values, hist = permutation_cluster_1samp_test(
                                my_condition1, n_permutations=500, tail=0)
    assert_equal(np.sum(cluster_p_values < 0.05), 1)

    T_obs_pos, _, cluster_p_values_pos, _ = permutation_cluster_1samp_test(
                                my_condition1, n_permutations=500, tail=1,
                                threshold=1.67)

    T_obs_neg, _, cluster_p_values_neg, _ = permutation_cluster_1samp_test(
                                -my_condition1, n_permutations=500, tail=-1,
                                threshold=-1.67)
    assert_array_equal(T_obs_pos, -T_obs_neg)
    assert_array_equal(cluster_p_values_pos < 0.05,
                       cluster_p_values_neg < 0.05)


def test_cluster_permutation_t_test_with_connectivity():
    """Test cluster level permutations T-test with connectivity matrix."""
    try:
        try:
            from sklearn.feature_extraction.image import grid_to_graph
        except ImportError:
            from scikits.learn.feature_extraction.image import grid_to_graph
    except ImportError:
        return

    out = permutation_cluster_1samp_test(condition1, n_permutations=500)
    connectivity = grid_to_graph(1, condition1.shape[1])
    out_connectivity = permutation_cluster_1samp_test(condition1,
                             n_permutations=500, connectivity=connectivity)
    assert_array_equal(out[0], out_connectivity[0])
    for a, b in zip(out_connectivity[1], out[1]):
        assert_true(np.sum(out[0][a]) == np.sum(out[0][b]))
        assert_true(np.all(a[b]))
