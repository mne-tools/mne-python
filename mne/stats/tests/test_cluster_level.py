import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from nose.tools import assert_true

from mne.stats.cluster_level import permutation_cluster_test, \
                            permutation_cluster_1samp_test

noiselevel = 20

normfactor = np.hanning(20).sum()

rng = np.random.RandomState(42)
condition1_1d = rng.randn(40, 350) * noiselevel
for c in condition1_1d:
    c[:] = np.convolve(c, np.hanning(20), mode="same") / normfactor

condition2_1d = rng.randn(33, 350) * noiselevel
for c in condition2_1d:
    c[:] = np.convolve(c, np.hanning(20), mode="same") / normfactor

pseudoekp = 5 * np.hanning(150)[None, :]
condition1_1d[:, 100:250] += pseudoekp
condition2_1d[:, 100:250] -= pseudoekp

condition1_2d = condition1_1d[:, :, np.newaxis]
condition2_2d = condition2_1d[:, :, np.newaxis]


def test_cluster_permutation_test():
    """Test cluster level permutations tests."""
    for condition1, condition2 in zip((condition1_1d, condition1_2d),
                                      (condition2_1d, condition2_2d)):
        T_obs, clusters, cluster_p_values, hist = permutation_cluster_test(
                                    [condition1, condition2],
                                    n_permutations=500, tail=1, seed=1,
                                    buffer_size=None)
        assert_equal(np.sum(cluster_p_values < 0.05), 1)

        T_obs, clusters, cluster_p_values, hist = permutation_cluster_test(
                                    [condition1, condition2],
                                    n_permutations=500, tail=0, seed=1,
                                    buffer_size=None)
        assert_equal(np.sum(cluster_p_values < 0.05), 1)

        # test with buffer_size enabled and 2 jobs
        buffer_size = condition1.shape[1] // 10
        T_obs, clusters, cluster_p_values_buff, hist =\
            permutation_cluster_test([condition1, condition2],
                                    n_permutations=500, tail=0, seed=1,
                                    buffer_size=buffer_size, n_jobs=2)

        assert_array_equal(cluster_p_values, cluster_p_values_buff)


def test_cluster_permutation_t_test():
    """Test cluster level permutations T-test."""
    for condition1 in (condition1_1d, condition1_2d):
        T_obs, clusters, cluster_p_values, hist =\
            permutation_cluster_1samp_test(condition1, n_permutations=500,
                                           tail=0, seed=1, buffer_size=None)
        assert_equal(np.sum(cluster_p_values < 0.05), 1)

        T_obs_pos, _, cluster_p_values_pos, _ =\
            permutation_cluster_1samp_test(condition1, n_permutations=500,
                                    tail=1, threshold=1.67, seed=1,
                                    buffer_size=None)

        T_obs_neg, _, cluster_p_values_neg, _ =\
            permutation_cluster_1samp_test(-condition1, n_permutations=500,
                                    tail=-1, threshold=-1.67, seed=1,
                                    buffer_size=None)
        assert_array_equal(T_obs_pos, -T_obs_neg)
        assert_array_equal(cluster_p_values_pos < 0.05,
                           cluster_p_values_neg < 0.05)

        # test with buffer_size enabled and 2 jobs
        buffer_size = condition1.shape[1] // 10
        T_obs_neg, _, cluster_p_values_neg_buff, _ = \
            permutation_cluster_1samp_test(-condition1, n_permutations=500,
                                            tail=-1, threshold=-1.67, seed=1,
                                            buffer_size=buffer_size, n_jobs=2)

        assert_array_equal(cluster_p_values_neg, cluster_p_values_neg_buff)


def test_cluster_permutation_t_test_with_connectivity():
    """Test cluster level permutations T-test with connectivity matrix."""
    try:
        try:
            from sklearn.feature_extraction.image import grid_to_graph
        except ImportError:
            from scikits.learn.feature_extraction.image import grid_to_graph
    except ImportError:
        return

    out = permutation_cluster_1samp_test(condition1_1d, n_permutations=500)
    connectivity = grid_to_graph(1, condition1_1d.shape[1])
    out_connectivity = permutation_cluster_1samp_test(condition1_1d,
                             n_permutations=500, connectivity=connectivity)
    assert_array_equal(out[0], out_connectivity[0])
    for a, b in zip(out_connectivity[1], out[1]):
        assert_true(np.sum(out[0][a]) == np.sum(out[0][b]))
        assert_true(np.all(a[b]))
