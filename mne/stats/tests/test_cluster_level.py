import numpy as np
from numpy.testing import assert_equal

from ..cluster_level import permutation_cluster_test, permutation_cluster_t_test


def test_cluster_permutation_test():
    """Test cluster level permutations tests.
    """
    noiselevel = 20

    normfactor = np.hanning(20).sum()

    condition1 = np.random.randn(40, 350) * noiselevel
    for c in condition1:
        c[:] = np.convolve(c, np.hanning(20), mode="same") / normfactor

    condition2 = np.random.randn(33, 350) * noiselevel
    for c in condition2:
        c[:] = np.convolve(c, np.hanning(20), mode="same") / normfactor

    pseudoekp = 5 * np.hanning(150)[None,:]
    condition1[:, 100:250] += pseudoekp
    condition2[:, 100:250] -= pseudoekp

    T_obs, clusters, cluster_p_values, hist = permutation_cluster_test(
                                [condition1, condition2], n_permutations=500,
                                tail=1)
    assert_equal(np.sum(cluster_p_values < 0.05), 1)

    T_obs, clusters, cluster_p_values, hist = permutation_cluster_test(
                                [condition1, condition2], n_permutations=500,
                                tail=0)
    assert_equal(np.sum(cluster_p_values < 0.05), 1)

    T_obs, clusters, cluster_p_values, hist = permutation_cluster_test(
                                [condition1, condition2], n_permutations=500,
                                tail=-1)
    assert_equal(np.sum(cluster_p_values < 0.05), 0)

    condition1 = condition1[:,:,None] # to test 2D also
    T_obs, clusters, cluster_p_values, hist = permutation_cluster_t_test(
                                condition1, n_permutations=500, tail=0)
    assert_equal(np.sum(cluster_p_values < 0.05), 1)
