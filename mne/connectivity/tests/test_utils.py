import numpy as np
from numpy.testing import assert_array_equal
import pytest

from mne.connectivity import seed_target_indices, degree


def test_indices():
    """Test connectivity indexing methods."""
    n_seeds_test = [1, 3, 4]
    n_targets_test = [2, 3, 200]
    rng = np.random.RandomState(42)
    for n_seeds in n_seeds_test:
        for n_targets in n_targets_test:
            idx = rng.permutation(np.arange(n_seeds + n_targets))
            seeds = idx[:n_seeds]
            targets = idx[n_seeds:]
            indices = seed_target_indices(seeds, targets)
            assert len(indices) == 2
            assert len(indices[0]) == len(indices[1])
            assert len(indices[0]) == n_seeds * n_targets
            for seed in seeds:
                assert np.sum(indices[0] == seed) == n_targets
            for target in targets:
                assert np.sum(indices[1] == target) == n_seeds


def test_degree():
    """Test degree function."""
    # degenerate conditions
    with pytest.raises(ValueError, match='threshold'):
        degree(np.eye(3), 2.)
    # a simple one
    corr = np.eye(10)
    assert_array_equal(degree(corr), np.zeros(10))
    # more interesting
    corr = np.array([[0.5, 0.7, 0.4],
                     [0.1, 0.3, 0.6],
                     [0.2, 0.8, 0.9]])
    deg = degree(corr, 1)
    assert_array_equal(deg, [2, 2, 2])

    # The values for assert_array_equal below were obtained with:
    #
    # >>> import bct
    # >>> bct.degrees_und(bct.utils.threshold_proportional(corr, 0.25) > 0)
    #
    # But they can also be figured out just from the structure.

    # Asymmetric (6 usable nodes)
    assert_array_equal(degree(corr, 0.33), [0, 2, 0])
    assert_array_equal(degree(corr, 0.5), [0, 2, 1])
    # Symmetric (3 usable nodes)
    corr = (corr + corr.T) / 2.
    assert_array_equal(degree(corr, 0.33), [0, 1, 1])
    assert_array_equal(degree(corr, 0.66), [1, 2, 1])
