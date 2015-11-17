import numpy as np
from nose.tools import assert_true

from mne.connectivity import seed_target_indices


def test_indices():
    """Test connectivity indexing methods"""
    n_seeds_test = [1, 3, 4]
    n_targets_test = [2, 3, 200]
    rng = np.random.RandomState(42)
    for n_seeds in n_seeds_test:
        for n_targets in n_targets_test:
            idx = rng.permutation(np.arange(n_seeds + n_targets))
            seeds = idx[:n_seeds]
            targets = idx[n_seeds:]
            indices = seed_target_indices(seeds, targets)
            assert_true(len(indices) == 2)
            assert_true(len(indices[0]) == len(indices[1]))
            assert_true(len(indices[0]) == n_seeds * n_targets)
            for seed in seeds:
                assert_true(np.sum(indices[0] == seed) == n_targets)
            for target in targets:
                assert_true(np.sum(indices[1] == target) == n_seeds)
