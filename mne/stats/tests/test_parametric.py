from itertools import product
from ..parametric import r_anova_twoway
from nose.tools import assert_raises

import numpy as np


def test_parametric_r_anova_twoway():
    """ Test 2-way anova """
    iter_params = product([4, 10], [2, 15], [4, 6, 8], ['A', 'B', 'A:B'],
        [1, 2], [False, True])
    for params in iter_params:
        n_subj, n_obs, n_levels, picks, n_jobs, correction = params
        data = np.random.random([n_subj, n_levels, n_obs])
        effects = {
            4: [2, 2],
            6: [2, 3],
            8: [2, 4]
        }
        r_anova_twoway(data, effects[n_levels], picks, n_jobs=n_jobs,
            correction=correction)
    data = np.random.random([n_subj, n_levels, 1])
    assert_raises(ValueError, r_anova_twoway, data, effects[n_levels],
                  n_jobs=2, correction=correction)
    assert_raises(ValueError, r_anova_twoway, data, effects[n_levels],
                  effects='C', n_jobs=2, correction=correction)
