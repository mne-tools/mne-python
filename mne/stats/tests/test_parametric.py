from itertools import product
from ..parametric import r_anova_twoway, f_threshold_twoway, defaults
from nose.tools import assert_raises, assert_true

import numpy as np


def test_parametric_r_anova_twoway():
    """ Test 2-way anova """
    iter_params = product([4, 10], [2, 15], [4, 6, 8], ['A', 'B', 'A:B'],
        [False, True])
    for params in iter_params:
        n_subj, n_obs, n_levels, picks, correction = params
        data = np.random.random([n_subj, n_levels, n_obs])
        effects = {
            4: [2, 2],
            6: [2, 3],
            8: [2, 4]
        }
        fvals, pvals = r_anova_twoway(data, effects[n_levels], picks,
                                      correction=correction)
        assert_true((fvals >= 0).all())
        if pvals.any():
            assert_true(((0 <= pvals) & (1 >= pvals)).all())
        n_effects = len(defaults['parse'][picks])
        assert_true(fvals.size == n_obs * n_effects)
        if n_effects == 1:  # test for principle of least surprise ...
            assert_true(fvals.ndim == 1)

        fvals_ = f_threshold_twoway(n_subj, effects[n_levels], picks)
        assert_true((fvals_ >= 0).all())
        assert_true(fvals_.size == n_effects)

    data = np.random.random([n_subj, n_levels, 1])
    assert_raises(ValueError, r_anova_twoway, data, effects[n_levels],
                  effects='C', correction=correction)
    data = np.random.random([n_subj, n_levels, n_obs, 3])
    # check for dimension handling
    r_anova_twoway(data, effects[n_levels], picks, correction=correction)
