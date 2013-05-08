from itertools import product
from ..parametric import f_twoway_rm, f_threshold_twoway_rm, \
    defaults_twoway_rm
from nose.tools import assert_raises, assert_true
from numpy.testing import assert_array_almost_equal

import numpy as np

# hardcoded external test results, manually transferred
test_external = {
    # SPSS, manually conducted analyis
    'spss_fvals': np.array([2.568, 0.240, 1.756]),
    'spss_pvals_uncorrected': np.array([0.126, 0.788, 0.186]),
    'spss_pvals_corrected': np.array([0.126, 0.784, 0.192]),
    # R 15.2
    # data generated using this code http://goo.gl/7UcKb
    'r_fvals': np.array([2.567619, 0.24006, 1.756380]),
    'r_pvals_uncorrected': np.array([0.12557, 0.78776, 0.1864])
}


def test_f_twoway_rm():
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
        fvals, pvals = f_twoway_rm(data, effects[n_levels], picks,
                                      correction=correction)
        assert_true((fvals >= 0).all())
        if pvals.any():
            assert_true(((0 <= pvals) & (1 >= pvals)).all())
        n_effects = len(defaults_twoway_rm['parse'][picks])
        assert_true(fvals.size == n_obs * n_effects)
        if n_effects == 1:  # test for principle of least surprise ...
            assert_true(fvals.ndim == 1)

        fvals_ = f_threshold_twoway_rm(n_subj, effects[n_levels], picks)
        assert_true((fvals_ >= 0).all())
        assert_true(fvals_.size == n_effects)

    data = np.random.random([n_subj, n_levels, 1])
    assert_raises(ValueError, f_twoway_rm, data, effects[n_levels],
                  effects='C', correction=correction)
    data = np.random.random([n_subj, n_levels, n_obs, 3])
    # check for dimension handling
    f_twoway_rm(data, effects[n_levels], picks, correction=correction)

    data = np.random.RandomState(42).randn(20, 6)
    fvals, pvals = f_twoway_rm(data, [2, 3])

    assert_array_almost_equal(fvals,
        test_external['spss_fvals'], 3)
    assert_array_almost_equal(pvals,
        test_external['spss_pvals_uncorrected'], 3)
    assert_array_almost_equal(fvals,
        test_external['r_fvals'], 4)
    assert_array_almost_equal(pvals,
        test_external['r_pvals_uncorrected'], 3)

    _, pvals = f_twoway_rm(data, [2, 3], correction=True)
    assert_array_almost_equal(pvals, test_external['spss_pvals_corrected'], 3)
