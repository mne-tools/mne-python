from itertools import product
from ..parametric import (f_twoway_rm, f_threshold_twoway_rm,
                          defaults_twoway_rm)
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

#  generated using this expression: `np.random.RandomState(42).randn(20, 6)`
test_data = np.array(
[[0.49671415, -0.1382643, 0.64768854, 1.52302986, -0.23415337, -0.23413696],
 [1.57921282, 0.76743473, -0.46947439, 0.54256004, -0.46341769, -0.46572975],
 [0.24196227, -1.91328024, -1.72491783, -0.56228753, -1.01283112, 0.31424733],
 [-0.90802408, -1.4123037, 1.46564877, -0.2257763, 0.0675282, -1.42474819],
 [-0.54438272, 0.11092259, -1.15099358, 0.37569802, -0.60063869, -0.29169375],
 [-0.60170661, 1.85227818, -0.01349722, -1.05771093, 0.82254491, -1.22084365],
 [0.2088636, -1.95967012, -1.32818605, 0.19686124, 0.73846658, 0.17136828],
 [-0.11564828, -0.3011037, -1.47852199, -0.71984421, -0.46063877, 1.05712223],
 [0.34361829, -1.76304016, 0.32408397, -0.38508228, -0.676922, 0.61167629],
 [1.03099952, 0.93128012, -0.83921752, -0.30921238, 0.33126343, 0.97554513],
 [-0.47917424, -0.18565898, -1.10633497, -1.19620662, 0.81252582, 1.35624003],
 [-0.07201012, 1.0035329, 0.36163603, -0.64511975, 0.36139561, 1.53803657],
 [-0.03582604, 1.56464366, -2.6197451, 0.8219025, 0.08704707, -0.29900735],
 [0.09176078, -1.98756891, -0.21967189, 0.35711257, 1.47789404, -0.51827022],
 [-0.8084936, -0.50175704, 0.91540212, 0.32875111, -0.5297602, 0.51326743],
 [0.09707755, 0.96864499, -0.70205309, -0.32766215, -0.39210815, -1.46351495],
 [0.29612028, 0.26105527, 0.00511346, -0.23458713, -1.41537074, -0.42064532],
 [-0.34271452, -0.80227727, -0.16128571, 0.40405086, 1.8861859, 0.17457781],
 [0.25755039, -0.07444592, -1.91877122, -0.02651388, 0.06023021, 2.46324211],
 [-0.19236096, 0.30154734, -0.03471177, -1.16867804, 1.14282281, 0.75193303]])


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

    # now check against external software results
    fvals, pvals = f_twoway_rm(test_data, [2, 3])

    assert_array_almost_equal(fvals,
        test_external['spss_fvals'], 3)
    assert_array_almost_equal(pvals,
        test_external['spss_pvals_uncorrected'], 3)
    assert_array_almost_equal(fvals,
        test_external['r_fvals'], 4)
    assert_array_almost_equal(pvals,
        test_external['r_pvals_uncorrected'], 3)

    _, pvals = f_twoway_rm(test_data, [2, 3], correction=True)
    assert_array_almost_equal(pvals, test_external['spss_pvals_corrected'], 3)
