from itertools import product

import pytest
from numpy.testing import assert_array_almost_equal
import numpy as np

from mne.stats.parametric import (f_mway_rm, f_threshold_mway_rm,
                                  _map_effects)

# hardcoded external test results, manually transferred
test_external = {
    # SPSS, manually conducted analysis
    'spss_fvals': np.array([2.568, 0.240, 1.756]),
    'spss_pvals_uncorrected': np.array([0.126, 0.788, 0.186]),
    'spss_pvals_corrected': np.array([0.126, 0.784, 0.192]),
    # R 15.2
    # data generated using this code http://goo.gl/7UcKb
    'r_fvals': np.array([2.567619, 0.24006, 1.756380]),
    'r_pvals_uncorrected': np.array([0.12557, 0.78776, 0.1864]),
    # and https://gist.github.com/dengemann/5539403
    'r_fvals_3way': np.array([
        0.74783999999999995,   # A
        0.20895,               # B
        0.21378,               # A:B
        0.99404000000000003,   # C
        0.094039999999999999,  # A:C
        0.11685,               # B:C
        2.78749]),              # A:B:C
    'r_fvals_1way': np.array([0.67571999999999999])
}


def generate_data(n_subjects, n_conditions):
    """Generate testing data."""
    rng = np.random.RandomState(42)
    data = rng.randn(n_subjects * n_conditions).reshape(
        n_subjects, n_conditions)
    return data


def test_map_effects():
    """Test ANOVA effects parsing."""
    selection, names = _map_effects(n_factors=2, effects='A')
    assert names == ['A']

    selection, names = _map_effects(n_factors=2, effects=['A', 'A:B'])
    assert names == ['A', 'A:B']

    selection, names = _map_effects(n_factors=3, effects='A*B')
    assert names == ['A', 'B', 'A:B']

    # XXX this might be wrong?
    selection, names = _map_effects(n_factors=3, effects='A*C')
    assert names == ['A', 'B', 'A:B', 'C', 'A:C']

    pytest.raises(ValueError, _map_effects, n_factors=2, effects='C')

    pytest.raises(ValueError, _map_effects, n_factors=27, effects='all')


def test_f_twoway_rm():
    """Test 2-way anova."""
    rng = np.random.RandomState(42)
    iter_params = product([4, 10], [2, 15], [4, 6, 8],
                          ['A', 'B', 'A:B'],
                          [False, True])
    _effects = {
        4: [2, 2],
        6: [2, 3],
        8: [2, 4]
    }
    for params in iter_params:
        n_subj, n_obs, n_levels, effects, correction = params
        data = rng.random_sample([n_subj, n_levels, n_obs])
        fvals, pvals = f_mway_rm(data, _effects[n_levels], effects,
                                 correction=correction)
        assert (fvals >= 0).all()
        if pvals.any():
            assert ((0 <= pvals) & (1 >= pvals)).all()
        n_effects = len(_map_effects(n_subj, effects)[0])
        assert fvals.size == n_obs * n_effects
        if n_effects == 1:  # test for principle of least surprise ...
            assert fvals.ndim == 1

        fvals_ = f_threshold_mway_rm(n_subj, _effects[n_levels], effects)
        assert (fvals_ >= 0).all()
        assert fvals_.size == n_effects

    data = rng.random_sample([n_subj, n_levels, 1])
    pytest.raises(ValueError, f_mway_rm, data, _effects[n_levels],
                  effects='C', correction=correction)
    data = rng.random_sample([n_subj, n_levels, n_obs, 3])
    # check for dimension handling
    f_mway_rm(data, _effects[n_levels], effects, correction=correction)

    # now check against external software results
    test_data = generate_data(n_subjects=20, n_conditions=6)
    fvals, pvals = f_mway_rm(test_data, [2, 3])

    assert_array_almost_equal(fvals, test_external['spss_fvals'], 3)
    assert_array_almost_equal(pvals, test_external['spss_pvals_uncorrected'],
                              3)
    assert_array_almost_equal(fvals, test_external['r_fvals'], 4)
    assert_array_almost_equal(pvals, test_external['r_pvals_uncorrected'], 3)

    _, pvals = f_mway_rm(test_data, [2, 3], correction=True)
    assert_array_almost_equal(pvals, test_external['spss_pvals_corrected'], 3)

    test_data = generate_data(n_subjects=20, n_conditions=8)
    fvals, _ = f_mway_rm(test_data, [2, 2, 2])
    assert_array_almost_equal(fvals, test_external['r_fvals_3way'], 5)

    fvals, _ = f_mway_rm(test_data, [8], 'A')
    assert_array_almost_equal(fvals, test_external['r_fvals_1way'], 5)
