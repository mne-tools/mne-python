import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
                           assert_array_equal)
from scipy import stats
import pytest

from mne.stats import fdr_correction, bonferroni_correction


def test_bonferroni_pval_clip():
    """Test that p-values are never exceed 1.0."""
    p = (0.2, 0.9)
    _, p_corrected = bonferroni_correction(p)
    assert p_corrected.max() <= 1.0


def test_multi_pval_correction():
    """Test pval correction for multi comparison (FDR and Bonferroni)."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 1000, 10)
    X[:, :50, 0] += 4.0  # 50 significant tests
    alpha = 0.05

    T, pval = stats.ttest_1samp(X, 0)

    n_samples = X.shape[0]
    n_tests = X.size / n_samples
    thresh_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)

    reject_bonferroni, pval_bonferroni = bonferroni_correction(pval, alpha)
    thresh_bonferroni = stats.t.ppf(1.0 - alpha / n_tests, n_samples - 1)
    assert pval_bonferroni.ndim == 2
    assert reject_bonferroni.ndim == 2
    assert_allclose(pval_bonferroni, (pval * 10000).clip(max=1))
    reject_expected = pval_bonferroni < alpha
    assert_array_equal(reject_bonferroni, reject_expected)

    fwer = np.mean(reject_bonferroni)
    assert_almost_equal(fwer, alpha, 1)

    reject_fdr, pval_fdr = fdr_correction(pval, alpha=alpha, method='indep')
    assert pval_fdr.ndim == 2
    assert reject_fdr.ndim == 2
    thresh_fdr = np.min(np.abs(T)[reject_fdr])
    assert 0 <= (reject_fdr.sum() - 50) <= 50 * 1.05
    assert thresh_uncorrected <= thresh_fdr <= thresh_bonferroni
    pytest.raises(ValueError, fdr_correction, pval, alpha, method='blah')
    assert np.all(fdr_correction(pval, alpha=0)[0] == 0)

    reject_fdr, pval_fdr = fdr_correction(pval, alpha=alpha, method='negcorr')
    thresh_fdr = np.min(np.abs(T)[reject_fdr])
    assert 0 <= (reject_fdr.sum() - 50) <= 50 * 1.05
    assert thresh_uncorrected <= thresh_fdr <= thresh_bonferroni
