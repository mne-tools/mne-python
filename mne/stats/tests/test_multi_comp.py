import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import assert_true
from scipy import stats

from mne.stats import fdr_correction, bonferroni_correction


def test_multi_pval_correction():
    """Test pval correction for multi comparison (FDR and Bonferroni)
    """
    rng = np.random.RandomState(0)
    X = rng.randn(10, 10000)
    X[:, :50] += 4.0  # 50 significant tests
    alpha = 0.05

    T, pval = stats.ttest_1samp(X, 0)

    n_samples, n_tests = X.shape
    thresh_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)

    reject_bonferroni, pval_bonferroni = bonferroni_correction(pval, alpha)
    thresh_bonferroni = stats.t.ppf(1.0 - alpha / n_tests, n_samples - 1)

    fwer = np.mean(reject_bonferroni)
    assert_almost_equal(fwer, alpha, 1)

    reject_fdr, pval_fdr = fdr_correction(pval, alpha=alpha, method='indep')
    thresh_fdr = np.min(np.abs(T)[reject_fdr])
    assert_true(0 <= (reject_fdr.sum() - 50) <= 50 * 1.05)
    assert_true(thresh_uncorrected <= thresh_fdr <= thresh_bonferroni)

    reject_fdr, pval_fdr = fdr_correction(pval, alpha=alpha, method='negcorr')
    thresh_fdr = np.min(np.abs(T)[reject_fdr])
    assert_true(0 <= (reject_fdr.sum() - 50) <= 50 * 1.05)
    assert_true(thresh_uncorrected <= thresh_fdr <= thresh_bonferroni)
