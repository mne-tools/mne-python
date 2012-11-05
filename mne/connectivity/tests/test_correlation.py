import numpy as np
from numpy.testing import assert_array_almost_equal

from mne.connectivity import pearsonr
from scipy.stats import pearsonr as sp_pearsonr

def test_pearsonr():
    """Test computation of correlation coefficients"""
    n_sig, n_samples = 10, 10000
    n_corr = 5

    # generate correlated data
    data = np.random.randn(n_sig, n_samples)
    idx = np.tril_indices(n_sig, -1)
    corr_idx = np.random.permutation(range(len(idx[0])))[:n_corr]
    A = np.eye(n_sig)
    A[idx[0][corr_idx], idx[1][corr_idx]] = 0.8 * np.ones((n_corr))
    data_corr = np.dot(A, data)

    # the true correlation coefficients are
    corr_true = reduce(np.dot, [A, np.cov(data), A.T])
    corr_diag = np.diag(corr_true).copy()
    for i in range(n_sig):
        for j in range(n_sig):
            corr_true[i, j] /= np.sqrt(corr_diag[i] * corr_diag[j])

    corr, pval = pearsonr(data_corr)
    corr_idx, pval_idx = pearsonr(data_corr, idx)

    #rel_err = np.sum((corr_true[idx] - corr[idx]) ** 2)\
    #          / np.sum(corr_true[idx] ** 2)
    #assert(rel_err < 1e-3)
    assert_array_almost_equal(corr_true[idx], corr[idx], decimal=2)
    assert_array_almost_equal(corr_true[idx], corr_idx, decimal=2)
    assert_array_almost_equal(pval[idx], pval_idx)

