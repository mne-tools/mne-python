# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from scipy import linalg, stats

import mne
from mne.stats.regression import ols_epochs


def test_regression():
    """Test Ordinary Least Squares Regression
    """
    # generate data
    np.random.seed(10)
    n_trials, n_chan, n_samples = 10, 30, 100
    data = np.random.randn(n_trials, n_chan, n_samples)
    design_matrix = np.ones((n_trials, 2))
    design_matrix[:, 1] = np.arange(n_trials) + 1

    # create an epochs object
    ch_names = ['CH %02d' % ch_no for ch_no in range(n_chan)]
    ch_types = ['mag'] * n_chan
    sfreq = 1000
    info = mne.io.array.create_info(ch_names, sfreq, ch_types)
    ep = mne.epochs.EpochsArray(data, info, np.zeros((n_trials, 3)))

    # do the regression: min |b - aX|
    y = np.reshape(data, (n_trials, -1))
    betas, residuals, _, _ = linalg.lstsq(design_matrix, data)

    res = ols_epochs(ep, design_matrix)

    # test for equivalence
    assert_array_equal(res['beta']['x0'].data, betas[0])
    assert_array_equal(res['beta']['x1'].data, betas[1])

    # degrees of freedom: observations - predictors
    n_rows, n_predictors = design_matrix.shape
    df = n_rows - n_predictors

    # root mean squared error
    rmse = np.sqrt(residuals / df)
    # model inverse covariance matrix
    design_invcov = linalg.inv(np.dot(design_matrix.T, design_matrix))
    unscaled_stderrs = np.sqrt(np.diag(design_invcov))

    stderrs = [unscaled_stderrs[0] * rmse]
    stderrs.append(unscaled_stderrs[1] * rmse)

    # test for equivalence
    assert_array_equal(res['stderr']['x0'].data, stderrs[0])
    assert_array_equal(res['stderr']['x1'].data, stderrs[1])

    # wald t-statistic
    ts = betas / stderrs

    # test for equivalence
    assert_array_equal(res['t']['x0'].data, ts[0])
    assert_array_equal(res['t']['x1'].data, ts[1])

    # p-value
    cdf = stats.t.cdf(np.abs(ts[0]), df)
    p = [(1 - cdf) * 2]
    cdf = stats.t.cdf(np.abs(ts[1]), df)
    p.append((1 - cdf) * 2)
    assert_array_equal(res['p']['x0'].data, p[0])
    assert_array_equal(res['p']['x1'].data, p[1])
