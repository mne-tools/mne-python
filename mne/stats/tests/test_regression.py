# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from scipy import linalg, stats

import mne
from mne.datasets import sample
from mne.stats.regression import ols_epochs


def test_regression():
    """Test Ordinary Least Squares Regression
    """
    data_path = sample.data_path()
    raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
    tmin, tmax = -0.2, 0.5
    event_id = dict(aud_l=1, aud_r=2)

    # Setup for reading the raw data
    raw = mne.io.Raw(raw_fname, preload=True)
    events = mne.read_events(event_fname)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        baseline=(None, 0), preload=True)
    data = epochs.get_data()
    data_shape = data.shape
    data = np.reshape(data, (data_shape[0], -1))
    design_matrix = epochs.events[:, 1:]
    # makes the intercept
    design_matrix[:, 0] += 1
    # creates contrast: aud_l=0, aud_r=1
    design_matrix[:, 1] -= 1

    # do the regression: min |b - aX|
    betas, residuals, _, _ = linalg.lstsq(design_matrix, data)
    betas = betas.reshape(betas.shape[0], data_shape[1], data_shape[2])

    res = ols_epochs(epochs, design_matrix)

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

    stderrs = [rmse * unscaled_stderr for unscaled_stderr in unscaled_stderrs]
    stderrs = [stderr.reshape(data_shape[1], data_shape[2]) for
               stderr in stderrs]

    # test for equivalence
    assert_array_equal(res['stderr']['x0'].data, stderrs[0])
    assert_array_equal(res['stderr']['x1'].data, stderrs[1])

    # wald t-statistic
    ts = betas / np.array(stderrs)

    # test for equivalence
    assert_array_equal(res['t']['x0'].data, ts[0])
    assert_array_equal(res['t']['x1'].data, ts[1])

    # p-value
    p = [(1 - stats.t.cdf(np.abs(t), df)) * 2 for t in ts]
    assert_array_equal(res['p']['x0'].data, p[0])
    assert_array_equal(res['p']['x1'].data, p[1])
