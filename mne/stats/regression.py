# Author: Tal Linzen <linzen@nyu.edu>
#
# License: Simplified BSD

import numpy as np
import scipy
from scipy import linalg

from mne.source_estimate import SourceEstimate

def ols(data, design_matrix, names):
    n_trials = data.shape[0]
    n_rows, n_predictors = design_matrix.shape

    if n_trials != n_rows:
        raise ValueError('Number of rows in design matrix must be equal '
                         'to number of observations')
    if n_predictors != len(names):
        raise ValueError('Number of predictor names must be equal to '
                         'number of predictors')

    y = np.reshape(data, (n_trials, -1))
    betas, resid_sum_squares, _, _ = linalg.lstsq(design_matrix, y)

    df = n_rows - n_predictors
    sqrt_noise_var = np.sqrt(resid_sum_squares / df)
    sqrt_noise_var = sqrt_noise_var.reshape(data.shape[1:])
    design_invcov = linalg.inv(np.dot(design_matrix.T, design_matrix))
    unscaled_stderrs = np.sqrt(np.diag(design_invcov))

    beta = {}
    stderr = {}
    t = {}
    p = {}
    for x, unscaled_stderr, pred in zip(betas, unscaled_stderrs, names):
        beta[pred] = x.reshape(data.shape[1:])
        stderr[pred] = sqrt_noise_var * unscaled_stderr
        t[pred] = beta[pred] / stderr[pred]
        cdf = scipy.stats.t.cdf(np.abs(t[pred]), df)
        p[pred] = (1 - cdf) * 2

    return dict(beta=beta, stderr=stderr, t=t, p=p)

def ols_epochs(epochs, design_matrix, names):
    evoked = epochs.average()
    data = epochs.get_data()
    ols_fit = ols(data, design_matrix, names)
    for v in ols_fit.values():
        for k in v.keys():
            ev = evoked.copy()
            ev.data = v[k]
            v[k] = ev
    return ols_fit

def ols_source_estimates(source_estimates, design_matrix, names):
    data = np.array([stc.data for stc in source_estimates])
    ols_fit = ols(data, design_matrix, names)
    s = source_estimates[0]
    for v in ols_fit.values():
        for k in v.keys():
            v[k] = SourceEstimate(v[k], vertices=s.vertno, tmin=s.tmin,
                                  tstep=s.tstep, subject=s.subject)
    return ols_fit
