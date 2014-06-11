# Author: Tal Linzen <linzen@nyu.edu>
#
# License: Simplified BSD

import numpy as np
import scipy
from scipy import linalg

from mne.source_estimate import SourceEstimate

def ols(data, design_matrix, names):
    """
    Parameters
    ----------
    data : instance of numpy.ndarray  (n_observations, ...)
        Measurements. Can have an arbitrary number of dimensions; a typical
        shape would be (n_observations, n_channels, n_timepoints).
    design_matrix : instance of numpy.ndarray  (n_observations, n_regressors)
        The regressors to be used. Must be a 2d array with as many rows as
        the first dimension of `data`. The first column of this matrix will
        typically consist of ones (intercept column).
    names : list-like | None
        Names of the regressors. The length must correspond to the
        number of columns present in regressors (including the intercept, if
        present).

    Returns
    -------
    results : dict
        Dictionary with the following keys:

            beta : regression coefficients
            stderr : standard error of regression coefficients
            t : t statistics (beta / stderr)
            p : two-sided p-value of t statistic under the t distribution

        The values are themselves dictionaries from regressor name to
        numpy arrays. The shape of each numpy array is the shape of the data
        minus the first dimension; e.g., if the shape of the original data was
        (n_observations, n_channels, n_timepoints), then the shape of each of
        the arrays will be (n_channels, n_timepoints).
    """

    n_trials = data.shape[0]
    if len(design_matrix.shape) != 2:
        raise ValueError('Design matrix must be a 2d array')
    n_rows, n_predictors = design_matrix.shape

    if n_trials != n_rows:
        raise ValueError('Number of rows in design matrix must be equal '
                         'to number of observations')
    if n_predictors != len(names):
        raise ValueError('Number of regressor names must be equal to '
                         'number of column in design matrix')

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
    """
    Parameters
    ----------
    epochs : instance of Epochs
    design_matrix : instance of numpy.ndarray  (n_observations, n_regressors)
        The regressors to be used. Must be a 2d array with as many rows as
        epochs in `epochs`. The first column of this matrix will typically
        consist of ones (intercept column).
        Note: use `epochs.selection` to align regressors with the remaining
        epochs if regressors were obtained from e.g. behavioral logs.
    names : list-like | None
        Names of the regressors. The length must correspond to the
        number of columns present in regressors (including the intercept, if
        present).

    Returns
    -------
    results : dict
        Dictionary with the following keys:

            beta : regression coefficients
            stderr : standard error of regression coefficients
            t : t statistics (beta / stderr)
            p : two-sided p-value of t statistic under the t distribution

        The values are themselves dictionaries from regressor name to
        Evoked objects. For instance, `results['t']['volume']` will be an
        Evoked object that represents the t statistic for the regressor
        'volume' in each channel at each timepoint.
    """
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
    """
    Parameters
    ----------
    source_estimates : list of SourceEstimate objects
    design_matrix : instance of numpy.ndarray  (n_observations, n_regressors)
        The regressors to be used. Must be a 2d array with the same number of
        rows as the number of objects in `source_estimates`. The first column
        of this matrix will typically consist of ones (intercept column).
    regressor_names : list-like | None
        Names of the regressors. The length must correspond to the
        number of columns present in regressors (including the intercept, if
        present).

    Returns
    -------
    results : dict
        Dictionary with the following keys:

            beta : regression coefficients
            stderr : standard error of regression coefficients
            t : t statistics (beta / stderr)
            p : two-sided p-value of t statistic under the t distribution

        The values are themselves dictionaries from regressor name to
        SourceEstimate objects. For instance, `results['t']['volume']` will be a
        SourceEstimate object that contains the t statistic for the regressor
        'volume' in each source at each timepoint.
    """
    data = np.array([stc.data for stc in source_estimates])
    ols_fit = ols(data, design_matrix, names)
    s = source_estimates[0]
    for v in ols_fit.values():
        for k in v.keys():
            v[k] = SourceEstimate(v[k], vertices=s.vertno, tmin=s.tmin,
                                  tstep=s.tstep, subject=s.subject)
    return ols_fit
