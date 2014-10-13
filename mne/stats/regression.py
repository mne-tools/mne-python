# Authors: Tal Linzen <linzen@nyu.edu>
#          Teon Brooks <teon@nyu.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

from collections import namedtuple
from inspect import isgenerator
import warnings

import numpy as np
from scipy import linalg, stats

from ..source_estimate import SourceEstimate
from ..epochs import _BaseEpochs
from ..evoked import Evoked, EvokedArray
from ..utils import logger
from ..io.pick import pick_types


def linear_regression(inst, design_matrix, names=None):
    """Fit Ordinary Least Squares regression (OLS)

    Parameters
    ----------
    inst : instance of Epochs | iterable of SourceEstimate
        The data to be regressed. Contains all the trials, sensors, and time
        points for the regression. For Source Estimates, accepts either a list
        or a generator object.
    design_matrix : ndarray, shape (n_observations, n_regressors)
        The regressors to be used. Must be a 2d array with as many rows as
        the first dimension of `data`. The first column of this matrix will
        typically consist of ones (intercept column).
    names : list-like | None
        Optional parameter to name the regressors. If provided, the length must
        correspond to the number of columns present in regressors
        (including the intercept, if present).
        Otherwise the default names are x0, x1, x2...xn for n regressors.

    Returns
    -------
    results : dict of namedtuple
        For each regressor (key) a namedtuple is provided with the
        following attributes:

            beta : regression coefficients
            stderr : standard error of regression coefficients
            t_val : t statistics (beta / stderr)
            p_val : two-sided p-value of t statistic under the t distribution
            mlog10_p_val : -log10 transformed p-value.

        The tuple members are numpy arrays. The shape of each numpy array is
        the shape of the data minus the first dimension; e.g., if the shape of
        the original data was (n_observations, n_channels, n_timepoints),
        then the shape of each of the arrays will be
        (n_channels, n_timepoints).
    """
    if names is None:
        names = ['x%i' % i for i in range(design_matrix.shape[1])]

    if isinstance(inst, _BaseEpochs):
        picks = pick_types(inst.info, meg=True, eeg=True, ref_meg=True,
                           stim=False, eog=False, ecg=False,
                           emg=False, exclude=['bads'])
        if [inst.ch_names[p] for p in picks] != inst.ch_names:
            warnings.warn('Fitting linear model to non-data or bad '
                          'channels. Check picking', UserWarning)
        msg = 'Fitting linear model to epochs'
        data = inst.get_data()
        out = EvokedArray(np.zeros(data.shape[1:]), inst.info, inst.tmin)
    elif isgenerator(inst):
        msg = 'Fitting linear model to source estimates (generator input)'
        out = next(inst)
        data = np.array([out.data] + [i.data for i in inst])
    elif isinstance(inst, list) and isinstance(inst[0], SourceEstimate):
        msg = 'Fitting linear model to source estimates (list input)'
        out = inst[0]
        data = np.array([i.data for i in inst])
    else:
        raise ValueError('Input must be epochs or iterable of source '
                         'estimates')
    logger.info(msg + ', (%s targets, %s regressors)' %
                (np.product(data.shape[1:]), len(names)))
    lm_params = _fit_lm(data, design_matrix, names)
    lm = namedtuple('lm', 'beta stderr t_val p_val mlog10_p_val')
    lm_fits = {}
    for name in names:
        parameters = [p[name] for p in lm_params]
        for ii, value in enumerate(parameters):
            out_ = out.copy()
            if isinstance(out_, SourceEstimate):
                out_._data[:] = value
            elif isinstance(out_, Evoked):
                out_.data[:] = value
            else:
                raise RuntimeError('Invalid container.')
            parameters[ii] = out_
        lm_fits[name] = lm(*parameters)
    logger.info('Done')
    return lm_fits


def _fit_lm(data, design_matrix, names):
    """Aux function"""
    n_samples = len(data)
    n_features = np.product(data.shape[1:])
    if design_matrix.ndim != 2:
        raise ValueError('Design matrix must be a 2d array')
    n_rows, n_predictors = design_matrix.shape

    if n_samples != n_rows:
        raise ValueError('Number of rows in design matrix must be equal '
                         'to number of observations')
    if n_predictors != len(names):
        raise ValueError('Number of regressor names must be equal to '
                         'number of column in design matrix')

    y = np.reshape(data, (n_samples, n_features))
    betas, resid_sum_squares, _, _ = linalg.lstsq(a=design_matrix, b=y)

    df = n_rows - n_predictors
    sqrt_noise_var = np.sqrt(resid_sum_squares / df).reshape(data.shape[1:])
    design_invcov = linalg.inv(np.dot(design_matrix.T, design_matrix))
    unscaled_stderrs = np.sqrt(np.diag(design_invcov))

    beta, stderr, t_val, p_val, mlog10_p_val = (dict() for _ in range(5))
    for x, unscaled_stderr, predictor in zip(betas, unscaled_stderrs, names):
        beta[predictor] = x.reshape(data.shape[1:])
        stderr[predictor] = sqrt_noise_var * unscaled_stderr
        t_val[predictor] = beta[predictor] / stderr[predictor]
        cdf = stats.t.cdf(np.abs(t_val[predictor]), df)
        p_val[predictor] = (1. - cdf) * 2.
        mlog10_p_val[predictor] = -np.log10(p_val[predictor])

    return beta, stderr, t_val, p_val, mlog10_p_val
