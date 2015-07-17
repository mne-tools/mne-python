# Authors: Tal Linzen <linzen@nyu.edu>
#          Teon Brooks <teon.brooks@gmail.com>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
# License: BSD (3-clause)

from collections import namedtuple
from inspect import isgenerator
import warnings

import numpy as np
from scipy import linalg

from ..source_estimate import SourceEstimate
from ..epochs import _BaseEpochs
from ..evoked import Evoked, EvokedArray
from ..utils import logger
from ..io.pick import pick_types


def get_fast_dot():
    try:
        from sklearn.utils.extmath import fast_dot
    except ImportError:
        fast_dot = np.dot
    return fast_dot


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
    from scipy import stats
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


def linear_regression_raw(raw, events, event_id=None,
                          tmin=-.1, tmax=1,
                          covariates=None,
                          reject=True, tstep=1.,
                          decim=1, picks=None,
                          solver='default'):
    """Estimate regression-based evoked potentials/fields by linear modelling
    of the full M/EEG time course, including correction for overlapping
    potentials and allowing for continuous/scalar predictors. Internally, this
    constructs a predictor matrix X of size n_samples * (n_conds * window
    length), solving the linear system Y = bX and returning b as evoked-like
    time series split by condition.

    See Smith, N. J., & Kutas, M. (2015). Regression-based estimation of ERP
    waveforms: II. Non-linear effects, overlap correction, and practical
    considerations. Psychophysiology, 52(2), 169-189.

    Parameters
    ----------
    raw : instance of Raw
        A raw object. Warning: be very careful about data that is not
        downsampled, as the resulting matrices can be enormous and easily
        overload your computer. Typically, 100 hz sampling rate is
        appropriate - or using the decim keyword (see below).
    events : array
        An n x 3 array, where the first column corresponds to samples in raw
        and the last to integer codes in event_id.
    event_id : dict
        As in Epochs; a dictionary where the values may be integers or
        list-like collections of integers, corresponding to the 3rd column of
        events, and the keys are condition names.
    tmin : float | dict
        If float, gives the lower limit (in seconds) for the time window for
        which all event types' effects are estimated. If a dict, can be used to
        specify time windows for specific event types: keys correspond to keys
        in event_id and/or covariates; for missing values, the default (-.1) is
        used.
    tmax : float | dict
        If float, gives the upper limit (in seconds) for the time window for
        which all event types' effects are estimated. If a dict, can be used to
        specify time windows for specific event types: keys correspond to keys
        in event_id and/or covariates; for missing values, the default (1.) is
        used.
    covariates : dict-like | None
        If dict-like, values have to be array-like and of the same length as
        the columns in ```events```. Keys correspond to additional event
        types/conditions to be estimated and are matched with the time points
        given by the first column of ```events```. If None, only binary events
        (from event_id) are used.
    reject : bool | dict
        Activate rejection parameters based on peak-to-peak amplitude in
        continuously selected subwindows. If reject is None, no rejection is
        done. If True, the following default is employed:
            reject = dict(grad=4000e-12, # T / m (gradiometers)
                          mag=4e-11, # T (magnetometers)
                          eeg=40e-5, # uV (EEG channels)
                          eog=250e-5 # uV (EOG channels)
                         )
        If dict, keys are types ('grad' | 'mag' | 'eeg' | 'eog' | 'ecg') and
        values are the maximal peak-to-peak values to select rejected epochs.
    tstep : float
        Length of windows for peak-to-peak detection.
    decim : int
        Decimate by choosing only a subsample of data points. Highly
        recommended for data recorded at high sampling frequencies, as
        otherwise huge intermediate matrices have to be created and inverted.
    picks : None | list
        List of indices of channels to be included. If None, defaults to all
        MEG and EEG channels.
    solver : str | function
        Either a function which takes as its inputs the predictor matrix X
        and the observation matrix Y, and returns the coefficient matrix b;
        or a string (for now, only 'defaults'), in which case the solver used
        is dot(scipy.linalg.pinv(dot(X.T, X)), dot(X.T, Y.T)).T.

    Returns
    -------
    ev_dict : dictionary
        A dictionary where the keys correspond to conditions and the values are
        Evoked objects with the rE[R/F]Ps. These can be used exactly like any
        other Evoked object, including e.g. plotting or statistics.
    """

    if solver == 'default':
        fast_dot = get_fast_dot()

        # inv is slightly (~10%) faster, but pinv seemingly more stable
        def solver(X, Y):
            return fast_dot(linalg.pinv(fast_dot(X.T, X)),
                            fast_dot(X.T, Y.T)).T

    # prepare raw and events
    if picks is None:
        picks = pick_types(raw.info, meg=True, eeg=True, ref_meg=True,
                           stim=False, eog=False, ecg=False,
                           emg=False, exclude=['bads'])
    raw = raw.pick_channels([raw.ch_names[ii] for ii in picks], copy=True)
    sfreq = raw.info["sfreq"] = raw.info["sfreq"] / decim
    data, times = raw[:]
    data = data[:, ::decim]
    times = times[::decim]
    events = events.copy()
    events[:, 0] -= raw.first_samp
    events[:, 0] /= decim

    conds = list(event_id.keys())
    if covariates is not None:
        conds += list(covariates.keys())

    # time windows (per event type)
    if isinstance(tmin, (float, int)):
        tmin = dict((cond, int(tmin * sfreq)) for cond in conds)
    else:
        tmin = dict((cond, int(tmin.get(cond, -.1) * sfreq))
                    for cond in conds)
    if isinstance(tmax, (float, int)):
        tmax = dict((cond, int((tmax * sfreq) + 1.)) for cond in conds)
    else:
        tmax = dict((cond, int((tmax.get(cond, 1.) * sfreq) + 1))
                    for cond in conds)

    # Construct predictor matrix
    # We do this by creating one array per event type, shape lags * samples
    # (where lags depends on tmin/tmax and can be different for different
    # event types). Columns correspond to predictors, predictors correspond to
    # time lags. Thus, each array is mostly sparse, with one diagonal of 1s
    # per event (for binary predictors).

    # This should probably be improved (including making it more robust to
    # high frequency data) by operating on sparse matrices. As-is, high
    # sampling rates plus long windows blow up the system due to the inversion
    # of massive matrices.
    # Furthermore, assigning to a preallocated array would be faster.

    cond_length = dict()
    pred_arrays = list()
    for cond in conds:

        # create the first row and column to be later used by toeplitz to build
        # the full predictor matrix
        tmin_, tmax_ = tmin[cond], tmax[cond]
        n_lags = int(tmax_ - tmin_)
        samples, lags = np.zeros(len(times)), np.zeros(n_lags)

        if cond in event_id.keys():  # for binary predictors
            ids = ([event_id[cond]] if isinstance(event_id[cond], int)
                   else event_id[cond])
            samples[events[np.in1d(events[:, 2], ids), 0] + int(tmin_)] = 1

        else:  # for predictors from covariates, e.g. continuous ones
            if len(covariates[cond]) != len(events):
                error = """Condition {} from ```covariates``` is
                        not the same length as ```events```""".format(cond)
                raise ValueError(error)
            for tx, v in zip(events[:, 0], covariates[cond]):
                samples[tx + int(tmin_)] = np.float(v)

        cond_length[cond] = len(np.nonzero(samples))

        # this is the magical part (thanks to Marijn van Vliet):
        # use toeplitz to construct series of diagonals
        pred_arrays.append(linalg.toeplitz(samples, lags))

    big_arr = np.hstack(pred_arrays).T
    # find only those positions where at least one predictor isn't 0
    has_val = np.asarray([len(np.nonzero(line)[0]) > 0
                          for line in big_arr.T])

    # additionally, reject positions based on extreme steps in the data
    if reject is not False:
        from ..utils import _reject_data_segments
        if not isinstance(reject, dict):
            reject = dict(grad=4000e-12,  # T / m (gradiometers)
                          mag=4e-11,  # T (magnetometers)
                          eeg=40e-5,  # uV (EEG channels)
                          eog=250e-5  # uV (EOG channels)
                          )

        _, inds = _reject_data_segments(data, reject, None, None,
                                        raw.info, tstep)
        for t0, t1 in inds:
            has_val[t0:t1] = False

    X = big_arr[:, has_val].T
#    X = np.vstack((X, np.ones(X.shape[1]))).T  # currently no intercept

    Y = data[:, has_val]

    # solve linear system
    coefs = solver(X, Y)

    # construct Evoked objects to be returned from output
    ev_dict = {}
    cum = 0
    for cond in conds:
        tmin_, tmax_ = tmin[cond], tmax[cond]
        ev_dict[cond] = EvokedArray(coefs[:, cum:cum + tmax_ - tmin_],
                                    raw.info, tmin_ / sfreq,
                                    comment=cond, nave=cond_length[cond],
                                    kind='mean')  # note that nave and kind are
        cum += tmax_ - tmin_                      # technically not correct

    return ev_dict
