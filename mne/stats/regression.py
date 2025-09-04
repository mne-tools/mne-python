# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from collections import namedtuple
from inspect import isgenerator

import numpy as np
from scipy import linalg, sparse, stats

from .._fiff.pick import _picks_to_idx, pick_info, pick_types
from ..epochs import BaseEpochs
from ..evoked import Evoked, EvokedArray
from ..source_estimate import SourceEstimate
from ..utils import _reject_data_segments, fill_doc, logger, warn


def linear_regression(inst, design_matrix, names=None):
    """Fit Ordinary Least Squares (OLS) regression.

    Parameters
    ----------
    inst : instance of Epochs | iterable of SourceEstimate
        The data to be regressed. Contains all the trials, sensors, and time
        points for the regression. For Source Estimates, accepts either a list
        or a generator object.
    design_matrix : ndarray, shape (n_observations, n_regressors)
        The regressors to be used. Must be a 2d array with as many rows as
        the first dimension of the data. The first column of this matrix will
        typically consist of ones (intercept column).
    names : array-like | None
        Optional parameter to name the regressors (i.e., the columns in the
        design matrix). If provided, the length must correspond to the number
        of columns present in design matrix (including the intercept, if
        present). Otherwise, the default names are ``'x0'``, ``'x1'``,
        ``'x2', …, 'x(n-1)'`` for ``n`` regressors.

    Returns
    -------
    results : dict of namedtuple
        For each regressor (key), a namedtuple is provided with the
        following attributes:

            - ``beta`` : regression coefficients
            - ``stderr`` : standard error of regression coefficients
            - ``t_val`` : t statistics (``beta`` / ``stderr``)
            - ``p_val`` : two-sided p-value of t statistic under the t
              distribution
            - ``mlog10_p_val`` : -log₁₀-transformed p-value.

        The tuple members are numpy arrays. The shape of each numpy array is
        the shape of the data minus the first dimension; e.g., if the shape of
        the original data was ``(n_observations, n_channels, n_timepoints)``,
        then the shape of each of the arrays will be
        ``(n_channels, n_timepoints)``.
    """
    if names is None:
        names = [f"x{i}" for i in range(design_matrix.shape[1])]

    if isinstance(inst, BaseEpochs):
        picks = pick_types(
            inst.info,
            meg=True,
            eeg=True,
            ref_meg=True,
            stim=False,
            eog=False,
            ecg=False,
            emg=False,
            exclude=["bads"],
        )
        if [inst.ch_names[p] for p in picks] != inst.ch_names:
            warn("Fitting linear model to non-data or bad channels. Check picking")
        msg = "Fitting linear model to epochs"
        data = inst.get_data(copy=False)
        out = EvokedArray(np.zeros(data.shape[1:]), inst.info, inst.tmin)
    elif isgenerator(inst):
        msg = "Fitting linear model to source estimates (generator input)"
        out = next(inst)
        data = np.array([out.data] + [i.data for i in inst])
    elif isinstance(inst, list) and isinstance(inst[0], SourceEstimate):
        msg = "Fitting linear model to source estimates (list input)"
        out = inst[0]
        data = np.array([i.data for i in inst])
    else:
        raise ValueError("Input must be epochs or iterable of source estimates")
    logger.info(msg + f", ({np.prod(data.shape[1:])} targets, {len(names)} regressors)")
    lm_params = _fit_lm(data, design_matrix, names)
    lm = namedtuple("lm", "beta stderr t_val p_val mlog10_p_val")
    lm_fits = {}
    for name in names:
        parameters = [p[name] for p in lm_params]
        for ii, value in enumerate(parameters):
            out_ = out.copy()
            if not isinstance(out_, SourceEstimate | Evoked):
                raise RuntimeError("Invalid container.")
            out_._data[:] = value
            parameters[ii] = out_
        lm_fits[name] = lm(*parameters)
    logger.info("Done")
    return lm_fits


def _fit_lm(data, design_matrix, names):
    """Aux function."""
    n_samples = len(data)
    n_features = np.prod(data.shape[1:])
    if design_matrix.ndim != 2:
        raise ValueError("Design matrix must be a 2d array")
    n_rows, n_predictors = design_matrix.shape

    if n_samples != n_rows:
        raise ValueError(
            "Number of rows in design matrix must be equal to number of observations"
        )
    if n_predictors != len(names):
        raise ValueError(
            "Number of regressor names must be equal to "
            "number of column in design matrix"
        )

    y = np.reshape(data, (n_samples, n_features))
    betas, resid_sum_squares, _, _ = linalg.lstsq(a=design_matrix, b=y)

    df = n_rows - n_predictors
    sqrt_noise_var = np.sqrt(resid_sum_squares / df).reshape(data.shape[1:])
    design_invcov = linalg.inv(np.dot(design_matrix.T, design_matrix))
    unscaled_stderrs = np.sqrt(np.diag(design_invcov))
    tiny = np.finfo(np.float64).tiny
    beta, stderr, t_val, p_val, mlog10_p_val = (dict() for _ in range(5))
    for x, unscaled_stderr, predictor in zip(betas, unscaled_stderrs, names):
        beta[predictor] = x.reshape(data.shape[1:])
        stderr[predictor] = sqrt_noise_var * unscaled_stderr
        p_val[predictor] = np.empty_like(stderr[predictor])
        t_val[predictor] = np.empty_like(stderr[predictor])

        stderr_pos = stderr[predictor] > 0
        beta_pos = beta[predictor] > 0
        t_val[predictor][stderr_pos] = (
            beta[predictor][stderr_pos] / stderr[predictor][stderr_pos]
        )
        cdf = stats.t.cdf(np.abs(t_val[predictor][stderr_pos]), df)
        p_val[predictor][stderr_pos] = np.clip((1.0 - cdf) * 2.0, tiny, 1.0)
        # degenerate cases
        mask = ~stderr_pos & beta_pos
        t_val[predictor][mask] = np.inf * np.sign(beta[predictor][mask])
        p_val[predictor][mask] = tiny
        # could do NaN here, but hopefully this is safe enough
        mask = ~stderr_pos & ~beta_pos
        t_val[predictor][mask] = 0
        p_val[predictor][mask] = 1.0
        mlog10_p_val[predictor] = -np.log10(p_val[predictor])

    return beta, stderr, t_val, p_val, mlog10_p_val


@fill_doc
def linear_regression_raw(
    raw,
    events,
    event_id=None,
    tmin=-0.1,
    tmax=1,
    covariates=None,
    reject=None,
    flat=None,
    tstep=1.0,
    decim=1,
    picks=None,
    solver="cholesky",
):
    """Estimate regression-based evoked potentials/fields by linear modeling.

    This models the full M/EEG time course, including correction for
    overlapping potentials and allowing for continuous/scalar predictors.
    Internally, this constructs a predictor matrix X of size
    n_samples * (n_conds * window length), solving the linear system
    ``Y = bX`` and returning ``b`` as evoked-like time series split by
    condition. See :footcite:`SmithKutas2015`.

    Parameters
    ----------
    raw : instance of Raw
        A raw object. Note: be very careful about data that is not
        downsampled, as the resulting matrices can be enormous and easily
        overload your computer. Typically, 100 Hz sampling rate is
        appropriate - or using the decim keyword (see below).
    events : ndarray of int, shape (n_events, 3)
        An array where the first column corresponds to samples in raw
        and the last to integer codes in event_id.
    event_id : dict | None
        As in Epochs; a dictionary where the values may be integers or
        iterables of integers, corresponding to the 3rd column of
        events, and the keys are condition names.
        If None, uses all events in the events array.
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
        If dict-like (e.g., a pandas DataFrame), values have to be array-like
        and of the same length as the rows in ``events``. Keys correspond
        to additional event types/conditions to be estimated and are matched
        with the time points given by the first column of ``events``. If
        None, only binary events (from event_id) are used.
    reject : None | dict
        For cleaning raw data before the regression is performed: set up
        rejection parameters based on peak-to-peak amplitude in continuously
        selected subepochs. If None, no rejection is done.
        If dict, keys are types ('grad' | 'mag' | 'eeg' | 'eog' | 'ecg')
        and values are the maximal peak-to-peak values to select rejected
        epochs, e.g.::

            reject = dict(grad=4000e-12, # T / m (gradiometers)
                          mag=4e-11, # T (magnetometers)
                          eeg=40e-5, # V (EEG channels)
                          eog=250e-5 # V (EOG channels))

    flat : None | dict
        For cleaning raw data before the regression is performed: set up
        rejection parameters based on flatness of the signal. If None, no
        rejection is done. If a dict, keys are ('grad' | 'mag' |
        'eeg' | 'eog' | 'ecg') and values are minimal peak-to-peak values to
        select rejected epochs.
    tstep : float
        Length of windows for peak-to-peak detection for raw data cleaning.
    decim : int
        Decimate by choosing only a subsample of data points. Highly
        recommended for data recorded at high sampling frequencies, as
        otherwise huge intermediate matrices have to be created and inverted.
    %(picks_good_data)s
    solver : str | callable
        Either a function which takes as its inputs the sparse predictor
        matrix X and the observation matrix Y, and returns the coefficient
        matrix b; or a string.
        X is of shape (n_times, n_predictors * time_window_length).
        y is of shape (n_channels, n_times).
        If str, must be ``'cholesky'``, in which case the solver used is
        ``linalg.solve(dot(X.T, X), dot(X.T, y))``.

    Returns
    -------
    evokeds : dict
        A dict where the keys correspond to conditions and the values are
        Evoked objects with the ER[F/P]s. These can be used exactly like any
        other Evoked object, including e.g. plotting or statistics.

    References
    ----------
    .. footbibliography::
    """
    if isinstance(solver, str):
        if solver not in {"cholesky"}:
            raise ValueError(f"No such solver: {solver}")
        if solver == "cholesky":

            def solver(X, y):
                a = (X.T * X).toarray()  # dot product of sparse matrices
                return linalg.solve(
                    a, X.T * y, assume_a="pos", overwrite_a=True, overwrite_b=True
                ).T

    elif callable(solver):
        pass
    else:
        raise TypeError("The solver must be a str or a callable.")

    # build data
    data, info, events = _prepare_rerp_data(raw, events, picks=picks, decim=decim)

    if event_id is None:
        event_id = {str(v): v for v in set(events[:, 2])}

    # build predictors
    X, conds, cond_length, tmin_s, tmax_s = _prepare_rerp_preds(
        n_samples=data.shape[1],
        sfreq=info["sfreq"],
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        covariates=covariates,
    )

    # remove "empty" and contaminated data points
    X, data = _clean_rerp_input(X, data, reject, flat, decim, info, tstep)

    # solve linear system
    coefs = solver(X, data.T)
    if coefs.shape[0] != data.shape[0]:
        raise ValueError(
            f"solver output has unexcepted shape {coefs.shape}. Supply a "
            "function that returns coefficients in the form "
            "(n_targets, n_features), where "
            f"n_targets == n_channels == {data.shape[0]}."
        )

    # construct Evoked objects to be returned from output
    evokeds = _make_evokeds(coefs, conds, cond_length, tmin_s, tmax_s, info)

    return evokeds


def _prepare_rerp_data(raw, events, picks=None, decim=1):
    """Prepare events and data, primarily for `linear_regression_raw`."""
    picks = _picks_to_idx(raw.info, picks)
    info = pick_info(raw.info, picks)
    decim = int(decim)
    with info._unlock():
        info["sfreq"] /= decim
    data, times = raw[:]
    data = data[picks, ::decim]
    if len(set(events[:, 0])) < len(events[:, 0]):
        raise ValueError(
            "`events` contains duplicate time points. Make "
            "sure all entries in the first column of `events` "
            "are unique."
        )

    events = events.copy()
    events[:, 0] -= raw.first_samp
    events[:, 0] //= decim
    if len(set(events[:, 0])) < len(events[:, 0]):
        raise ValueError(
            "After decimating, `events` contains duplicate time "
            "points. This means some events are too closely "
            "spaced for the requested decimation factor. Choose "
            "different events, drop close events, or choose a "
            "different decimation factor."
        )

    return data, info, events


def _prepare_rerp_preds(
    n_samples, sfreq, events, event_id=None, tmin=-0.1, tmax=1, covariates=None
):
    """Build predictor matrix and metadata (e.g. condition time windows)."""
    conds = list(event_id)
    if covariates is not None:
        conds += list(covariates)

    # time windows (per event type) are converted to sample points from times
    # int(round()) to be safe and match Epochs constructor behavior
    if isinstance(tmin, float | int):
        tmin_s = {cond: int(round(tmin * sfreq)) for cond in conds}
    else:
        tmin_s = {cond: int(round(tmin.get(cond, -0.1) * sfreq)) for cond in conds}
    if isinstance(tmax, float | int):
        tmax_s = {cond: int(round(tmax * sfreq) + 1) for cond in conds}
    else:
        tmax_s = {cond: int(round(tmax.get(cond, 1.0) * sfreq)) + 1 for cond in conds}

    # Construct predictor matrix
    # We do this by creating one array per event type, shape (lags, samples)
    # (where lags depends on tmin/tmax and can be different for different
    # event types). Columns correspond to predictors, predictors correspond to
    # time lags. Thus, each array is mostly sparse, with one diagonal of 1s
    # per event (for binary predictors).

    cond_length = dict()
    xs = []
    for cond in conds:
        tmin_, tmax_ = tmin_s[cond], tmax_s[cond]
        n_lags = int(tmax_ - tmin_)  # width of matrix
        if cond in event_id:  # for binary predictors
            ids = (
                [event_id[cond]] if isinstance(event_id[cond], int) else event_id[cond]
            )
            onsets = -(events[np.isin(events[:, 2], ids), 0] + tmin_)
            values = np.ones((len(onsets), n_lags))

        else:  # for predictors from covariates, e.g. continuous ones
            covs = covariates[cond]
            if len(covs) != len(events):
                error = (
                    f"Condition {cond} from ``covariates`` is not the same length as "
                    "``events``"
                )
                raise ValueError(error)
            onsets = -(events[np.where(covs != 0), 0] + tmin_)[0]
            v = np.asarray(covs)[np.nonzero(covs)].astype(float)
            values = np.ones((len(onsets), n_lags)) * v[:, np.newaxis]

        cond_length[cond] = len(onsets)
        xs.append(sparse.dia_matrix((values, onsets), shape=(n_samples, n_lags)))

    return sparse.hstack(xs), conds, cond_length, tmin_s, tmax_s


def _clean_rerp_input(X, data, reject, flat, decim, info, tstep):
    """Remove empty and contaminated points from data & predictor matrices."""
    # find only those positions where at least one predictor isn't 0
    has_val = np.unique(X.nonzero()[0])

    # reject positions based on extreme steps in the data
    if reject is not None:
        _, inds = _reject_data_segments(
            data, reject, flat, decim=None, info=info, tstep=tstep
        )
        for t0, t1 in inds:
            has_val = np.setdiff1d(has_val, range(t0, t1))

    return X.tocsr()[has_val], data[:, has_val]


def _make_evokeds(coefs, conds, cond_length, tmin_s, tmax_s, info):
    """Create a dictionary of Evoked objects.

    These will be created from a coefs matrix and condition durations.
    """
    evokeds = dict()
    cumul = 0
    for cond in conds:
        tmin_, tmax_ = tmin_s[cond], tmax_s[cond]
        evokeds[cond] = EvokedArray(
            coefs[:, cumul : cumul + tmax_ - tmin_],
            info=info,
            comment=cond,
            tmin=tmin_ / float(info["sfreq"]),
            nave=cond_length[cond],
            kind="average",
        )  # nave and kind are technically incorrect
        cumul += tmax_ - tmin_
    return evokeds
