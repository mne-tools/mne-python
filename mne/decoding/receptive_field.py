# -*- coding: utf-8 -*-
# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>

# License: BSD (3-clause)

import numbers

import numpy as np

from .base import get_coef, BaseEstimator, _check_estimator
from .time_delaying_ridge import TimeDelayingRidge
from ..externals.six import string_types


class ReceptiveField(BaseEstimator):
    """Fit a receptive field model.

    This allows you to fit a model using time-lagged input features. For
    example, a spectro- or spatio-temporal receptive field (STRF).

    Parameters
    ----------
    tmin : float
        The starting lag, in seconds (or samples if ``sfreq`` == 1).
        Negative values correspond to times in the past.
    tmax : float
        The ending lag, in seconds (or samples if ``sfreq`` == 1).
        Positive values correspond to times in the future.
        Must be >= tmin.
    sfreq : float
        The sampling frequency used to convert times into samples.
    feature_names : array, shape (n_features,) | None
        Names for input features to the model. If None, feature names will
        be auto-generated from the shape of input data after running `fit`.
    estimator : instance of sklearn estimator | float | None
        The model used in fitting inputs and outputs. This can be any
        scikit-learn-style model that contains a fit and predict method. If a
        float is passed, it will be interpreted as the `alpha` parameter
        to be passed to a Ridge regression model. If `None`, then a Ridge
        regression model with an alpha of 0 will be used.
    fit_intercept : bool | None
        If True (default), the sample mean is removed before fitting.
        If ``estimator`` is a :class:`sklearn.base.BaseEstimator`,
        this must be None or match ``estimator.fit_intercept``.
    scoring : ['r2', 'corrcoef']
        Defines how predictions will be scored. Currently must be one of
        'r2' (coefficient of determination) or 'corrcoef' (the correlation
        coefficient).

    Attributes
    ----------
    ``coef_`` : array, shape ([n_outputs, ]n_features, n_delays)
        The coefficients from the model fit, reshaped for easy visualization.
        During :meth:`mne.decoding.ReceptiveField.fit`, if ``y`` has one
        dimension (time), the ``n_outputs`` dimension here is omitted.
    ``delays_``: array, shape (n_delays,), dtype int
        The delays used to fit the model, in indices. To return the delays
        in seconds, use ``self.delays_ / self.sfreq``
    ``valid_samples_`` : slice
        The rows to keep during model fitting after removing rows with
        missing values due to time delaying. This can be used to get an
        output equivalent to using :func:`numpy.convolve` or
        :func:`numpy.correlate` with ``mode='valid'``.

    See Also
    --------
    mne.decoding.TimeDelayingRidge

    References
    ----------
    .. [1] Theunissen, F. E. et al. Estimating spatio-temporal receptive
           fields of auditory and visual neurons from their responses to
           natural stimuli. Network 12, 289-316 (2001).

    .. [2] Willmore, B. & Smyth, D. Methods for first-order kernel
           estimation: simple-cell receptive fields from responses to
           natural scenes. Network 14, 553-77 (2003).

    .. [3] Crosse, M. J., Di Liberto, G. M., Bednar, A. & Lalor, E. C. (2016).
           The Multivariate Temporal Response Function (mTRF) Toolbox:
           A MATLAB Toolbox for Relating Neural Signals to Continuous Stimuli.
           Frontiers in Human Neuroscience 10, 604.
           doi:10.3389/fnhum.2016.00604

    .. [4] Holdgraf, C. R. et al. Rapid tuning shifts in human auditory cortex
           enhance speech intelligibility. Nature Communications,
           7, 13654 (2016). doi:10.1038/ncomms13654
    """

    def __init__(self, tmin, tmax, sfreq, feature_names=None, estimator=None,
                 fit_intercept=None, scoring='r2'):  # noqa: D102
        self.feature_names = feature_names
        self.sfreq = float(sfreq)
        self.tmin = tmin
        self.tmax = tmax
        self.estimator = 0. if estimator is None else estimator
        self.fit_intercept = fit_intercept
        self.scoring = scoring

    def __repr__(self):  # noqa: D105
        s = "tmin, tmax : (%.3f, %.3f), " % (self.tmin, self.tmax)
        estimator = self.estimator
        if not isinstance(estimator, string_types):
            estimator = type(self.estimator)
        s += "estimator : %s, " % (estimator,)
        if hasattr(self, 'coef_'):
            feats = self.feature_names
            if len(feats) == 1:
                s += "feature: %s, " % feats[0]
            else:
                s += "features : [%s, ..., %s], " % (feats[0], feats[-1])
            s += "fit: True"
        else:
            s += "fit: False"
        if hasattr(self, 'scores_'):
            s += "scored (%s)" % self.scoring
        return "<ReceptiveField  |  %s>" % s

    def _delay_and_reshape(self, X, y=None):
        """Delay and reshape the variables."""
        if not isinstance(self.estimator_, TimeDelayingRidge):
            # X is now shape (n_times, n_epochs, n_feats, n_delays)
            X_del = _delay_time_series(X, self.tmin, self.tmax, self.sfreq,
                                       fill_mean=self.fit_intercept)
        else:
            X_del = X[..., np.newaxis]

        X_del = _reshape_for_est(X_del)
        # Concat times + epochs
        if y is not None:
            y = y.reshape(-1, y.shape[-1], order='F')
        return X_del, y

    def fit(self, X, y):
        """Fit a receptive field model.

        Parameters
        ----------
        X : array, shape (n_times[, n_epochs], n_features)
            The input features for the model.
        y : array, shape (n_times[, n_epochs][, n_outputs])
            The output features for the model.

        Returns
        -------
        self : instance
            The instance so you can chain operations.
        """
        if self.scoring not in _SCORERS.keys():
            raise ValueError('scoring must be one of %s, got'
                             '%s ' % (sorted(_SCORERS.keys()), self.scoring))
        from sklearn.base import is_regressor, clone
        X, y, _, self._y_dim = self._check_dimensions(X, y)

        # Initialize delays
        self.delays_ = _times_to_delays(self.tmin, self.tmax, self.sfreq)

        # Define the slice that we should use in the middle
        self.valid_samples_ = _delays_to_slice(self.delays_)

        if isinstance(self.estimator, numbers.Real):
            if self.fit_intercept is None:
                self.fit_intercept = True
            estimator = TimeDelayingRidge(self.tmin, self.tmax, self.sfreq,
                                          alpha=self.estimator,
                                          fit_intercept=self.fit_intercept)
        elif is_regressor(self.estimator):
            estimator = clone(self.estimator)
            if self.fit_intercept is not None and \
                    estimator.fit_intercept != self.fit_intercept:
                raise ValueError(
                    'Estimator fit_intercept (%s) != initialization '
                    'fit_intercept (%s), initialize ReceptiveField with the '
                    'same fit_intercept value or use fit_intercept=None'
                    % (estimator.fit_intercept, self.fit_itercept))
            self.fit_intercept = estimator.fit_intercept
        else:
            raise ValueError('`estimator` must be a float or an instance'
                             ' of `BaseEstimator`,'
                             ' got type %s.' % type(self.estimator))
        self.estimator_ = estimator
        del estimator
        _check_estimator(self.estimator_)

        # Create input features
        n_times, n_epochs, n_feats = X.shape

        # Update feature names if we have none
        if self.feature_names is None:
            self.feature_names = ['feature_%s' % ii for ii in range(n_feats)]
        if len(self.feature_names) != n_feats:
            raise ValueError('n_features in X does not match feature names '
                             '(%s != %s)' % (n_feats, len(self.feature_names)))

        # Create input features
        # (eventually the FFT-based method could be made more memory efficient
        # by moving the padding to TimeDelayingRidge, which would need to be
        # made epochs-aware)

        # zero-pad if necessary
        if isinstance(self.estimator, TimeDelayingRidge):
            X = _pad_time_series(X, n_delays=len(self.delays_),
                                 fill_mean=self.fit_intercept)
            y = _pad_time_series(y, n_delays=len(self.delays_),
                                 fill_mean=self.fit_intercept)
        # convert to sklearn and back
        X, y = self._delay_and_reshape(X, y)
        self.estimator_.fit(X, y)
        del X, y
        coef = get_coef(self.estimator_, 'coef_')  # (n_targets, n_features)
        shape = [n_feats, len(self.delays_)]
        shape = ([-1] if self._y_dim > 1 else []) + shape
        self.coef_ = coef.reshape(shape)
        return self

    def predict(self, X):
        """Generate predictions with a receptive field.

        Parameters
        ----------
        X : array, shape (n_times[, n_epochs], n_channels)
            The input features for the model.

        Returns
        -------
        y_pred : array, shape (n_times[, n_epochs][, n_outputs])
            The output predictions. "Note that valid samples (those
            unaffected by edge artifacts during the time delaying step) can
            be obtained using ``y_pred[rf.valid_samples_]``.
        """
        if not hasattr(self, 'delays_'):
            raise ValueError('Estimator has not been fit yet.')
        X, _, X_dim = self._check_dimensions(X, None, predict=True)[:3]
        del _
        # zero-pad if necessary
        if isinstance(self.estimator, TimeDelayingRidge):
            X = _pad_time_series(X, n_delays=len(self.delays_),
                                 fill_mean=self.fit_intercept)
        # convert to sklearn and back
        pred_shape = X.shape[:-1]
        if self._y_dim > 1:
            pred_shape = pred_shape + (self.coef_.shape[0],)
        X, _ = self._delay_and_reshape(X)
        y_pred = self.estimator_.predict(X)
        y_pred = y_pred.reshape(pred_shape, order='F')
        # undo padding
        if isinstance(self.estimator, TimeDelayingRidge):
            y_pred = y_pred[:-(len(self.delays_) - 1)]
        shape = list(y_pred.shape)
        if X_dim <= 2:
            shape.pop(1)  # epochs
            extra = 0
        else:
            extra = 1
        shape = shape[:self._y_dim + extra]
        y_pred.shape = shape
        return y_pred

    def score(self, X, y):
        """Score predictions generated with a receptive field.

        This calls `self.predict`, then masks the output of this
        and `y` with `self.mask_prediction_`. Finally, it passes
        this to a `sklearn` scorer.

        Parameters
        ----------
        X : array, shape (n_times[, n_epochs], n_channels)
            The input features for the model.
        y : array, shape (n_times[, n_epochs][, n_outputs])
            Used for scikit-learn compatibility.

        Returns
        -------
        scores : list of float, shape (n_outputs,)
            The scores estimated by the model for each output (e.g. mean
            R2 of ``predict(X)``).
        """
        # Create our scoring object
        scorer_ = _SCORERS[self.scoring]

        # Generate predictions, then reshape so we can mask time
        X, y = self._check_dimensions(X, y, predict=True)[:2]
        n_times, n_epochs, n_outputs = y.shape
        y_pred = self.predict(X)
        y_pred = y_pred[self.valid_samples_]
        y = y[self.valid_samples_]

        # Re-vectorize and call scorer
        y = y.reshape([-1, n_outputs], order='F')
        y_pred = y_pred.reshape([-1, n_outputs], order='F')
        assert y.shape == y_pred.shape
        scores = scorer_(y, y_pred, multioutput='raw_values')
        return scores

    def _check_dimensions(self, X, y, predict=False):
        X_dim = X.ndim
        y_dim = y.ndim if y is not None else 0
        if X_dim == 2:
            # Ensure we have a 3D input by adding singleton epochs dimension
            X = X[:, np.newaxis, :]
            if y is not None:
                if y_dim == 1:
                    y = y[:, np.newaxis, np.newaxis]  # epochs, outputs
                elif y_dim == 2:
                    y = y[:, np.newaxis, :]  # epochs
                else:
                    raise ValueError('y must be shape (n_times[, n_epochs]'
                                     '[,n_outputs], got %s' % (y.shape,))
        elif X.ndim == 3:
            if y is not None:
                if y.ndim == 2:
                    y = y[:, :, np.newaxis]  # Add an outputs dim
                elif y.ndim != 3:
                    raise ValueError('If X has 3 dimensions, '
                                     'y must have 2 or 3 dimensions')
        else:
            raise ValueError('X must be shape (n_times[, n_epochs],'
                             ' n_features), got %s' % (X.shape,))
        if y is not None:
            if X.shape[0] != y.shape[0]:
                raise ValueError('X any y do not have the same n_times\n'
                                 '%s != %s' % (X.shape[0], y.shape[0]))
            if X.shape[1] != y.shape[1]:
                raise ValueError('X any y do not have the same n_epochs\n'
                                 '%s != %s' % (X.shape[1], y.shape[1]))
            if predict and y.shape[-1] != len(self.estimator_.coef_):
                    raise ValueError('Number of outputs does not match'
                                     ' estimator coefficients dimensions')
        return X, y, X_dim, y_dim


def _pad_time_series(X, n_delays, fill_mean=True):
    """Return a zero- or mean-padded input time series.

    Parameters
    ----------
    X : array, shape (n_times[, n_epochs], n_features)
        The time series to pad.
    n_delays : int
        The number of delays.
    fill_mean : bool
        If True, the fill value will be the mean along the time dimension
        of the feature. If False, the fill value will be zero.

    Returns
    -------
    padded : array, shape(n_padded[, n_epochs], n_features)
        The padded data, where ``n_padded = n_times + n_delays - 1``.
    """
    fill_value = 0
    if fill_mean:
        fill_value = np.mean(X, axis=0, keepdims=True)
        if X.ndim == 3:
            fill_value = np.mean(fill_value, axis=1, keepdims=True)
    X = np.pad(X, ((0, n_delays - 1),) + ((0, 0),) * (X.ndim - 1), 'constant')
    X[-(n_delays - 1):] = fill_value
    return X


def _delay_time_series(X, tmin, tmax, sfreq, fill_mean=False):
    """Return a time-lagged input time series.

    Parameters
    ----------
    X : array, shape (n_times[, n_epochs][, n_features])
        The time series to delay.
    tmin : int | float
        The starting lag. Negative values correspond to times in the past.
    tmax : int | float
        The ending lag. Positive values correspond to times in the future.
        Must be >= tmin.
    sfreq : int | float
        The sampling frequency of the series. Defaults to 1.0.
    fill_mean : bool
        If True, the fill value will be the mean along the time dimension
        of the feature. If False, the fill value will be zero.

    Returns
    -------
    delayed : array, shape(n_times[, n_epochs][, n_features], n_delays)
        The delayed data. It has the same shape as X, with an extra dimension
        appended to the end.

    Examples
    --------
    >>> tmin, tmax = -0.2, 0.1
    >>> sfreq = 10.
    >>> x = np.arange(1, 6)
    >>> x_del = _delay_time_series(x, tmin, tmax, sfreq)
    >>> print(x_del)
    [[ 0.  0.  1.  2.]
     [ 0.  1.  2.  3.]
     [ 1.  2.  3.  4.]
     [ 2.  3.  4.  5.]
     [ 3.  4.  5.  0.]]
    """
    _check_delayer_params(tmin, tmax, sfreq)
    delays = _times_to_delays(tmin, tmax, sfreq)
    # Iterate through indices and append
    delayed = np.zeros(X.shape + (len(delays),))
    if fill_mean:
        fill_value = X.mean(axis=0, keepdims=True)
        if X.ndim == 3:
            fill_value = np.mean(fill_value, axis=1, keepdims=True)
        delayed[...] = fill_value[..., np.newaxis]
    for ii, ix_delay in enumerate(delays):
        take = [slice(None)] * X.ndim
        put = [slice(None)] * X.ndim
        # Create zeros to populate w/ delays
        if ix_delay < 0:
            take[0] = slice(None, ix_delay)
            put[0] = slice(-ix_delay, None)
        elif ix_delay > 0:
            take[0] = slice(ix_delay, None)
            put[0] = slice(None, -ix_delay)
        delayed[put + [ii]] = X[take]

    return delayed


def _times_to_delays(tmin, tmax, sfreq):
    """Convert a tmin/tmax in seconds to delays."""
    # Convert seconds to samples
    delays = np.arange(int(np.round(tmin * sfreq)),
                       int(np.round(tmax * sfreq) + 1))
    return delays


def _delays_to_slice(delays):
    """Find the slice to be taken in order to remove missing values."""
    # Negative values == cut off rows at the beginning
    min_delay = np.clip(delays.min(), None, 0)
    min_delay = None if min_delay >= 0 else -1 * min_delay
    # Positive values == cut off rows at the end
    max_delay = np.clip(delays.max(), 0, None)
    max_delay = None if max_delay <= 0 else -1 * max_delay
    return slice(min_delay, max_delay)


def _check_delayer_params(tmin, tmax, sfreq):
    """Check delayer input parameters. For future custom delay support."""
    if not isinstance(sfreq, (int, float, np.int_)):
        raise ValueError('`sfreq` must be an integer or float')
    sfreq = float(sfreq)

    if not all([isinstance(ii, (int, float, np.int_))
                for ii in [tmin, tmax]]):
        raise ValueError('tmin/tmax must be an integer or float')
    if not tmin <= tmax:
        raise ValueError('tmin must be <= tmax')


def _reshape_for_est(X_del):
    """Convert X_del to a sklearn-compatible shape."""
    n_times, n_epochs, n_feats, n_delays = X_del.shape
    X_del = X_del.reshape(n_times, n_epochs, -1)  # concatenate feats
    X_del = X_del.reshape(n_times * n_epochs, -1, order='F')
    return X_del


# Create a correlation scikit-learn-style scorer
def _corr_score(y_true, y, multioutput=None):
    from scipy.stats import pearsonr
    for this_y in (y_true, y):
        if this_y.ndim != 2:
            raise ValueError('inputs must shape (samples, outputs), got %s'
                             % (this_y.shape,))
    return [pearsonr(y_true[:, ii], y[:, ii])[0] for ii in range(y.shape[-1])]


def _r2_score(y_true, y, multioutput=None):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y)

_SCORERS = {'r2': _r2_score, 'corrcoef': _corr_score}
