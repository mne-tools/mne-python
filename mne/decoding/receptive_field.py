# -*- coding: utf-8 -*-
# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>

# License: BSD (3-clause)

import numbers

import numpy as np
from scipy import linalg

from .base import get_coef, BaseEstimator, _check_estimator
from .time_delaying_ridge import TimeDelayingRidge
from ..fixes import is_regressor
from ..utils import _validate_type, verbose


class ReceptiveField(BaseEstimator):
    """Fit a receptive field model.

    This allows you to fit an encoding model (stimulus to brain) or a decoding
    model (brain to stimulus) using time-lagged input features (for example, a
    spectro- or spatio-temporal receptive field, or STRF).

    Parameters
    ----------
    tmin : float
        The starting lag, in seconds (or samples if ``sfreq`` == 1).
    tmax : float
        The ending lag, in seconds (or samples if ``sfreq`` == 1).
        Must be >= tmin.
    sfreq : float
        The sampling frequency used to convert times into samples.
    feature_names : array, shape (n_features,) | None
        Names for input features to the model. If None, feature names will
        be auto-generated from the shape of input data after running `fit`.
    estimator : instance of sklearn.base.BaseEstimator | float | None
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
    patterns : bool
        If True, inverse coefficients will be computed upon fitting using the
        covariance matrix of the inputs, and the cross-covariance of the
        inputs/outputs, according to [5]_. Defaults to False.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if CuPy
        is installed properly and ``estimator is None``.

        .. versionadded:: 0.18
    edge_correction : bool
        If True (default), correct the autocorrelation coefficients for
        non-zero delays for the fact that fewer samples are available.
        Disabling this speeds up performance at the cost of accuracy
        depending on the relationship between epoch length and model
        duration. Only used if ``estimator`` is float or None.

        .. versionadded:: 0.18
    verbose : bool, str, int, or None
        If not None, override default verbose level (see
        :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
        for more).

    Attributes
    ----------
    coef_ : array, shape ([n_outputs, ]n_features, n_delays)
        The coefficients from the model fit, reshaped for easy visualization.
        During :meth:`mne.decoding.ReceptiveField.fit`, if ``y`` has one
        dimension (time), the ``n_outputs`` dimension here is omitted.
    patterns_ : array, shape ([n_outputs, ]n_features, n_delays)
        If fit, the inverted coefficients from the model.
    delays_ : array, shape (n_delays,), dtype int
        The delays used to fit the model, in indices. To return the delays
        in seconds, use ``self.delays_ / self.sfreq``
    valid_samples_ : slice
        The rows to keep during model fitting after removing rows with
        missing values due to time delaying. This can be used to get an
        output equivalent to using :func:`numpy.convolve` or
        :func:`numpy.correlate` with ``mode='valid'``.

    See Also
    --------
    mne.decoding.TimeDelayingRidge

    Notes
    -----
    For a causal system, the encoding model will have significant
    non-zero values only at positive lags. In other words, lags point
    backward in time relative to the input, so positive lags correspond
    to previous input time samples, while negative lags correspond to
    future input time samples.

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

    .. [5] Haufe, S., Meinecke, F., Goergen, K., Daehne, S., Haynes, J.-D.,
           Blankertz, B., & Biessmann, F. (2014). On the interpretation of
           weight vectors of linear models in multivariate neuroimaging.
           NeuroImage, 87, 96-110. doi:10.1016/j.neuroimage.2013.10.067
    """

    @verbose
    def __init__(self, tmin, tmax, sfreq, feature_names=None, estimator=None,
                 fit_intercept=None, scoring='r2', patterns=False,
                 n_jobs=1, edge_correction=True, verbose=None):
        self.feature_names = feature_names
        self.sfreq = float(sfreq)
        self.tmin = tmin
        self.tmax = tmax
        self.estimator = 0. if estimator is None else estimator
        self.fit_intercept = fit_intercept
        self.scoring = scoring
        self.patterns = patterns
        self.n_jobs = n_jobs
        self.edge_correction = edge_correction
        self.verbose = verbose

    def __repr__(self):  # noqa: D105
        s = "tmin, tmax : (%.3f, %.3f), " % (self.tmin, self.tmax)
        estimator = self.estimator
        if not isinstance(estimator, str):
            estimator = type(self.estimator)
        s += "estimator : %s, " % (estimator,)
        if hasattr(self, 'coef_'):
            if self.feature_names is not None:
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
            X = _delay_time_series(X, self.tmin, self.tmax, self.sfreq,
                                   fill_mean=self.fit_intercept)
            X = _reshape_for_est(X)
            # Concat times + epochs
            if y is not None:
                y = y.reshape(-1, y.shape[-1], order='F')
        return X, y

    @verbose
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
        from sklearn.base import clone
        X, y, _, self._y_dim = self._check_dimensions(X, y)

        if self.tmin > self.tmax:
            raise ValueError('tmin (%s) must be at most tmax (%s)'
                             % (self.tmin, self.tmax))
        # Initialize delays
        self.delays_ = _times_to_delays(self.tmin, self.tmax, self.sfreq)

        # Define the slice that we should use in the middle
        self.valid_samples_ = _delays_to_slice(self.delays_)

        if isinstance(self.estimator, numbers.Real):
            if self.fit_intercept is None:
                self.fit_intercept = True
            estimator = TimeDelayingRidge(
                self.tmin, self.tmax, self.sfreq, alpha=self.estimator,
                fit_intercept=self.fit_intercept, n_jobs=self.n_jobs,
                edge_correction=self.edge_correction)
        elif is_regressor(self.estimator):
            estimator = clone(self.estimator)
            if self.fit_intercept is not None and \
                    estimator.fit_intercept != self.fit_intercept:
                raise ValueError(
                    'Estimator fit_intercept (%s) != initialization '
                    'fit_intercept (%s), initialize ReceptiveField with the '
                    'same fit_intercept value or use fit_intercept=None'
                    % (estimator.fit_intercept, self.fit_intercept))
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
        n_outputs = y.shape[-1]
        n_delays = len(self.delays_)

        # Update feature names if we have none
        if ((self.feature_names is not None) and
                (len(self.feature_names) != n_feats)):
            raise ValueError('n_features in X does not match feature names '
                             '(%s != %s)' % (n_feats, len(self.feature_names)))

        # Create input features
        X, y = self._delay_and_reshape(X, y)

        self.estimator_.fit(X, y)
        coef = get_coef(self.estimator_, 'coef_')  # (n_targets, n_features)
        shape = [n_feats, n_delays]
        if self._y_dim > 1:
            shape.insert(0, -1)
        self.coef_ = coef.reshape(shape)

        # Inverse-transform model weights
        if self.patterns:
            if isinstance(self.estimator_, TimeDelayingRidge):
                cov_ = self.estimator_.cov_ / float(n_times * n_epochs - 1)
                y = y.reshape(-1, y.shape[-1], order='F')
            else:
                X = X - X.mean(0, keepdims=True)
                cov_ = np.cov(X.T)
            del X

            # Inverse output covariance
            if y.ndim == 2 and y.shape[1] != 1:
                y = y - y.mean(0, keepdims=True)
                inv_Y = linalg.pinv(np.cov(y.T))
            else:
                inv_Y = 1. / float(n_times * n_epochs - 1)
            del y

            # Inverse coef according to Haufe's method
            # patterns has shape (n_feats * n_delays, n_outputs)
            coef = np.reshape(self.coef_, (n_feats * n_delays, n_outputs))
            patterns = cov_.dot(coef.dot(inv_Y))
            self.patterns_ = patterns.reshape(shape)

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
        # convert to sklearn and back
        pred_shape = X.shape[:-1]
        if self._y_dim > 1:
            pred_shape = pred_shape + (self.coef_.shape[0],)
        X, _ = self._delay_and_reshape(X)
        y_pred = self.estimator_.predict(X)
        y_pred = y_pred.reshape(pred_shape, order='F')
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

        This calls ``self.predict``, then masks the output of this
        and ``y` with ``self.mask_prediction_``. Finally, it passes
        this to a :mod:`sklearn.metrics` scorer.

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
                raise ValueError('X and y do not have the same n_times\n'
                                 '%s != %s' % (X.shape[0], y.shape[0]))
            if X.shape[1] != y.shape[1]:
                raise ValueError('X and y do not have the same n_epochs\n'
                                 '%s != %s' % (X.shape[1], y.shape[1]))
            if predict and y.shape[-1] != len(self.estimator_.coef_):
                raise ValueError('Number of outputs does not match'
                                 ' estimator coefficients dimensions')
        return X, y, X_dim, y_dim


def _delay_time_series(X, tmin, tmax, sfreq, fill_mean=False):
    """Return a time-lagged input time series.

    Parameters
    ----------
    X : array, shape (n_times[, n_epochs], n_features)
        The time series to delay. Must be 2D or 3D.
    tmin : int | float
        The starting lag.
    tmax : int | float
        The ending lag.
        Must be >= tmin.
    sfreq : int | float
        The sampling frequency of the series. Defaults to 1.0.
    fill_mean : bool
        If True, the fill value will be the mean along the time dimension
        of the feature, and each cropped and delayed segment of data
        will be shifted to have the same mean value (ensuring that mean
        subtraction works properly). If False, the fill value will be zero.

    Returns
    -------
    delayed : array, shape(n_times[, n_epochs][, n_features], n_delays)
        The delayed data. It has the same shape as X, with an extra dimension
        appended to the end.

    Examples
    --------
    >>> tmin, tmax = -0.1, 0.2
    >>> sfreq = 10.
    >>> x = np.arange(1, 6)
    >>> x_del = _delay_time_series(x, tmin, tmax, sfreq)
    >>> print(x_del)  # doctest:+SKIP
    [[2. 1. 0. 0.]
     [3. 2. 1. 0.]
     [4. 3. 2. 1.]
     [5. 4. 3. 2.]
     [0. 5. 4. 3.]]
    """
    _check_delayer_params(tmin, tmax, sfreq)
    delays = _times_to_delays(tmin, tmax, sfreq)
    # Iterate through indices and append
    delayed = np.zeros(X.shape + (len(delays),))
    if fill_mean:
        mean_value = X.mean(axis=0)
        if X.ndim == 3:
            mean_value = np.mean(mean_value, axis=0)
        delayed[:] = mean_value[:, np.newaxis]
    for ii, ix_delay in enumerate(delays):
        # Create zeros to populate w/ delays
        if ix_delay < 0:
            out = delayed[:ix_delay, ..., ii]
            use_X = X[-ix_delay:]
        elif ix_delay > 0:
            out = delayed[ix_delay:, ..., ii]
            use_X = X[:-ix_delay]
        else:  # == 0
            out = delayed[..., ii]
            use_X = X
        out[:] = use_X
        if fill_mean:
            out[:] += (mean_value - use_X.mean(axis=0))
    return delayed


def _times_to_delays(tmin, tmax, sfreq):
    """Convert a tmin/tmax in seconds to delays."""
    # Convert seconds to samples
    delays = np.arange(int(np.round(tmin * sfreq)),
                       int(np.round(tmax * sfreq) + 1))
    return delays


def _delays_to_slice(delays):
    """Find the slice to be taken in order to remove missing values."""
    # Negative values == cut off rows at the end
    min_delay = None if delays[-1] <= 0 else delays[-1]
    # Positive values == cut off rows at the end
    max_delay = None if delays[0] >= 0 else delays[0]
    return slice(min_delay, max_delay)


def _check_delayer_params(tmin, tmax, sfreq):
    """Check delayer input parameters. For future custom delay support."""
    _validate_type(sfreq, 'numeric', '`sfreq`')

    for tlim in (tmin, tmax):
        _validate_type(tlim, 'numeric', 'tmin/tmax')
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
    assert multioutput == 'raw_values'
    for this_y in (y_true, y):
        if this_y.ndim != 2:
            raise ValueError('inputs must be shape (samples, outputs), got %s'
                             % (this_y.shape,))
    return np.array([pearsonr(y_true[:, ii], y[:, ii])[0]
                     for ii in range(y.shape[-1])])


def _r2_score(y_true, y, multioutput=None):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y, multioutput=multioutput)


_SCORERS = {'r2': _r2_score, 'corrcoef': _corr_score}
