import numpy as np
import numbers
from .base import get_coef, BaseEstimator, _check_estimator


class ReceptiveField(BaseEstimator):
    """Fit a receptive field model.

    This allows you to fit a model using time-lagged input features. For
    example, a spectro- or spatio-temporal receptive field (STRF).

    Parameters
    ----------
    tmin : int | float
        The starting lag, in seconds (or samples if ``sfreq`` == 1).
        Negative values correspond to times in the past.
    tmax : int | float
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
    scoring : ['r2', 'corrcoef']
        Defines how predictions will be scored. Currently must be one of
        'r2' (coefficient of determination) or 'corrcoef' (the correlation
        coefficient).

    Attributes
    ----------
    ``coef_`` : array, shape (n_outputs, n_features, n_delays)
        The coefficients from the model fit, reshaped for easy visualization.
        If you want the raw (1d) coefficients, access them from the estimator
        stored in ``self.estimator_``.
    ``delays_``: array, shape (n_delays,), dtype int
        The delays used to fit the model, in indices. To return the delays
        in seconds, use ``self.delays_ / self.sfreq``
    ``keep_samples_`` : slice
        The rows to keep during model fitting after removing rows with
        missing values due to time delaying.


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

    def __init__(self, tmin, tmax, sfreq, feature_names=None,
                 estimator=None, scoring='r2'):  # noqa: D102
        self.feature_names = feature_names
        self.sfreq = float(sfreq)
        self.tmin = tmin
        self.tmax = tmax
        self.estimator = 0. if estimator is None else estimator
        if scoring not in _SCORERS.keys():
            raise ValueError('scoring must be one of %s, got'
                             '%s ' % (_SCORERS.keys(), scoring))
        self.scoring = scoring

    def __repr__(self):  # noqa: D105
        s = "tmin, tmax : (%.3f, %.3f), " % (self.tmin, self.tmax)
        if isinstance(self.estimator, str):
            estimator = self.estimator
        else:
            estimator = str(type(self.estimator))
        s += "estimator : %s, " % str(estimator)
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

    def fit(self, X, y):
        """Fit a receptive field model.

        Parameters
        ----------
        X : array, shape (n_times[, n_epochs], n_features)
            The input features for the model.
        y : array, shape (n_times[, n_epochs], n_outputs)
            The output features for the model.
        """
        from sklearn.linear_model import Ridge
        from sklearn.base import is_regressor, clone
        X, y = self._check_dimensions(X, y)

        # Initialize delays
        self.delays_ = _times_to_delays(self.tmin, self.tmax, self.sfreq)

        # Define the slice that we should use in the middle
        self.keep_samples_ = _delays_to_slice(self.delays_)

        if isinstance(self.estimator, numbers.Real):
            estimator = Ridge(alpha=self.estimator)
        elif is_regressor(self.estimator):
            estimator = clone(self.estimator)
        else:
            raise ValueError('`estimator` must be a float or an instance'
                             ' of `BaseEstimator`,'
                             ' got type %s.' % type(self.estimator))
        self.estimator_ = estimator
        _check_estimator(self.estimator_)

        # Create input features
        n_times, n_epochs, n_feats = X.shape
        n_outputs = y.shape[2]
        X_del = _delay_time_series(X, self.tmin, self.tmax, self.sfreq,
                                   newaxis=X.ndim)

        # Remove timepoints that don't have lag data after delaying
        X_del = X_del[self.keep_samples_]
        y = y[self.keep_samples_]

        # Convert to 2d by making epochs 1st axis and vstacking
        X_del = X_del.reshape([-1, len(self.delays_) * n_feats], order='F')
        y = y.reshape([-1, n_outputs], order='F')

        # Update feature names if we have none
        if self.feature_names is None:
            self.feature_names = ['feature_%s' % ii for ii in range(n_feats)]
        if len(self.feature_names) != n_feats:
            raise ValueError('n_features in X does not match feature names '
                             '(%s != %s)' % (n_feats, len(self.feature_names)))

        self.estimator_.fit(X_del, y)

        coefs = get_coef(self.estimator_, 'coef_')
        coefs = coefs.reshape([n_outputs, n_feats, len(self.delays_)])
        if len(coefs) == 1:
            # Remove a singleton first dimension if only 1 output
            coefs = coefs[0]
        self.coef_ = coefs
        return self

    def predict(self, X, y=None):
        """Generate predictions with a receptive field.

        Parameters
        ----------
        X : array, shape (n_times[, n_epochs], n_channels)
            The input features for the model.

        Returns
        -------
        y_pred : array, shape (n_times * n_epochs)
            The output predictions with time concatenated.
        """
        if not hasattr(self, 'delays_'):
            raise ValueError('Estimator has not been fit yet.')
        X, y = self._check_dimensions(X, y, predict=True)
        X_del = _delay_time_series(X, self.tmin, self.tmax, self.sfreq,
                                   newaxis=X.ndim)
        # Convert nans to 0 since scikit-learn will error otherwise
        X_del[np.isnan(X_del)] = 0
        X_del = X_del.reshape([-1, len(self.delays_) * X.shape[-1]], order='F')
        y_pred = self.estimator_.predict(X_del)
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
        y : array, shape (n_times[, n_epochs], n_outputs)
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
        X, y = self._check_dimensions(X, y, predict=True)
        y_pred = self.predict(X)
        n_outputs = y_pred.shape[-1]
        y_pred = y_pred.reshape(y.shape, order='F')
        y_pred = y_pred[self.keep_samples_]
        y = y[self.keep_samples_]

        # Re-vectorize and call scorer
        y = y.reshape([-1, n_outputs], order='F')
        y_pred = y_pred.reshape([-1, n_outputs], order='F')
        scores = scorer_(y, y_pred, multioutput='raw_values')
        return scores

    def _check_dimensions(self, X, y, predict=False):
        if X.ndim == 1:
            raise ValueError('X must be shape (n_times[, n_epochs],'
                             ' n_features)')
        elif X.ndim == 2:
            # Ensure we have a 3D input by adding singleton epochs dimension
            X = X[:, np.newaxis, :]
            if y is None:
                pass
            elif y.ndim == 1:
                y = y[:, np.newaxis, np.newaxis]  # Add epochs and outputs dim
            elif y.ndim == 2:
                y = y[:, np.newaxis]  # Only add epochs dim
            else:
                raise ValueError('y must be shape (n_times[, n_epochs],'
                                 ' n_outputs')
        elif X.ndim == 3:
            if y is None:
                pass
            elif y.ndim == 2:
                y = y[:, :, np.newaxis]  # Add an outputs dim
            elif y.ndim != 3:
                raise ValueError('If X has 3 dimensions, '
                                 'y must be at least 2 dimensions')
        else:
            raise ValueError('X must be of shape '
                             '(n_times[, n_epochs], n_features)')
        if y is None:
            # If y is None, then we don't need any more checks
            pass
        elif X.shape[0] != y.shape[0]:
            raise ValueError('X any y do not have the same n_times\n'
                             '%s != %s' % (X.shape[0], y.shape[0]))
        elif X.shape[1] != y.shape[1]:
            raise ValueError('X any y do not have the same n_epochs\n'
                             '%s != %s' % (X.shape[1], y.shape[1]))
        elif predict is True:
            if y.shape[-1] != len(self.estimator_.coef_):
                raise ValueError('Number of outputs does not match'
                                 ' estimator coefficients dimensions')
        return X, y


def _delay_time_series(X, tmin, tmax, sfreq, newaxis=0, axis=0):
    """Return a time-lagged input time series.

    Parameters
    ----------
    X : array, shape (n_times[, n_epochs], n_features)
        The time series to delay.
    tmin : int | float
        The starting lag. Negative values correspond to times in the past.
    tmax : int | float
        The ending lag. Positive values correspond to times in the future.
        Must be >= tmin.
    sfreq : int | float
        The sampling frequency of the series. Defaults to 1.0.
    newaxis : int
        The axis in the output array that corresponds to time delays.
        Defaults to 0, for the first axis.
    axis : int
        The axis corresponding to the time dimension.

    Returns
    -------
    delayed: array, shape(..., n_delays, ...)
        The delayed data. It has the same shape as X, with an extra dimension
        created at ``newaxis`` that corresponds to each delay.
    """
    _check_delayer_params(tmin, tmax, sfreq)
    delays = _times_to_delays(tmin, tmax, sfreq)
    # XXX : add Vectorize=True parameter to switch on/off 2D output
    # Iterate through indices and append
    delayed = np.zeros((len(delays),) + X.shape)
    for ii, ix_delay in enumerate(delays):
        take = [slice(None)] * X.ndim
        put = [slice(None)] * X.ndim
        # Create zeros to populate w/ delays
        if ix_delay < 0:
            take[axis] = slice(None, ix_delay)
            put[axis] = slice(-ix_delay, None)
        elif ix_delay > 0:
            take[axis] = slice(ix_delay, None)
            put[axis] = slice(None, -ix_delay)
        delayed[ii][put] = X[take]

    # Now swapaxes so that the new axis is in the right place
    delayed = np.rollaxis(delayed, 0, newaxis + 1)
    return delayed


def _times_to_delays(tmin, tmax, sfreq):
    """Convert a tmin/tmax in seconds to delays."""
    # Convert seconds to samples
    delays = np.arange(np.round(tmin * sfreq),
                       np.round(tmax * sfreq) + 1).astype(int)
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


# Create a correlation scikit-learn-style scorer
def _corr_score(y_true, y, multioutput=None):
    from scipy.stats import pearsonr
    if any(ii.ndim != 2 for ii in [y_true, y]):
        raise ValueError('inputs must shape (samples, outputs)')
    return [pearsonr(y_true[:, ii], y[:, ii])[0] for ii in range(y.shape[-1])]


def _r2_score(y_true, y, multioutput=None):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y)

_SCORERS = {'r2': _r2_score, 'corrcoef': _corr_score}
