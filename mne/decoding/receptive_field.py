import numpy as np
from .base import _get_final_est, BaseEstimator, _check_estimator


class ReceptiveField(BaseEstimator):
    """Fit a receptive field model.

    This allows you to fit a model using time-lagged input features. For
    example, a spectro- or spatio-temporal receptive field (STRF).

    Parameters
    ----------
    tmin : int | float
        The starting lag. Negative values correspond to times in the past.
    tmax : int | float
        The ending lag. Positive values correspond to times in the future.
        Must be >= tmin.
    sfreq : float
        The sampling frequency used to convert times into samples.
    feature_names : array, shape (n_features,) | None
        Names for input features to the model. If None, feature names will
        be auto-generated from the shape of input data after running `fit`.
    estimator : instance of sklearn estimator | float | None
        The model used in fitting inputs and outputs. This can be any
        sklearn-style model that contains a fit and predict method. If a
        float is passed, it will be interpreted as the `alpha` parameter
        to be passed to a Ridge regression model. If `None`, then a Ridge
        regression model with an alpha of 0 will be used.
    scoring : object | None | str
        A function that takes the inputs `y_true` and `y_pred` and outputs
        a score between them. Or, a str type indicating the name of the
        scorer. If None, set to ``r2``.

    Attributes
    ----------
    ``coef_`` : array, shape (n_features, n_delays)
        The coefficients from the model fit, reshaped for easy visualization.
        If you want the raw (1d) coefficients, access them from the estimator
        stored in `self.estimator_`.
    ``times``: array, shape (n_delays), dtype float
        The delays used to fit the model, in seconds
    ``mask_fit_`` : array of bool, shape (n_times,)
        A mask with True values corresponding to datapoints that were used
        in fitting the model. Creating time delays necessitates that
        datapoints at edges of the input do not have matching time-lagged
        datapoints for all lags, and are thus removed in model fitting.
    ``mask_pred_`` : array of bool, shape (n_times_pred,)
        A mask with True values corresponding to samples that have matching
        time-lagged datapoints for all lags. This is created after calling
        `self.predict`. The mask is applied in `self.score`.
    ``scorer_`` : object
        scikit-learn Scorer instance.


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
                 estimator=None, scoring=None):  # noqa: D102
        self.feature_names = feature_names
        self.sfreq = float(sfreq)
        self.tmin = tmin
        self.tmax = tmax
        self.estimator = 0. if estimator is None else estimator
        self.scoring = scoring

        # Initialize delays
        self._delays = _times_to_delays(self.tmin, self.tmax, self.sfreq)
        self.times = self._delays / self.sfreq

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
            s += "scored"
            if callable(self.scorer_):
                s += " (%s)" % (self.scorer_.__name__)
        return "<ReceptiveField  |  %s>" % s

    def fit(self, X, y):
        """Fit a receptive field model.

        Parameters
        ----------
        X : array, shape (n_times, [n_epochs], n_features)
            The input features for the model.
        y : array, shape (n_times, [n_epochs], n_outputs)
            The output features for the model.
        """
        from sklearn.linear_model import Ridge
        from sklearn.base import is_regressor
        X, y = self._check_dimensions(X, y)

        if isinstance(self.estimator, (float, int)):
            estimator = Ridge(alpha=self.estimator)
        elif is_regressor(self.estimator):
            estimator = self.estimator
        else:
            raise ValueError('`estimator` must be a float or an instance'
                             ' of `BaseEstimator`.')
        self.estimator_ = estimator
        _check_estimator(self.estimator_)

        # Create input features
        n_times, n_epochs, n_feats = X.shape
        n_outputs = y.shape[-1]
        X_del, msk = self._delay_for_fit(X)
        n_delays = len(self._delays)

        # Remove timepoints that don't have lag data after delaying
        X_del = X_del[msk]
        y = y[msk]

        # Convert to 2d by making epochs 1st axis and vstacking
        X_del = X_del.reshape([-1, n_delays * n_feats], order='F')
        y = y.reshape([-1, n_outputs], order='F')

        # Update feature names if we have none
        if self.feature_names is None:
            self.feature_names = ['feature_%s' % ii for ii in range(n_feats)]
        if len(self.feature_names) != n_feats:
            raise ValueError('n_features in X does not match feature names '
                             '(%s != %s)' % (n_feats, len(self.feature_names)))

        self.estimator_.fit(X_del, y)
        self.mask_fit_ = msk

        coefs = _get_final_est(self.estimator_).coef_
        coefs = coefs.reshape([n_outputs, n_feats, len(self._delays)])
        self.coef_ = coefs.squeeze()

    def predict(self, X, y=None):
        """Generate predictions with a receptive field.

        Parameters
        ----------
        X : array, shape (n_times, [n_epochs], n_channels)
            The input features for the model.
        y : None
            Used for sklearn compatibility.

        Returns
        -------
        y_pred : array, shape (n_times * n_epochs)
            The output predictions with time concatenated.
        """
        if not hasattr(self, '_delays'):
            raise ValueError('Estimator has not been fit yet.')
        X, y = self._check_dimensions(X, y, predict=True)
        X_del, msk = self._delay_for_fit(X)
        X_del = X_del.reshape([-1, len(self._delays) * X.shape[-1]], order='F')
        y_pred = self.estimator_.predict(X_del)
        self.mask_predict_ = msk
        return y_pred

    def score(self, X, y):
        """Score predictions generated with a receptive field.

        This calls `self.predict`, then masks the output of this
        and `y` with `self.mask_prediction_`. Finally, it passes
        this to a `sklearn` scorer.

        Parameters
        ----------
        X : array, shape (n_times, [n_epochs], n_channels)
            The input features for the model.
        y : array, shape (n_times, [n_epochs], n_outputs)
            Used for sklearn compatibility.

        Returns
        -------
        scores : list of float, shape (n_outputs,)
            The scores estimated by ``scorer_`` for each output (e.g. mean
            R2 of ``predict(X)``).
        """
        from sklearn import metrics

        # Create our scoring object
        self.scorer_ = self.scoring
        if self.scorer_ is None:
            self.scorer_ = metrics.r2_score

        if isinstance(self.scorer_, str):
            if hasattr(metrics, '%s_score' % self.scorer_):
                self.scorer_ = getattr(metrics, '%s_score' % self.scorer_)
            else:
                raise KeyError("{0} scorer Doesn't appear to be valid a "
                               "scikit-learn scorer.".format(self.scorer_))

        # Generate predictions, then reshape so we can mask time
        X, y = self._check_dimensions(X, y, predict=True)
        y_pred = self.predict(X)
        n_outputs = y_pred.shape[-1]
        y_pred = y_pred.reshape(y.shape, order='F')
        y_pred = y_pred[self.mask_predict_]
        y = y[self.mask_predict_]

        # Re-vectorize and call scorer
        y = y.reshape([-1, n_outputs], order='F')
        y_pred = y_pred.reshape([-1, n_outputs], order='F')
        scores = self.scorer_(y, y_pred, multioutput='raw_values')
        return scores

    def _delay_for_fit(self, X):
        # First delay
        X_del = _delay_time_series(X, self.tmin, self.tmax, self.sfreq,
                                   newaxis=X.ndim)

        # Mask for removing edges later
        msk_helper = np.ones(X.shape[0])
        msk_helper = _delay_time_series(msk_helper, self.tmin, self.tmax,
                                        self.sfreq)
        msk = ~np.any(msk_helper == 0, axis=0)
        return X_del, msk

    def _check_dimensions(self, X, y, predict=False):
        if X.ndim == 1:
            raise ValueError('X must be shape (n_times, [n_epochs],'
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
                raise ValueError('y must be shape (n_times, [n_epochs],'
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
                             '(n_times, [n_epochs], n_features)')
        if y is None:
            # If y is None, then we don't need any more checks
            return X, y

        if X.shape[0] != y.shape[0]:
            raise ValueError('X any y do not have the same n_times\n'
                             '%s != %s' % (X.shape[0], y.shape[0]))
        if X.shape[1] != y.shape[1]:
            raise ValueError('X any y do not have the same n_epochs\n'
                             '%s != %s' % (X.shape[1], y.shape[1]))
        if predict is True:
            if y.shape[-1] != len(self.estimator_.coef_):
                raise ValueError('Number of outputs does not match'
                                 ' estimator coefficients dimensions')
        return X, y


def _delay_time_series(X, tmin, tmax, sfreq, newaxis=0, axis=0):
    """Return a time-lagged input time series.

    Parameters
    ----------
    X : array, shape (n_times, [n_epochs], n_features)
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
        else:
            pass
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
