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
    model : instance of sklearn model | string | None
        The model used in fitting inputs and outputs. This can be any
        sklearn-style model that contains a fit and predict method. Currently
        the only supported string is 'ridge', which uses the default
        :mod:`sklearn` ridge model. This is also what `None` defaults to.

    Attributes
    ----------
    coef_ : array, shape (n_features, n_delays)
        The coefficients from the model fit, reshaped for easy visualization.
        If you want the raw (1d) coefficients, access them from the estimator
        stored in `self.estimator_`.
    mask_pred_ : array of bool, shape (n_times,)
        A mask with True values corresponding to datapoints that were used
        in generating model predictions. Creating time delays necessitates that
        datapoints at edges of the input do not have matching time-lagged
        datapoints for all lags, and are thus removed in model fitting.


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
    """

    def __init__(self, tmin, tmax, sfreq,
                 feature_names=None, model=None):  # noqa: D102
        self.feature_names = feature_names
        self.sfreq = float(sfreq)
        self.tmin = tmin
        self.tmax = tmax
        self.model = 'ridge' if model is None else model

    def __repr__(self):  # noqa: D105
        s = "tmin, tmax : (%.3f, %.3f), " % (self.tmin, self.tmax)
        if isinstance(self.model, str):
            model = self.model
        else:
            model = str(type(self.model))
        s += "model : %s, " % str(model)
        if hasattr(self, 'delays_'):
            feats = self.feature_names
            if len(feats) == 1:
                s += "feature: %s, " % feats[0]
            else:
                s += "features : [%s, ..., %s], " % (feats[0], feats[-1])
            s += "fit: True"
        else:
            s += "fit: False"
        return "<ReceptiveField  |  %s>" % s

    def fit(self, X, y):
        """Fit a receptive field model.

        Parameters
        ----------
        X : array, shape ([n_epochs], n_features, n_times)
            The input features for the model.
        y : array, shape ([n_epochs], n_times)
            The output feature for the model.
        """
        from sklearn.linear_model import Ridge
        model_dict = dict(ridge=Ridge())
        if X.ndim == 1:
            raise ValueError('X must be shape ([n_epochs],'
                             ' n_features, n_times)')
        elif X.ndim == 2:
            # Ensure we have a 3D input
            X = X[np.newaxis, ...]
            if y.ndim == 1:
                y = y[np.newaxis, np.newaxis, :]  # Add epochs and outputs dim
            else:
                y = y[np.newaxis, :]  # Only add epochs dim
        elif X.ndim == 3 and y.ndim == 2:
            y = y[:, np.newaxis, :]  # Add an outputs dim
        else:
            raise ValueError('X must be of shape '
                             '([n_epochs], n_features, n_times)')
        if X.shape[-1] != y.shape[-1]:
            raise ValueError('X any y do not have the same n_times\n'
                             '%s != %s' % (X.shape[-1], y.shape[-1]))
        if X.shape[0] != y.shape[0]:
            raise ValueError('X any y do not have the same n_epochs\n'
                             '%s != %s' % (X.shape[0], y.shape[0]))
        # Initialize delays and model
        self.delays_ = _times_to_delays(self.tmin, self.tmax, self.sfreq)

        if isinstance(self.model, str):
            if self.model not in list(model_dict.keys()):
                raise ValueError('If string, model must be one'
                                 ' of %s' % model_dict.keys())
            model = model_dict[self.model]
        else:
            model = self.model
        self.estimator_ = model
        _check_estimator(self.estimator_)

        # Create input features
        n_epochs, n_feats, n_times = X.shape
        n_outputs = y.shape[1]
        X_del, msk = self._delay_for_fit(X)

        # Reshape so we have times x features
        X_del = X_del[..., ~msk]
        y = y[..., ~msk]

        # Convert to 2d by making epochs 1st axis and hstacking
        X_del = X_del.swapaxes(0, 1).reshape(n_epochs, -1, X_del.shape[-1])
        X_del = np.hstack(X_del)
        y = np.hstack(y)

        # Update feature names if we have none
        if self.feature_names is None:
            self.feature_names = ['feature_%s' % ii for ii in range(n_feats)]
        if len(self.feature_names) != n_feats:
            raise ValueError('n_features in X does not match feature names '
                             '(%s != %s)' % (n_feats, len(self.feature_names)))

        self.estimator_.fit(X_del.T, y.T)
        self.mask_remove = msk

        coefs = _get_final_est(self.estimator_).coef_
        self.coef_ = coefs.reshape([n_outputs, n_feats, len(self.delays_)])
        if n_outputs == 1:
            self.coef_ = self.coef_[0]

    def predict(self, X, y=None):
        """Make predictions using a receptive field.

        Parameters
        ----------
        X : array, shape ([n_epochs], n_channels, n_times)
            The input features for the model.
        y : None
            Used for sklearn compatibility.

        Returns
        -------
        y_pred : array, shape (n_times * n_epochs)
            The output predictions with time concatenated.
        """
        X_del, msk = self._delay_for_fit(X)
        X_del = X_del[..., ~msk]
        X_del = X_del.reshape([-1, X_del.shape[-1]]).T
        y_pred = self.estimator_.predict(X_del)
        self.mask_pred_ = ~msk
        return y_pred

    def _delay_for_fit(self, X):
        # First delay
        X_del = delay_time_series(X, self.tmin, self.tmax, self.sfreq)

        # Mask for removing edges later
        msk_helper = np.ones(X.shape[-1])
        msk_helper = delay_time_series(msk_helper, self.tmin,
                                       self.tmax, self.sfreq)
        msk = np.any(msk_helper == 0, axis=0)
        return X_del, msk


def delay_time_series(X, tmin, tmax, sfreq, newaxis=0):
    """Return a time-lagged input time series.

    Parameters
    ----------
    X : array, shape ([n_epochs], n_features, n_times)
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
        # Create zeros to populate w/ delays
        if ix_delay <= 0:
            i_slice = slice(-ix_delay, None)
        else:
            i_slice = slice(None, -ix_delay)
        delayed[ii, ..., i_slice] = np.roll(
            X, -ix_delay, axis=-1)[..., i_slice]

    # Now swapaxes so that the new axis is in the right place
    for ii in range(newaxis):
        delayed = delayed.swapaxes(ii, ii + 1)
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
