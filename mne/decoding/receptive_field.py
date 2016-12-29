import numpy as np
from .base import get_coefs, BaseEstimator, _check_estimator
from ..utils import warn


class ReceptiveField(BaseEstimator):
    """Fit a receptive field model.

    Parameters
    ----------
    lags : array, shape (n_lags,), dtype int
        Time lags to be included in the model. They should be in samples
        (not seconds).
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
    coef_ : array, shape (n_features, n_lags)
        The coefficients from the model fit, reshaped for easy visualization.
        If you want the raw (2d) coefficients, use `mne.decoding.get_coefs`
        on `self.model`.
    mask_pred : array, dtype bool, shape (n_times,)
        A mask with True values corresponding to datapoints that were used
        in generating model predictions. Creating time lags necessitates that
        datapoints at the beginning / end of the input are not used.
    """

    def __init__(self, lags, feature_names=None, model=None):  # noqa: D102
        from sklearn.linear_model import Ridge
        model_dict = dict(ridge=Ridge())
        if isinstance(model, str):
            if model not in list(model_dict.keys()):
                raise ValueError('If string, model must be one'
                                 ' of %s' % model_dict.keys())
            model = model_dict[model]
        self.feature_names = feature_names

        lags = np.asarray(lags)
        if not np.issubdtype(lags.dtype, int):
            raise ValueError('lags must be dtype `int`, not %s' % lags.dtype)
        self.lags = lags
        self.model = Ridge() if model is None else model
        _check_estimator(self.model)

    def __repr__(self):  # noqa: D105
        str_features = self.feature_names
        s = "features : [%s, %s]" % (str_features[0], str_features[-1])
        s += ", lags : [%f, %f]" % (self.lags[0], self.lags[-1])
        s += ", model : %s" % str(self.model)
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
        n_feats = X.shape[1] if X.ndim == 3 else X.shape[0]
        X_del, msk = self._delay_for_fit(X)

        # Reshape so we have times x features
        X_del = X_del[..., ~msk]
        y = y[..., ~msk]

        if X_del.ndim == 4:
            # We have epochs, so hstack now
            n_ep = X_del.shape[1]
            X_del = X_del.swapaxes(0, 1).reshape(n_ep, -1, X_del.shape[-1])
            X_del = np.hstack(X_del)
            y = np.hstack(y)

        # Update feature names if we have none
        if self.feature_names is None:
            self.feature_names = ['feature_%s' % ii for ii in range(n_feats)]
        if len(self.feature_names) != n_feats:
            raise ValueError('n_features in X does not match feature names '
                             '(%s != %s)' % (n_feats, len(self.feature_names)))

        # Reshape so time is last dimension
        X_del = X_del.reshape([-1, X_del.shape[-1]])
        y = y.reshape([-1, X_del.shape[-1]])

        self.model.fit(X_del.T, y.T)
        self.msk_remove = msk

        coefs = get_coefs(self.model)
        self.coef_ = coefs.reshape([len(self.feature_names), len(self.lags)])

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
        y_pred = self.model.predict(X_del)
        self.mask_pred = ~msk
        return y_pred

    def _delay_for_fit(self, X):
        # First delay
        X_del = delay_time_series(X, self.lags, 1)

        # Mask for removing edges later
        msk_helper = np.ones(X.shape[-1])
        msk_helper = delay_time_series(msk_helper, self.lags, 1)
        msk = np.any(msk_helper == 0, axis=0)
        return X_del, msk


def delay_time_series(X, delays, sfreq=1., newaxis=0):
    """Return a time-lagged input time series.

    Parameters
    ----------
    X : array, shape ([n_epochs], n_features, n_times)
        The time series to delay.
    delays : array of floats, shape (n_delays,)
        The time (in seconds) of each delay for specifying
        pre-defined delays. Negative means time points in the past,
        positive means time points in the future.
    sfreq : int | float
        The sampling frequency of the series. Defaults to 1.0.
    newaxis : int
        The axis in the output array that corresponds to time lags.
        Defaults to 0, for the first axis.

    Returns
    -------
    delayed: array, shape(..., n_delays, ...)
        The delayed data. It has the same shape as X, with an extra dimension
        created at ``newaxis`` that corresponds to each delay.
    """
    delays, delays_ixs, sfreq = _check_delayer_params(delays, sfreq)

    # XXX : add Vectorize=True parameter to switch on/off 2D output
    # Iterate through indices and append
    delayed = np.zeros((len(delays),) + X.shape)
    for ii, ix_delay in enumerate(delays_ixs):
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


def _check_delayer_params(delays, sfreq):
    """Check delayer input parameters."""
    if not isinstance(sfreq, (int, float, np.int_)):
        raise ValueError('`sfreq` must be an integer or float')
    sfreq = float(sfreq)

    # XXX for use w/ future time window support.
    # Convert delays to a list of arrays
    delays = np.atleast_1d(delays)
    if delays.ndim != 1:
        raise ValueError('Delays must be shape (n_delays,)')

    # Remove duplicated ixs
    if delays.dtype not in [int, float]:
        raise ValueError('`delays` must be of type integer or float.')

    delays_ixs = (delays * sfreq).astype(int)
    if np.unique(delays_ixs).shape[0] != delays_ixs.shape[0]:
        warn('Converting delays to indices resulted in duplicates.')

    return delays, delays_ixs, sfreq
