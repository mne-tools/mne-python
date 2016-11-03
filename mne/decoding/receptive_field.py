import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from .base import get_coefs


class ReceptiveField(object):
    """Fit a receptive field model.

    Parameters
    ----------
    stim_features : array, shape (n_features,)
        Names for the input features to the model.
    lags : array, shape (n_lags)
        Time lags to be included in the model. This will be combined
        with `sfreq` to convert these values into indices in the data.
        If `sfreq` == 1, then these should be indices. If not, then
        time lags should be in seconds.
    sfreq : float
        The sampling frequency of the data.
    model : instance of sklearn model | None
        The model used in fitting inputs and outputs. This can be any
        sklearn-style model that contains a fit and predict method.
    """
    def __init__(self, stim_features, lags, sfreq=1., model=None):
        self.stim_features = stim_features
        self.lags = lags
        self.model = Ridge() if model is None else model
        self.sfreq = sfreq

    def fit(self, X, y):
        """Fit a receptive field model.

        Parameters
        ----------
        X : array, shape ([n_epochs], n_channels, n_times)
            The input features for the model.
        y : array, shape ([n_epochs], n_times)
            The output feature for the model.
        """
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

        # Reshape so time is last dimension
        X_del = X_del.reshape([-1, X_del.shape[-1]])
        y = y.reshape([-1, X_del.shape[-1]])

        self.model.fit(X_del.T, y.T)
        self.msk_remove = msk

    def transform(self, X=None, y=None):
        """Does nothing, used for sklearn compatibility.

        Parameters
        ----------
        X : None
            Used for sklearn compatibility.
        y : None
            Used for sklearn compatibility.
        """
        return self

    def fit_transform(self, X=None, y=None):
        """Does nothing, used for sklearn compatibility.

        Parameters
        ----------
        X : None
            Used for sklearn compatibility.
        y : None
            Used for sklearn compatibility.
        """
        return self

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
        return y_pred

    def _delay_for_fit(self, X):
        # First delay
        X_del = delay_time_series(
            X, self.lags, self.sfreq)

        # Mask for removing edges later
        msk_helper = np.ones(X.shape[-1])
        msk_helper = delay_time_series(msk_helper, self.lags, self.sfreq)
        msk = (msk_helper == 0).any(axis=0)
        return X_del, msk

    def plot_coefs(self, ax=None):
        """Plot the fitted model coefficients.

        Parameters
        ----------
        ax : instance of matplotlib axis object | None
            The axis to use for plotting.

        Returns
        -------
        ax : instance of matplotlib axis object
            The axis used for plotting.
        """
        coefs = self.coefs

        # Plot
        if ax is None:
            fig, ax = plt.subplots()
        ax.pcolormesh(self.lags, self.stim_features,
                      coefs, cmap='coolwarm')
        _ = plt.setp(ax,
                     xlim=[self.lags.min(), self.lags.max()],
                     ylim=[self.stim_features.min(), self.stim_features.max()])
        return ax

    @property
    def coefs(self):
        """Return the model coefficients."""
        coefs = get_coefs(self.model)
        coefs = coefs.reshape([len(self.stim_features), len(self.lags)])
        return coefs


def delay_time_series(X, delays, sfreq=1.):
    """Return a time-lagged input time series.

    Parameters
    ----------
    X: array, shape ([n_epochs], n_features, n_times)
        The time series to delay.
    delays: array of floats, shape (n_delays,)
        The time (in seconds) of each delay for specifying
        pre-defined delays. Negative means time points in the past,
        positive means time points in the future.
    sfreq: int | float
        The sampling frequency of the series. Defaults to 1.0.

    Returns
    -------
    delayed: array, shape((n_delays,) + X.shape)
        The delayed data. An extra dimension is created (now the 1st dimension)
        corresponding to each delay.
    """

    delays, delays_ixs, sfreq = _check_delayer_params(delays, sfreq)

    # XXX : add Vectorize=True parameter to switch on/off 2D output
    # Iterate through indices and append
    delayed = np.zeros((len(delays),) + X.shape)  # Adding delays as 1st dim.
    for ii, ix_delay in enumerate(delays_ixs):
        # Create zeros to populate w/ delays
        if ix_delay <= 0:
            i_slice = slice(-ix_delay, None)
        else:
            i_slice = slice(None, -ix_delay)
        delayed[ii, ..., i_slice] = np.roll(
            X, -ix_delay, axis=-1)[..., i_slice]

    return delayed


def _check_delayer_params(delays, sfreq):
    """Check delayer input parameters.
    """
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
