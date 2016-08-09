"""Classes for transforming, preprocessing, and fitting encoding models."""
# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.de>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)
import numpy as np
from scipy.sparse import spmatrix, csr_matrix
from ..utils import warn
from ..decoding.mixin import TransformerMixin


class FeatureDelayer(object):
    """Creates delayed versions of an input stimulus.

    Parameters
    ----------
    delays: array of floats, shape (ix_delayays,)
        The time (in seconds) of each delay for specifying
        pre-defined delays. Negative means time points in the past,
        positive means time points in the future. Default is no delays.
    sfreq: float
        The sampling frequency of the series. Defaults to 1.0

    Attributes
    ----------
    ``names_`` : ndarray, shape (n_features * ix_delayays, 2)
        If fit, a collection of `(feature_name, delay)` pairs
        corresponding to columns of the output of `transform`.
    """
    def __init__(self, delays=0., sfreq=1.):
        delays, _, sfreq = _check_delayer_params(
            delays, sfreq)
        self.delays = delays
        self.sfreq = sfreq

    def fit(self, X, y=None):
        """Does nothing, only used for sklearn compatibility.

        Parameters
        ----------
        X : array
            Training data.
        y : array
            Target values.

        Returns
        -------
        self
        """
        return self

    def transform(self, X=None, y=None):
        """Create a time-lagged representation of X.

        Parameters
        ----------
        X : array-like, shape (n_times, n_features)
            The input data to be time-lagged and returned.

        Returns
        -------
        X_delayed : array, shape (n_times, n_features * ix_delayays)
            The delayed features of X.
        """
        X_delayed = delay_time_series(X, delays=self.delays,
                                      sfreq=self.sfreq)
        return X_delayed

    def fit_transform(self, X, y=None):
        """Create a time-lagged representation of X.

        Parameters
        ----------
        X : array-like, shape (n_times, n_features)
            The input data to be time-lagged and returned.

        Returns
        -------
        X_delayed : array, shape (n_times, n_features * ix_delayays)
            The delayed features of X.
        """
        return self.transform(X)


class DelaysVectorizer(TransformerMixin):
    """Transforms delays data to fit into a scikit-learn pipeline.

    Parameters
    ----------
    sfreq : int
        The sampling frequency for the features. Used for calculating times.

    Attributes
    ----------
    n_delays : int
        The number of delays.
    n_times : int
        The number of samples.
    n_features : int
        The number of features.
    """
    def __init__(self, sfreq=1.):
        self.sfreq = float(sfreq)

    def fit(self, X=None, y=None):
        """Does nothing, only used for sklearn compatibility.

        Parameters
        ----------
        X : array
            Training data.
        y : array
            Target values.

        Returns
        -------
        self : instance of DelaysVectorizer
            returns self
        """
        return self

    def transform(self, delayed, y=None):
        """Combine a delayed data array into a 2D array suitable for sklearn.

        Parameters
        ----------
        delayed : array, shape (n_samples, n_features, n_delays)
            The data to concatenate channels.
        y : array, shape (n_delays,)
            The amount (in samples) of each delay.

        Returns
        -------
        X : array, shape (n_samples, n_features * n_delays)
            The data concatenated over delays.
        """
        if not isinstance(delayed, np.ndarray):
            raise ValueError("delayed should be of type ndarray (got %s)."
                             % type(delayed))
        if delayed.ndim != 3:
            raise ValueError("delayed must be shape (n_samples, n_features, "
                             "n_delays)")

        n_samples, n_features, n_delays = delayed.shape
        X = delayed.reshape(n_samples, n_features * n_delays)

        # save attributes for inverse_transform
        self.n_samples_ = n_samples
        self.n_features_ = n_features
        self.n_delays_ = n_delays
        if y is not None:
            y = np.asarray(y)
            if y.ndim != 1:
                raise ValueError('y must be shape (n_delays,)')
            if y.shape[0] != delayed.shape[-1]:
                raise ValueError('Mismatch in length between delay values and '
                                 'last dimension of delayed.'' and delays')
            self.delays_ = y / self.sfreq
        return X

    def inverse_transform(self, X, y=None):
        """For each sample, reshape a feature vector into the original data shape.

        This can also be used to reshape coefficient vectors into
        shape (n_features, n_delays).

        Parameters
        ----------
        X : array, shape (n_samples, n_features * n_delays)
            The feature vector concatenated over delays
        y : None
            Not used. Only for sklearn compatibility.

        Returns
        -------
        delayed : array, shape (n_samples, n_features, n_delays)
            The original data
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)."
                             % type(X))

        return X.reshape(-1, self.n_features_, self.n_delays_)


class EventsBinarizer(object):
    """Create a continuous representation of event onsets.

    Parameters
    ----------
    n_times : int
        The total number of time samples in the output array.
    sfreq : float
        The sampling frequency used to convert times to sample indices.
        Defaults to 1.0
    sparse : bool
        Whether to output continuous events as a dense array or a sparse
        matrix. Defaults to False.
    """
    def __init__(self, n_times, sfreq=1., sparse=False):
        if not isinstance(n_times, int):
            raise ValueError('n_times must be an integer')
        if not isinstance(sfreq, (int, float)):
            raise ValueError('sfreq must be an integer or float')
        if not isinstance(sparse, bool):
            raise ValueError('sparse must be of type bool')
        self.n_times = n_times
        self.sfreq = sfreq
        self.sparse = sparse

    def fit(self, X=None, y=None):
        """Does nothing, only used for sklearn compatibility.

        Parameters
        ----------
        X : array
            Training data.
        y : array
            Target values.

        Returns
        -------
        self
        """
        return self

    def transform(self, X):
        """Binarize events and return as continuous data.

        Parameters
        ----------
        X : array, shape (n_events,)
            The event indices or onsets in seconds. `If self.sfreq == 1`,
            then these are assumed to be indices. If `self.sfreq != 1`, these
            should be onsets in seconds and will be multiplied by `sfreq`.

        Returns
        -------
        events_continuous : array, shape (n_times,)
            A binary array with 1s in the array corresponding to event onsets.
        """

        events_continuous = binarize_events(X, self.n_times, sfreq=self.sfreq,
                                            sparse=self.sparse)
        return events_continuous

    def fit_transform(self, X):
        """Does nothing, only used for sklearn compatibility.

        Parameters
        ----------
        X : array
            Training data.
        y : array
            Target values.

        Returns
        -------
        self
        """
        out = self.transform(X)
        return out


def binarize_events(events, n_times, sfreq=1., sparse=False):
    """Turn event times into a continuous array of events.

    Parameters
    ----------
    events : array, shape (n_events,)
        The event indices or onsets in seconds. `If self.sfreq == 1`,
        then these are assumed to be indices. If `self.sfreq != 1`, these
        should be onsets in seconds and will be multiplied by `sfreq`.
    n_times : int
        The total number of time samples in the output array.
    sfreq : float
        The sampling frequency used to convert times to sample indices.
        Defaults to 1.0.
    sparse : bool
        Whether to output continuous events as a dense array or a sparse
        matrix. Defaults to False.

    Returns
    -------
    events_continuous : array, shape (n_times,)
        A binary array with 1s in the array corresponding to event onsets.
    """
    events = np.asarray(events)
    if events.ndim > 1:
        raise ValueError("events must be shape (n_events,),"
                         " found shape %s" % str(events.shape))

    # Turn event_ixs from seconds to indices in case sfreq != 1
    events = (events * sfreq).astype(int)
    if events.max() > n_times:
        raise ValueError('Event index exceeds n_times')

    # Iterate through event types and create columns of event onsets.
    events_continuous = np.zeros(n_times)
    events_continuous[events] = 1

    if sparse is True:
        events_continuous = csr_matrix(events_continuous)
    return events_continuous


def delay_time_series(X, delays, sfreq=1.):
    """Return a time-lagged input time series.

    Parameters
    ----------
    X: array, shape (n_times, n_features)
        The time series to delay.
    delays: array of floats, shape (n_delays,)
        The time (in seconds) of each delay for specifying
        pre-defined delays. Negative means time points in the past,
        positive means time points in the future.
    sfreq: int | float
        The sampling frequency of the series. Defaults to 1.0.

    Returns
    -------
    delayed: array, shape(n_times, n_features * ix_delayays)
        The delayed matrix.
    """

    delays, delays_ixs, sfreq = _check_delayer_params(delays, sfreq)
    n_times, n_features = X.shape

    # Convert ts to dense if it is sparse
    if isinstance(X, spmatrix):
        # XXX : Implement sparse outputs (maybe only w/ vectorization)
        X = X.toarray()

    # XXX : add Vectorize=True parameter to switch on/off 2D output
    # Iterate through indices and append
    delayed = np.zeros(X.shape + (len(delays),))
    for ii, ix_delay in enumerate(delays_ixs):
        # Create zeros to populate w/ delays
        if ix_delay <= 0:
            i_slice = slice(-ix_delay, None)
        else:
            i_slice = slice(None, -ix_delay)
        delayed[i_slice, :, ii] = np.roll(X, -ix_delay, axis=0)[i_slice]

    return delayed


def _check_delayer_params(delays, sfreq):
    """Check delayer input parameters.
    """
    if not isinstance(sfreq, (int, float)):
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
