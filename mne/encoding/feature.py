"""A collection of classes for transforming, preprocessing, and fitting."""
import numpy as np
from scipy import sparse
from ..utils import warn, _reject_data_segments


class DataDelayer(object):
    """Transformer for creating delayed versions of an input stimulus.

    Parameters
    ----------
    time_window : array, shape (2,)
        Specifies a minimum/maximum delay in seconds. In this case, all
        delays between `min(time_window)` and `max(time_window)`
        will be calculated using `sfreq`.
    delays: array of floats, shape (n_delays,)
        The time (in seconds) of each delay for specifying
        pre-defined delays. Negative means timepoints in the past,
        positive means timepoints in the future.
    sfreq: float
        The sampling frequency of the series. Defaults to 1.
    """
    def __init__(self, delays=None, time_window=None, sfreq=1.):
        # Check if we need to creafte delays ourselves
        if time_window is not None:
            if sfreq is None:
                raise ValueError('If time_window given, must give sfreq.')
            if delays is not None:
                raise ValueError('Delays must be None if time_window is given')
            tmin, tmax = time_window
            start_idx = int(round(tmin * sfreq))
            delays = np.arange(start_idx, int(round(tmax * sfreq)) + 1) / sfreq
        elif delays is None:
            raise ValueError('One of `time_window` or `delays` must be given.')
        self.delays = np.asarray(delays)
        self.sfreq = float(sfreq)
        self.feature_type = 'delay'

    def fit(self, X, y=None):
        """Create a time-lagged representation of X.

        Parameters
        ----------
        X : array-like, shape (n_times, n_features)
            The input data to be time-lagged and returned.
        """
        X_del, names = _delay_timeseries(X, self.delays, sfreq=self.sfreq)
        self.X_delayed_ = X_del
        self.names_ = names

    def transform(self, X=None, y=None):
        """See `fit` method."""
        return self.X_delayed_

    def fit_transform(self, X, y=None):
        """See `fit` method."""
        self.fit(X)
        return self.transform()


class EventsBinarizer(object):
    """Create a continuous-representation of event onsets.

    Parameters
    ----------
    n_times : int
        The total number of times in the output array.
    sfreq : float
        The sampling frequency used to convert times to sample indices.
        Defaults to 1.
    sparse : bool
        Whether to output continuous events as a dense array or a sparse
        matrix. Defaults to False.
    """
    def __init__(self, n_times, sfreq=1., sparse=False):
        self.n_times = n_times
        self.sfreq = sfreq
        self.feature_type = 'event_id'
        self.sparse = sparse

    def fit(self, event_ixs, event_ids=None, event_dict=None):
        """Binarize the events.

        Parameters
        ----------
        event_ixs : array, shape (n_events)
            The event indices or onsets in seconds. `If self.sfreq == 1`,
            then these are assumed to be indices. If `self.sfreq != 1`, these
            should be onsets in seconds and will be multiplied by `sfreq`.
        event_ids : array-like, shape (n_events) | None
            The event ID for each event. If None, all events are assumed to be
            of same type.
        event_dict : dict
            A dictionary of (event_id: event_id_name) pairs. Defines a string
            name for each event type.
        """
        if event_ixs.ndim > 1:
            raise ValueError("events must be shape (n_trials, 3),"
                             " found shape %s" % str(event_ixs.shape))
        event_ids = np.ones_like(event_ixs) if event_ids is None else event_ids
        unique_event_types = np.unique(event_ids)

        if event_dict is None:
            event_dict = dict(('event_%s' % ii, ii)
                              for ii in unique_event_types)
        ev_dict_rev = dict((value, key) for key, value in event_dict.items())

        # Turn event_ixs from seconds to indices in case sfreq != 1
        event_ixs = (event_ixs * self.sfreq).astype(int)

        # Iterate through event types and create columns of event onsets.
        events_continuous = np.zeros([self.n_times, len(unique_event_types)])
        event_names = []
        for ii, ev_type in enumerate(unique_event_types):
            msk_events = event_ids == ev_type
            i_ev = event_ixs[msk_events]
            events_continuous[i_ev, ii] = 1

            # Handle event names
            event_names.append(ev_dict_rev[ev_type])
        if self.sparse is True:
            events_continuous = sparse.csr_matrix(events_continuous)

        self.names_ = event_names
        self.events_continuous_ = events_continuous
        self.unique_event_types_ = unique_event_types

    def transform(self, X=None, y=None):
        """Return continuous events data.

        Returns
        -------
        events_continuous : array, shape (n_times, n_unique_events)
            A binary array with one column for each unique event type. 1s
            in the array correspond to event onsets for that row.
        """
        return self.events_continuous_

    def fit_transform(self, event_ixs, event_ids=None, event_dict=None):
        self.fit(event_ixs, event_ids=event_ids, event_dict=event_dict)
        return self.transform()


class DataSubsetter(object):
    """Use for returning a subset of time.

    Parameters
    ----------
    ixs : array, shape (n_ixs,)
        The indices to select from the data."""
    def __init__(self, ixs):
        self.ixs = np.asarray(ixs).astype(int)

    def fit(self, data, y=None):
        """Pull a subset of data points.

        Parameters
        ----------
        data : array, shape (n_samples, n_features)
            The data to be subsetted. The data points specified by
            `self.ixs` will be taken from rows of this array.
        """
        if data.shape[0] < self.ixs.max():
            raise ValueError('ixs exceed data shape')
        self.data_ = np.asarray(data)

    def transform(self, X=None, y=None):
        """Return a subset of data points.

        Returns
        -------
        data : array, shape (n_features, n_ixs)
            A subset of rows from the input array.
        """
        return self.data_[self.ixs, :]

    def fit_transform(self, ixs, y=None):
        self.fit(ixs)
        return self.transform()


def clean_inputs(X, y, reject=None, flat=None, info=None, decim=None,
                 tstep=None, nonzero=True):
    """Keep data points based on peak to peak amplitude / nonzero values.

    Parameters
    ----------
    X : array, shape (n_times, n_features)
        The input array (usually stimuli or continuous events)
    y : array, shape (n_times, n_channels)
        The neural data. Will be used for detecting noisy datapoints
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
        or cleaning raw data before the regression is performed: set up
        rejection parameters based on flatness of the signal. If None, no
        rejection is done. If a dict, keys are ('grad' | 'mag' |
        'eeg' | 'eog' | 'ecg') and values are minimal peak-to-peak values to
        select rejected epochs.
    info : None | instance of MNE Info object
        The Info object corresponding to the neural data `y`.
    decim : int
        Decimate by choosing only a subsample of data points. Highly
        recommended for data recorded at high sampling frequencies, as
        otherwise huge intermediate matrices have to be created and inverted.
    tstep : float
        Length of windows for peak-to-peak detection for raw data cleaning.
    nonzero : bool
        Whether to reject timepoints with nonzero values in all input features.
        Defaults to True.

    Returns
    -------
    X_cleaned : array, shape (n_clean_times, n_features)
        The input X array w/ non-active and noisy rows removed.
    y_cleaned : array, shape (n_clean_times, n_channels)
        The input y array w/ non-active and noisy rows removed.
    """

    # find only those positions where at least one predictor isn't 0
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y have different numbers of timepoints')
    if isinstance(X, sparse.spmatrix):
        nonzero_rows = np.unique(X.nonzero()[0])
        is_sparse = True
    else:
        nonzero_rows = np.unique(np.where((X != 0).any(1))[0])
        is_sparse = False
    keep_rows = nonzero_rows if nonzero is True else np.arange(X.shape[0])

    # reject positions based on extreme steps in the data
    if reject is not None:
        _, inds_bad = _reject_data_segments(y.T, reject, flat, decim=None,
                                            info=info, tstep=tstep)
        # Expand to include all bad indices, and remove from rows to keep
        inds_bad = np.hstack(range(t0, t1) for t0, t1 in inds_bad)
        keep_rows = np.setdiff1d(keep_rows, inds_bad)

    if is_sparse is True:
        X = X.tocsr()[keep_rows]
    else:
        X = X[keep_rows]

    return X, y[keep_rows]


def _delay_timeseries(ts, delays, sfreq=1.):
    """Return a time-lagged input timeseries.

    Parameters
    ----------
    ts: array, shape (n_times, n_feats)
        The timeseries to delay
    delays: array of floats, shape (n_delays,)
        The time (in seconds) of each delay for specifying
        pre-defined delays. Negative means timepoints in the past,
        positive means timepoints in the future.
    sfreq: int
        The sampling frequency of the series

    Returns
    -------
    delayed: array, shape(n_times, n_feats * n_delays)
        The delayed matrix
    """
    n_times, n_feats = ts.shape
    delays_ixs = np.round((delays * sfreq)).astype(int)

    # Remove duplicated ixs
    if np.unique(delays_ixs).shape[0] != delays_ixs.shape[0]:
        warn('Converting delays to indices resulted in duplicates.')

    # Convert ts to dense if it is sparse
    if isinstance(ts, sparse.spmatrix):
        is_sparse = True
        sp_type = type(ts)
        ts = ts.todense()
    else:
        is_sparse = False

    # Iterate through indices and append
    delayed = []
    delayed_names = []
    for ii, delay_ix in enumerate(delays_ixs):
        # Zeros to append to either the beginning or end.
        zeros = np.zeros([np.abs(delay_ix), n_feats])
        if delay_ix > 0:
            # Delay is in the future, so push data backward
            rolled = np.vstack([ts[delay_ix:, ...], zeros])
        elif delay_ix < 0:
            # Delay is in the past, so push data forward
            rolled = np.vstack([zeros, ts[:delay_ix, ...]])
        else:
            # If delay_ix is 0, just pass rolled
            rolled = ts.copy()
        delayed.append(rolled)

        # Save (feat_ix, delay) pairs
        delayed_names.append([(ii, delay_ix / float(sfreq))
                              for ii in range(n_feats)])

    delayed = np.array(delayed).swapaxes(0, 2)
    delayed = np.hstack(delayed)
    delayed_names = np.vstack(delayed_names)

    if is_sparse is True:
        delayed = sp_type(delayed)
    return delayed, delayed_names
