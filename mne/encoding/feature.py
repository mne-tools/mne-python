"""A collection of classes for transforming, preprocessing, and fitting."""
import numpy as np


class DataDelayer(object):
    """Transformer for creating delayed versions of an input stimulus.

    Parameters
    ----------
    delays : array-like, shape (n_delays,)
        The delays to create for the data. If sfreq is 1, it is assumed
        that these are in samples. If sfreq != 1, these should be in
        seconds.
    sfreq : float | int
        The sampling frequency to use when constructing delays. If 1, then
        delays should contain sample indices, not time in seconds.
    """
    def __init__(self, delays=[0.], sfreq=1.):
        self.delays = np.asarray(delays)
        self.sfreq = float(sfreq)

    def fit(self, X, X_names=None):
        """Create a time-lagged representation of X.

        Parameters
        ----------
        X : array-like, shape (n_features, n_times)
            The input data to be time-lagged and returned.
        """
        self.X_delayed = delay_timeseries(X, self.sfreq, self.delays)
        self.X_names = X_names

    def transform(self, X=None, y=None):
        if self.X_names is None:
            return self.X_delayed
        else:
            return self.X_delayed, self.X_names

    def fit_transform(self, X, X_names=None):
        """See `transform` method."""
        self.fit(X, X_names=X_names)
        return self.transform()


class EventsBinarizer(object):
    """Create a continuous events-based feature set from event indices."""
    def __init__(self, n_times, sfreq=1.):
        self.n_times = n_times
        self.sfreq = sfreq

    def fit(self, event_ixs, event_ids=None, event_dict=None):
        """Create a continuous-representation of event onsets.

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
                             " found shape %s" % event_ixs.shape)
        event_ids = np.ones_like(event_ixs) if event_ids is None else event_ids
        unique_event_types = np.unique(event_ids)
        if event_dict is None:
            event_dict = dict((ii, 'event_%s' % ii)
                              for ii in unique_event_types)

        # Turn event_ixs from seconds to indices in case sfreq != 1
        event_ixs = event_ixs * self.sfreq
        event_ixs = event_ixs.astype(int)

        # Iterate through event types and create columns of event onsets.
        events_continuous = np.zeros([len(unique_event_types), self.n_times])
        event_names = []
        for ii, ev_type in enumerate(unique_event_types):
            msk_events = event_ids == ev_type
            i_ev = event_ixs[msk_events]
            events_continuous[ii, i_ev] = 1

            # Handle event names
            event_names.append(event_dict[ev_type])
        self.event_names = event_names
        self.events_continuous = events_continuous
        self.unique_event_types = unique_event_types

    def transform(self, X=None, y=None):
        """Return continuous events data.

        Returns
        -------
        events_continuous : array, shape (n_unique_events, n_times)
            A binary array with one row for each unique event type. 1s
            in the array correspond to event onsets for that row.
        """
        return self.events_continuous

    def fit_transform(self, event_ixs, event_ids=None, event_dict=None):
        self.fit(event_ixs, event_ids=event_ids, event_dict=event_dict)
        return self.transform()


class DataSubsetter(object):
    """Return a subset of data."""
    def __init__(self, ixs):
        self.ixs = ixs.astype(int)

    def fit(self, data, y=None):
        if data.shape[-1] < self.ixs.max():
            raise ValueError('ixs exceed data shape')
        self.data = np.asarray(data)

    def transform(self, X=None, y=None):
        return self.data[..., self.ixs]

    def fit_transform(self, ixs, y=None):
        self.fit(ixs)
        return self.transform()


def delay_timeseries(ts, sfreq, delays):
    """Return a time-lagged input timeseries.

    Parameters
    ----------
    ts: array, shape (n_feats, n_times)
        The timeseries to delay
    sfreq: int
        The sampling frequency of the series
    delays: list of floats
        The time (in seconds) of each delay. Negative means
        timepoints in the past, positive means timepoints in
        the future.

    Returns
    -------
    delayed: array, shape(n_feats * n_delays, n_times)
        The delayed matrix
    """
    delayed = np.zeros([len(delays), ts.shape[-1]])
    for ii, delay in enumerate(delays):
        roll_amount = -1 * int(delay * sfreq)
        rolled = np.roll(ts, roll_amount, axis=-1)
        if roll_amount < 0:
            rolled[:, roll_amount:0] = 0
        elif roll_amount > 0:
            rolled[:, 0:roll_amount] = 0
        delayed[ii] = rolled
    delayed = np.vstack(delayed)
    return delayed
