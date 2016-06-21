"""A collection of classes for transforming, preprocessing, and fitting."""
import numpy as np


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
    sfreq: int
        The sampling frequency of the series
    """
    def __init__(self, delays=None, time_window=None, sfreq=1.):
        # Check if we need to create delays ourselves
        if time_window is not None:
            delays = np.linspace(time_window[0], time_window[1], sfreq)
        elif delays is None:
            raise ValueError('One of `time_window` or `delays` must be given.')
        self.delays = np.asarray(delays)
        self.sfreq = float(sfreq)
        self.feature_type = 'delay'

    def fit(self, X, y=None):
        """Create a time-lagged representation of X.

        Parameters
        ----------
        X : array-like, shape (n_features, n_times)
            The input data to be time-lagged and returned.
        """
        self.X_delayed, self.names = _delay_timeseries(
            X, self.delays, sfreq=self.sfreq)

    def transform(self, X=None, y=None):
        return self.X_delayed

    def fit_transform(self, X, y=None):
        """See `transform` method."""
        self.fit(X)
        return self.transform()


class EventsBinarizer(object):
    """Create a continuous-representation of event onsets.

    Patameters
    ----------
    n_times : int
        The total number of times in the output array
    sfreq : float
        The sampling frequency used to convert times to sample indices.
    """
    def __init__(self, n_times, sfreq=1.):
        self.n_times = n_times
        self.sfreq = sfreq
        self.feature_type = 'eventid'

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
        self.names = event_names
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
    """Use for returning a subset of data.

    Parameters
    ----------
    ixs : array, shape (n_ixs,)
        The indices to select from the data."""
    def __init__(self, ixs):
        self.ixs = ixs.astype(int)

    def fit(self, data, y=None):
        """Pull a subset of data points.

        Parameters
        ----------
        data : array, shape (n_features, n_samples)
            The data to be subsetted. The data points specified by
            `self.ixs` will be taken from columns of this array.
        """
        if data.shape[-1] < self.ixs.max():
            raise ValueError('ixs exceed data shape')
        self.data = np.asarray(data)

    def transform(self, X=None, y=None):
        """Return a subset of data points.

        Returns
        -------
        data : array, shape (n_features, n_ixs)
            A subset of columns from the input array.
        """
        return self.data[..., self.ixs]

    def fit_transform(self, ixs, y=None):
        self.fit(ixs)
        return self.transform()


def _delay_timeseries(ts, delays, sfreq=1.):
    """Return a time-lagged input timeseries.

    Parameters
    ----------
    ts: array, shape (n_feats, n_times)
        The timeseries to delay
    delays: array of floats, shape (n_delays,)
        The time (in seconds) of each delay for specifying
        pre-defined delays. Negative means timepoints in the past,
        positive means timepoints in the future.
    sfreq: int
        The sampling frequency of the series

    Returns
    -------
    delayed: array, shape(n_feats * n_delays, n_times)
        The delayed matrix
    """
    n_feats, n_times = ts.shape
    delays_ixs = -1 * (delays * sfreq).astype(int)
    delays_ixs = np.unique(delays_ixs)  # Remove duplicated ixs

    # Iterate through indices and append
    delayed = []
    delayed_names = []
    for ii, roll_amount in enumerate(delays_ixs):
        # Convert to sample indices
        rolled = np.roll(ts, roll_amount, axis=-1)
        if roll_amount < 0:
            rolled[:, roll_amount:] = 0
        elif roll_amount > 0:
            rolled[:, :roll_amount] = 0
        delayed.append(rolled)
        # Save (feat_ix, delay) pairs
        delayed_names.append([(ii, roll_amount / float(sfreq))
                              for ii in range(n_feats)])
    delayed = np.vstack(delayed)
    delayed_names = np.hstack(delayed_names)
    return delayed, delayed_names
