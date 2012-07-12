import numpy as np
from scipy import signal, interpolate


def eliminate_stim_artifact(raw, events, event_id, tmin=-0.005,
                                tmax=0.01, mode='linear'):
    """Eliminates stimulations artifacts from raw data
     
    The raw object will be modified in place (no copy)

    Parameters
    ----------
    raw: Raw object
        raw data object
    events: array, shape (n_events, 3)
        The list of events
    event_id: int
        The id of the events generating the stimulation artifacts.
    tmin : float
        Start time before event in seconds
    tmax : float
        End time after event in seconds
    mode : 'linear' | 'window'
        way to fill the artifacted time interval
        'linear' does linear interpolation
        'window' applies a (1 - hanning) window

    Returns
    -------
    raw: Raw object
        raw data object
    """
    events_sel = (events[:, 2] == event_id)
    event_start = events[events_sel, 0]
    s_start = np.ceil(raw.info['sfreq'] * np.abs(tmin))[0]
    s_end = np.ceil(raw.info['sfreq'] * tmax)[0]
    if mode == 'linear':
        for k in range(len(event_start)):
            x = np.array([event_start[k] - raw.first_samp - s_start, \
                            event_start[k] - raw.first_samp + s_end])
            data = raw._data[:, [event_start[k] - raw.first_samp - s_start, \
                                event_start[k] - raw.first_samp + s_end]]
            f = interpolate.interp1d(x, data)
            xnew = np.arange(event_start[k] - raw.first_samp - s_start, \
                             event_start[k] - raw.first_samp + s_end)
            interp_data = f(xnew)
            raw._data[:, event_start[k] - raw.first_samp - s_start:\
                         event_start[k] - raw.first_samp + s_end] = interp_data
    if mode == 'window':
        window = 1 - np.r_[signal.hann(4)[:2], np.ones(s_end + s_start - 4), \
                            signal.hann(4)[-2:]].T
        for k in range(len(event_start)):
            raw._data[:, event_start[k] - raw.first_samp - s_start:\
                         event_start[k] - raw.first_samp + s_end] *= \
                                                    window[np.newaxis, :]
    return raw
