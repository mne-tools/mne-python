# Authors: Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD (3-clause)

import numpy as np
from scipy import signal, interpolate

from .. import pick_types


def eliminate_stim_artifact(raw, events, event_id, tmin=-0.005,
                            tmax=0.01, mode='linear'):
    """Eliminates stimulations artifacts from raw data

    The raw object will be modified in place (no copy)

    Parameters
    ----------
    raw : Raw object
        raw data object.
    events : array, shape (n_events, 3)
        The list of events.
    event_id : int
        The id of the events generating the stimulation artifacts.
    tmin : float
        Start time of the interpolation window in seconds.
    tmax : float
        End time of the interpolation window in seconds.
    mode : 'linear' | 'window'
        way to fill the artifacted time interval.
        'linear' does linear interpolation
        'window' applies a (1 - hanning) window.

    Returns
    -------
    raw: Raw object
        raw data object.
    """
    if not raw.preload:
        raise RuntimeError('Modifying data of Raw is only supported '
                           'when preloading is used. Use preload=True '
                           '(or string) in the constructor.')
    events_sel = (events[:, 2] == event_id)
    event_start = events[events_sel, 0]
    s_start = int(np.ceil(raw.info['sfreq'] * tmin))
    s_end = int(np.ceil(raw.info['sfreq'] * tmax))

    picks = pick_types(raw.info, meg=True, eeg=True, eog=True, ecg=True,
                       emg=True, ref_meg=True, misc=True, chpi=True,
                       exclude='bads', stim=False, resp=False)

    if mode == 'window':
        window = 1 - np.r_[signal.hann(4)[:2],
                           np.ones(np.abs(s_end - s_start) - 4),
                           signal.hann(4)[-2:]].T

    for k in range(len(event_start)):
        first_samp = int(event_start[k]) - raw.first_samp + s_start
        last_samp = int(event_start[k]) - raw.first_samp + s_end
        data, _ = raw[picks, first_samp:last_samp]
        if mode == 'linear':
            x = np.array([first_samp, last_samp])
            f = interpolate.interp1d(x, data[:, (0, -1)])
            xnew = np.arange(first_samp, last_samp)
            interp_data = f(xnew)
            raw[picks, first_samp:last_samp] = interp_data
        elif mode == 'window':
            raw[picks, first_samp:last_samp] = data * window[np.newaxis, :]
    return raw
