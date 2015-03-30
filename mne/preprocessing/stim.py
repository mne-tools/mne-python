# Authors: Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD (3-clause)

import numpy as np
from scipy import signal, interpolate
from ..evoked import Evoked
from ..epochs import Epochs
from ..io import Raw
from ..utils import deprecated

from .. import pick_types
from ..io.pick import pick_channels


def _get_window(start, end):
    window = 1 - np.r_[signal.hann(4)[:2],
                       np.ones(np.abs(end - start) - 4),
                       signal.hann(4)[-2:]].T
    return window


def _fix_artifact(orig, data, window, picks, first_samp, last_samp, mode):
    if mode == 'linear':
        x = np.array([first_samp, last_samp])
        f = interpolate.interp1d(x, data[:, (0, -1)])
        xnew = np.arange(first_samp, last_samp)
        interp_data = f(xnew)
        orig[picks, first_samp:last_samp] = interp_data
    if mode == 'window':
        orig[picks, first_samp:last_samp] = \
            data[picks, first_samp:last_samp] * window[np.newaxis, :]


@deprecated('`eliminate_stim_artifact` will be deprecated '
            'in v0.10 : Use fix_stim_artifact_raw')
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
        window = _get_window(s_start, s_end)

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


def fix_stim_artifact(inst, events=None, event_id=None, tmin=0.,
                      tmax=0.01, mode='linear', copy=False):
    """Eliminates stimulations artifacts from instance

    The instance will be modified in place (no copy)

    Parameters
    ----------
    inst : instance of evoked or epochs
        instance
    events : array, shape (n_events, 3)
        The list of events. No need when inst is epochs or evoked
    event_id : int
        The id of the events generating the stimulation artifacts.
        No need when inst is epochs or evoked
    tmin : float
        Start time of the interpolation window in seconds.
    tmax : float
        End time of the interpolation window in seconds.
    mode : 'linear' | 'window'
        way to fill the artifacted time interval.
        'linear' does linear interpolation
        'window' applies a (1 - hanning) window.
    copy : bool
        If False inst is cropped in place.

    Returns
    -------
    inst : instance of evoked or epochs
        instance with modified data
    """
    if copy:
        inst = inst.copy()
    s_start = int(np.ceil(inst.info['sfreq'] * tmin))
    s_end = int(np.ceil(inst.info['sfreq'] * tmax))
    if (s_end - s_start - 4) < 0:
        raise ValueError('Time between tmin and tmax is too short'
                         ' to fix artifact. Input longer values')
    window = None
    if mode == 'window':
        window = _get_window(s_start, s_end)
    ch_names = inst.info['ch_names']
    picks = pick_channels(ch_names, ch_names)

    if isinstance(inst, Raw):
        if inst.preload is False:
            raise RuntimeError('Modifying data of Raw is only supported '
                               'when preloading is used. Use preload=True '
                               '(or string) in the constructor.')
        events_sel = (events[:, 2] == event_id)
        event_start = events[events_sel, 0]
        if mode is 'linear':
            data, _ = inst[picks, :]
        elif mode is 'window':
            data, _ = inst[:, :]
        for k in range(len(event_start)):
            first_samp = int(event_start[k]) - inst.first_samp + s_start
            last_samp = int(event_start[k]) - inst.first_samp + s_end
            _fix_artifact(inst, data, window, picks, first_samp, last_samp, mode)

    elif isinstance(inst, Epochs):
        if inst.preload is False:
            raise RuntimeError('Modifying data of Epochs is only supported '
                               'when preloading is used. Use preload=True '
                               'in the constructor.')
        if inst.reject:
            raise RuntimeError('Reject is already applied. Use reject=False '
                               'in the constructor.')
        e_start = int(np.ceil(inst.info['sfreq'] * inst.tmin))
        first_samp = s_start - e_start
        last_samp = s_end - e_start
        data = inst.get_data()[:, picks, :]
        k = 0
        for epoch in data:
            _fix_artifact(inst._data[k], epoch, window, picks, first_samp, last_samp, mode)
            k += 1

    elif isinstance(inst, Evoked):
        first_samp = s_start - inst.first
        last_samp = s_end - inst.first
        data = inst.data
        _fix_artifact(inst.data, data, window, picks, first_samp, last_samp, mode)

    else:
        raise TypeError('Not a Epochs or Evoked or Raw')

    return inst
