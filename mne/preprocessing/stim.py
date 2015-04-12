# Authors: Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD (3-clause)

import numpy as np
from scipy import signal, interpolate
from ..evoked import Evoked
from ..epochs import Epochs
from ..io import Raw
from ..utils import deprecated
from ..event import find_events

from .. import pick_types
from ..io.pick import pick_channels


def _get_window(start, end):
    """Return window which has length as much as parameter start - end"""
    window = 1 - np.r_[signal.hann(4)[:2],
                       np.ones(np.abs(end - start) - 4),
                       signal.hann(4)[-2:]].T
    return window


def _check_preload(inst):
    """Check if inst.preload is False. If it is False, raising error"""
    if inst.preload is False:
        raise RuntimeError('Modifying data of Instance is only supported '
                           'when preloading is used. Use preload=True '
                           '(or string) in the constructor.')


def _fix_artifact(data, window, picks, first_samp, last_samp, mode):
    """Modify original data by using parameter data"""
    if mode == 'linear':
        x = np.array([first_samp, last_samp])
        f = interpolate.interp1d(x, data[:, (first_samp, last_samp)])
        xnew = np.arange(first_samp, last_samp)
        interp_data = f(xnew)
        data[picks, first_samp:last_samp] = interp_data
    if mode == 'window':
        data[picks, first_samp:last_samp] = \
            data[picks, first_samp:last_samp] * window[np.newaxis, :]


@deprecated('`eliminate_stim_artifact` will be deprecated '
            'in v0.10 : Use fix_stim_artifact')
def eliminate_stim_artifact(raw, events, event_id, tmin=-0.005,
                            tmax=0.01, mode='linear'):
    """Eliminate stimulations artifacts from raw data

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
    """Eliminate stimulation's artifacts from instance

    Parameters
    ----------
    inst : instance of Raw or Epochs or Evoked
        The data.
    events : array, shape (n_events, 3)
        The list of events. Required only when inst is Raw.
    event_id : int
        The id of the events generating the stimulation artifacts.
        If None, read all events. Required only when inst is Raw.
    tmin : float
        Start time of the interpolation window in seconds.
    tmax : float
        End time of the interpolation window in seconds.
    mode : 'linear' | 'window'
        Way to fill the artifacted time interval.
        'linear' does linear interpolation
        'window' applies a (1 - hanning) window.
    copy : bool
        If True, data will be copied. Else data may be modified in place.

    Returns
    -------
    inst : instance of Raw or Evoked or Epochs
        Instance with modified data
    """
    if mode not in ('linear', 'window'):
        raise ValueError("mode has to be 'linear' or 'window' (got %s)" % mode)

    if copy:
        inst = inst.copy()
    s_start = int(np.ceil(inst.info['sfreq'] * tmin))
    s_end = int(np.ceil(inst.info['sfreq'] * tmax))
    if (mode == "window") and (s_end - s_start) < 4:
        raise ValueError('Time range is too short. Use a larger interval '
                         'or set mode to "linear".')
    window = None
    if mode == 'window':
        window = _get_window(s_start, s_end)
    ch_names = inst.info['ch_names']
    picks = pick_channels(ch_names, ch_names)

    if isinstance(inst, Raw):
        _check_preload(inst)
        if events is None:
            events = find_events(inst)
        if len(events) == 0:
            raise ValueError('No events are found')
        if event_id is None:
            events_sel = np.arange(len(events))
        else:
            events_sel = (events[:, 2] == event_id)
        event_start = events[events_sel, 0]
        data, _ = inst[:, :]
        for event_idx in event_start:
            first_samp = int(event_idx) - inst.first_samp + s_start
            last_samp = int(event_idx) - inst.first_samp + s_end
            _fix_artifact(data, window, picks, first_samp, last_samp, mode)

    elif isinstance(inst, Epochs):
        _check_preload(inst)
        if inst.reject is not None:
            raise RuntimeError('Reject is already applied. Use reject=None '
                               'in the constructor.')
        e_start = int(np.ceil(inst.info['sfreq'] * inst.tmin))
        first_samp = s_start - e_start
        last_samp = s_end - e_start
        data = inst._data
        for epoch in data:
            _fix_artifact(epoch, window, picks, first_samp, last_samp, mode)

    elif isinstance(inst, Evoked):
        first_samp = s_start - inst.first
        last_samp = s_end - inst.first
        data = inst.data
        _fix_artifact(data, window, picks, first_samp, last_samp, mode)

    else:
        raise TypeError('Not a Raw or Epochs or Evoked (got %s).' % type(inst))

    return inst
