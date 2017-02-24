# Authors: Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD (3-clause)

import numpy as np
from ..evoked import Evoked
from ..epochs import BaseEpochs
from ..io import BaseRaw
from ..event import find_events

from ..io.pick import _pick_data_channels
from ..io.base import _check_preload


def _get_window(start, end):
    """Return window which has length as much as parameter start - end."""
    from scipy.signal import hann
    window = 1 - np.r_[hann(4)[:2],
                       np.ones(np.abs(end - start) - 4),
                       hann(4)[-2:]].T
    return window


def _fix_artifact(data, window, picks, first_samp, last_samp, mode):
    """Modify original data by using parameter data."""
    from scipy.interpolate import interp1d
    if mode == 'linear':
        x = np.array([first_samp, last_samp])
        f = interp1d(x, data[:, (first_samp, last_samp)][picks])
        xnew = np.arange(first_samp, last_samp)
        interp_data = f(xnew)
        data[picks, first_samp:last_samp] = interp_data
    if mode == 'window':
        data[picks, first_samp:last_samp] = \
            data[picks, first_samp:last_samp] * window[np.newaxis, :]


def fix_stim_artifact(inst, events=None, event_id=None, tmin=0.,
                      tmax=0.01, mode='linear', stim_channel=None):
    """Eliminate stimulation's artifacts from instance.

    .. note:: This function operates in-place, consider passing
              ``inst.copy()`` if this is not desired.

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
    stim_channel : str | None
        Stim channel to use.

    Returns
    -------
    inst : instance of Raw or Evoked or Epochs
        Instance with modified data
    """
    if mode not in ('linear', 'window'):
        raise ValueError("mode has to be 'linear' or 'window' (got %s)" % mode)
    s_start = int(np.ceil(inst.info['sfreq'] * tmin))
    s_end = int(np.ceil(inst.info['sfreq'] * tmax))
    if (mode == "window") and (s_end - s_start) < 4:
        raise ValueError('Time range is too short. Use a larger interval '
                         'or set mode to "linear".')
    window = None
    if mode == 'window':
        window = _get_window(s_start, s_end)
    picks = _pick_data_channels(inst.info)

    _check_preload(inst, 'fix_stim_artifact')
    if isinstance(inst, BaseRaw):
        if events is None:
            events = find_events(inst, stim_channel=stim_channel)
        if len(events) == 0:
            raise ValueError('No events are found')
        if event_id is None:
            events_sel = np.arange(len(events))
        else:
            events_sel = (events[:, 2] == event_id)
        event_start = events[events_sel, 0]
        data = inst._data
        for event_idx in event_start:
            first_samp = int(event_idx) - inst.first_samp + s_start
            last_samp = int(event_idx) - inst.first_samp + s_end
            _fix_artifact(data, window, picks, first_samp, last_samp, mode)

    elif isinstance(inst, BaseEpochs):
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
