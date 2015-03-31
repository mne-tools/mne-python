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
    """Return window which has length as much as parameter start - end"""
    if np.abs(end - start) <= 4:  # When time range is too small
        window = np.zeros(end - start)
    else:
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


def _fix_artifact(orig, data, window, picks, first_samp, last_samp, mode):
    """Modify original data by using parameter data"""
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
    inst : instance of Raw or Evoked or Epochs
        instance
    events : array, shape (n_events, 3)
        The list of events. Required only when inst is raw
    event_id : int
        The id of the events generating the stimulation artifacts.
        Required only when inst is raw
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
    inst : instance of Raw or Evoked or Epochs
        instance with modified data
    """
    if copy:
        inst = inst.copy()
    s_start = int(np.ceil(inst.info['sfreq'] * tmin))
    s_end = int(np.ceil(inst.info['sfreq'] * tmax))
    window = None
    if mode == 'window':
        window = _get_window(s_start, s_end)
    ch_names = inst.info['ch_names']
    picks = pick_channels(ch_names, ch_names)

    if isinstance(inst, Raw):
        _check_preload(inst)
        events_sel = (events[:, 2] == event_id)
        event_start = events[events_sel, 0]
        data, _ = inst[:, :]
        for event_idx in event_start:
            first_samp = int(event_idx) - inst.first_samp + s_start
            last_samp = int(event_idx) - inst.first_samp + s_end
            _fix_artifact(inst, data, window, picks, first_samp,
                          last_samp, mode)

    elif isinstance(inst, Epochs):
        _check_preload(inst)
        if inst.reject:
            raise RuntimeError('Reject is already applied. Use reject=None '
                               'in the constructor.')
        e_start = int(np.ceil(inst.info['sfreq'] * inst.tmin))
        first_samp = s_start - e_start
        last_samp = s_end - e_start
        data = inst.get_data()[:, picks, :]
        for epoch, k in zip(data, range(len(data[0][0]))):
            _fix_artifact(inst.get_data()[k], epoch, window, picks, first_samp,
                          last_samp, mode)

    elif isinstance(inst, Evoked):
        first_samp = s_start - inst.first
        last_samp = s_end - inst.first
        data = inst.data
        _fix_artifact(inst.data, data, window, picks, first_samp,
                      last_samp, mode)

    else:
        raise TypeError('Not a Raw or Epochs or Evoked')

    return inst
