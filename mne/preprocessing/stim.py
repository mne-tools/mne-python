# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal.windows import hann

from .._fiff.pick import _picks_to_idx
from ..epochs import BaseEpochs
from ..event import find_events
from ..evoked import Evoked
from ..io import BaseRaw
from ..utils import _check_option, _check_preload, _validate_type, fill_doc


def _get_window(start, end):
    """Return window which has length as much as parameter start - end."""
    window = 1 - np.r_[hann(4)[:2], np.ones(np.abs(end - start) - 4), hann(4)[-2:]].T
    return window


def _fix_artifact(
    data, window, picks, first_samp, last_samp, base_tmin, base_tmax, mode
):
    """Modify original data by using parameter data."""
    if mode == "linear":
        x = np.array([first_samp, last_samp])
        f = interp1d(x, data[:, (first_samp, last_samp)][picks])
        xnew = np.arange(first_samp, last_samp)
        interp_data = f(xnew)
        data[picks, first_samp:last_samp] = interp_data
    if mode == "window":
        data[picks, first_samp:last_samp] = (
            data[picks, first_samp:last_samp] * window[np.newaxis, :]
        )
    if mode == "constant":
        data[picks, first_samp:last_samp] = data[picks, base_tmin:base_tmax].mean(
            axis=1
        )[:, None]


@fill_doc
def fix_stim_artifact(
    inst,
    events=None,
    event_id=None,
    tmin=0.0,
    tmax=0.01,
    *,
    baseline=None,
    mode="linear",
    stim_channel=None,
    picks=None,
):
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
    baseline : None | tuple, shape (2,)
        The baseline to use when ``mode='constant'``, in which case it
        must be non-None.

        .. versionadded:: 1.8
    mode : 'linear' | 'window' | 'constant'
        Way to fill the artifacted time interval.

        ``"linear"``
            Does linear interpolation.
        ``"window"``
            Applies a ``(1 - hanning)`` window.
        ``"constant"``
            Uses baseline average. baseline parameter must be provided.

        .. versionchanged:: 1.8
           Added the ``"constant"`` mode.
    stim_channel : str | None
        Stim channel to use.
    %(picks_all_data)s

    Returns
    -------
    inst : instance of Raw or Evoked or Epochs
        Instance with modified data.
    """
    _check_option("mode", mode, ["linear", "window", "constant"])
    s_start = int(np.ceil(inst.info["sfreq"] * tmin))
    s_end = int(np.ceil(inst.info["sfreq"] * tmax))
    if mode == "constant":
        _validate_type(
            baseline, (tuple, list), "baseline", extra="when mode='constant'"
        )
        _check_option("len(baseline)", len(baseline), [2])
        for bi, b in enumerate(baseline):
            _validate_type(
                b, "numeric", f"baseline[{bi}]", extra="when mode='constant'"
            )
        b_start = int(np.ceil(inst.info["sfreq"] * baseline[0]))
        b_end = int(np.ceil(inst.info["sfreq"] * baseline[1]))
    else:
        b_start = b_end = np.nan
    if (mode == "window") and (s_end - s_start) < 4:
        raise ValueError(
            'Time range is too short. Use a larger interval or set mode to "linear".'
        )
    window = None
    if mode == "window":
        window = _get_window(s_start, s_end)

    picks = _picks_to_idx(inst.info, picks, "data", exclude=())

    _check_preload(inst, "fix_stim_artifact")
    if isinstance(inst, BaseRaw):
        if events is None:
            events = find_events(inst, stim_channel=stim_channel)
        if len(events) == 0:
            raise ValueError("No events are found")
        if event_id is None:
            events_sel = np.arange(len(events))
        else:
            events_sel = events[:, 2] == event_id
        event_start = events[events_sel, 0]
        data = inst._data
        for event_idx in event_start:
            first_samp = int(event_idx) - inst.first_samp + s_start
            last_samp = int(event_idx) - inst.first_samp + s_end
            base_t1 = int(event_idx) - inst.first_samp + b_start
            base_t2 = int(event_idx) - inst.first_samp + b_end
            _fix_artifact(
                data, window, picks, first_samp, last_samp, base_t1, base_t2, mode
            )
    elif isinstance(inst, BaseEpochs):
        if inst.reject is not None:
            raise RuntimeError(
                "Reject is already applied. Use reject=None in the constructor."
            )
        e_start = int(np.ceil(inst.info["sfreq"] * inst.tmin))
        first_samp = s_start - e_start
        last_samp = s_end - e_start
        data = inst._data
        base_t1 = b_start - e_start
        base_t2 = b_end - e_start
        for epoch in data:
            _fix_artifact(
                epoch, window, picks, first_samp, last_samp, base_t1, base_t2, mode
            )

    elif isinstance(inst, Evoked):
        first_samp = s_start - inst.first
        last_samp = s_end - inst.first
        data = inst.data
        base_t1 = b_start - inst.first
        base_t2 = b_end - inst.first

        _fix_artifact(
            data, window, picks, first_samp, last_samp, base_t1, base_t2, mode
        )

    else:
        raise TypeError(f"Not a Raw or Epochs or Evoked (got {type(inst)}).")

    return inst
