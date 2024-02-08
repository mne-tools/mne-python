# ruff: noqa
# flake8: noqa

from scipy import integrate
import numpy as np
from .channels.layout import _merge_ch_data, _pair_grad_sensors
from .utils import (
    _check_option,
)


def _erp_measure_setup(evoked, ch_type, merge_grads):
    supported = (
        "mag",
        "grad",
        "eeg",
        "seeg",
        "dbs",
        "ecog",
        "misc",
        "None",
    ) + _FNIRS_CH_TYPES_SPLIT
    types_used = evoked.get_channel_types(unique=True, only_data_chs=True)

    _check_option("ch_type", str(ch_type), supported)

    if ch_type is not None and ch_type not in types_used:
        raise ValueError(f'Channel type "{ch_type}" not found in this evoked object.')

    elif len(types_used) > 1 and ch_type is None:
        raise RuntimeError(
            'Multiple data channel types found. Please pass the "ch_type" ' "parameter."
        )

    if merge_grads:
        if ch_type != "grad":
            raise ValueError('Channel type must be "grad" for merge_grads')
        elif mode == "neg":
            raise ValueError(
                "Negative mode (mode=neg) does not make " "sense with merge_grads=True"
            )

    if ch_type is not None:
        if merge_grads:
            picks = _pair_grad_sensors(evoked.info, topomap_coords=False)
        else:
            sel_evoked = evoked.pick(
                ["meg", "eeg", "misc", "seeg", "ecog", "dbs", "fnirs"]
            )

    data = sel_evoked.data
    ch_names = sel_evoked.ch_names

    if merge_grads:
        data, _ = _merge_ch_data(data, ch_type, [])
        ch_names = [ch_name[:-1] + "X" for ch_name in ch_names[::2]]

    return data, ch_names


def _restrict_time_interval(tmin, tmax, times, data):
    if tmin is None:
        tmin = times[0]
    if tmax is None:
        tmax = times[-1]

    if tmin < times.min() or tmax > times.max():
        if tmin < times.min():
            param_name = "tmin"
            param_val = tmin
        else:
            param_name = "tmax"
            param_val = tmax

        raise ValueError(
            f"{param_name} ({param_val}) is out of bounds. It must be "
            f"between {times.min()} and {times.max()}"
        )
    elif tmin > tmax:
        raise ValueError(f"tmin ({tmin}) must be <= tmax ({tmax})")

    time_win = (times >= tmin) & (times <= tmax)
    time_mask = np.ones_like(data).astype(bool)
    time_mask[:, time_win] = False

    return time_mask


def get_peak(
    insta,
    ch_type=None,
    tmin=None,
    tmax=None,
    mode="abs",
    time_as_index=False,
    merge_grads=False,
    return_amplitude=False,
    *,
    strict=True,
):
    """Get location and latency of peak amplitude.

    Parameters
    ----------
    ch_type : str | None
        The channel type to use. Defaults to None. If more than one channel
        type is present in the data, this value **must** be provided.
    tmin : float | None
        The minimum point in time to be considered for peak getting.
        If None (default), the beginning of the data is used.
    tmax : float | None
        The maximum point in time to be considered for peak getting.
        If None (default), the end of the data is used.
    mode : 'pos' | 'neg' | 'abs'
        How to deal with the sign of the data. If 'pos' only positive
        values will be considered. If 'neg' only negative values will
        be considered. If 'abs' absolute values will be considered.
        Defaults to 'abs'.
    time_as_index : bool
        Whether to return the time index instead of the latency in seconds.
    merge_grads : bool
        If True, compute peak from merged gradiometer data.
    return_amplitude : bool
        If True, return also the amplitude at the maximum response.

        .. versionadded:: 0.16
    strict : bool
        If True, raise an error if values are all positive when detecting
        a minimum (mode='neg'), or all negative when detecting a maximum
        (mode='pos'). Defaults to True.

        .. versionadded:: 1.7

    Returns
    -------
    ch_name : str
        The channel exhibiting the maximum response.
    latency : float | int
        The time point of the maximum response, either latency in seconds
        or index.
    amplitude : float
        The amplitude of the maximum response. Only returned if
        return_amplitude is True.

        .. versionadded:: 0.16
    """  # noqa: E501
    supported = (
        "mag",
        "grad",
        "eeg",
        "seeg",
        "dbs",
        "ecog",
        "misc",
        "None",
    ) + _FNIRS_CH_TYPES_SPLIT
    types_used = self.get_channel_types(unique=True, only_data_chs=True)

    _check_option("ch_type", str(ch_type), supported)

    if ch_type is not None and ch_type not in types_used:
        raise ValueError(f'Channel type "{ch_type}" not found in this evoked object.')

    elif len(types_used) > 1 and ch_type is None:
        raise RuntimeError(
            'Multiple data channel types found. Please pass the "ch_type" ' "parameter."
        )

    if merge_grads:
        if ch_type != "grad":
            raise ValueError('Channel type must be "grad" for merge_grads')
        elif mode == "neg":
            raise ValueError(
                "Negative mode (mode=neg) does not make " "sense with merge_grads=True"
            )

    meg = eeg = misc = seeg = dbs = ecog = fnirs = False
    picks = None
    if ch_type in ("mag", "grad"):
        meg = ch_type
    elif ch_type == "eeg":
        eeg = True
    elif ch_type == "misc":
        misc = True
    elif ch_type == "seeg":
        seeg = True
    elif ch_type == "dbs":
        dbs = True
    elif ch_type == "ecog":
        ecog = True
    elif ch_type in _FNIRS_CH_TYPES_SPLIT:
        fnirs = ch_type

    if ch_type is not None:
        if merge_grads:
            picks = _pair_grad_sensors(self.info, topomap_coords=False)
        else:
            picks = pick_types(
                self.info,
                meg=meg,
                eeg=eeg,
                misc=misc,
                seeg=seeg,
                ecog=ecog,
                ref_meg=False,
                fnirs=fnirs,
                dbs=dbs,
            )
    data = self.data
    ch_names = self.ch_names

    if picks is not None:
        data = data[picks]
        ch_names = [ch_names[k] for k in picks]

    if merge_grads:
        data, _ = _merge_ch_data(data, ch_type, [])
        ch_names = [ch_name[:-1] + "X" for ch_name in ch_names[::2]]

    ch_idx, time_idx, max_amp = _get_peak(
        data,
        self.times,
        tmin,
        tmax,
        mode,
        strict=strict,
    )

    out = (ch_names[ch_idx], time_idx if time_as_index else self.times[time_idx])

    if return_amplitude:
        out += (max_amp,)

    return out


def _get_peak(data, times, tmin=None, tmax=None, mode="abs", *, strict=True):
    """Get feature-index and time of maximum signal from 2D array.

    Note. This is a 'getter', not a 'finder'. For non-evoked type
    data and continuous signals, please use proper peak detection algorithms.

    Parameters
    ----------
    data : instance of numpy.ndarray (n_locations, n_times)
        The data, either evoked in sensor or source space.
    times : instance of numpy.ndarray (n_times)
        The times in seconds.
    tmin : float | None
        The minimum point in time to be considered for peak getting.
    tmax : float | None
        The maximum point in time to be considered for peak getting.
    mode : {'pos', 'neg', 'abs'}
        How to deal with the sign of the data. If 'pos' only positive
        values will be considered. If 'neg' only negative values will
        be considered. If 'abs' absolute values will be considered.
        Defaults to 'abs'.
    strict : bool
        If True, raise an error if values are all positive when detecting
        a minimum (mode='neg'), or all negative when detecting a maximum
        (mode='pos'). Defaults to True.

    Returns
    -------
    max_loc : int
        The index of the feature with the maximum value.
    max_time : int
        The time point of the maximum response, index.
    max_amp : float
        Amplitude of the maximum response.
    """
    _check_option("mode", mode, ["abs", "neg", "pos"])

    if tmin is None:
        tmin = times[0]
    if tmax is None:
        tmax = times[-1]

    if tmin < times.min() or tmax > times.max():
        if tmin < times.min():
            param_name = "tmin"
            param_val = tmin
        else:
            param_name = "tmax"
            param_val = tmax

        raise ValueError(
            f"{param_name} ({param_val}) is out of bounds. It must be "
            f"between {times.min()} and {times.max()}"
        )
    elif tmin > tmax:
        raise ValueError(f"tmin ({tmin}) must be <= tmax ({tmax})")

    time_win = (times >= tmin) & (times <= tmax)
    mask = np.ones_like(data).astype(bool)
    mask[:, time_win] = False

    maxfun = np.argmax
    if mode == "pos":
        if strict and not np.any(data[~mask] > 0):
            raise ValueError(
                "No positive values encountered. Cannot " "operate in pos mode."
            )
    elif mode == "neg":
        if strict and not np.any(data[~mask] < 0):
            raise ValueError(
                "No negative values encountered. Cannot " "operate in neg mode."
            )
        maxfun = np.argmin

    masked_index = np.ma.array(np.abs(data) if mode == "abs" else data, mask=mask)

    max_loc, max_time = np.unravel_index(maxfun(masked_index), data.shape)

    return max_loc, max_time, data[max_loc, max_time]


def get_mean_amplitude(
    insta,
    ch_type=None,
    tmin=None,
    tmax=None,
    mode="abs",
    time_as_index=False,
    merge_grads=False,
    return_amplitude=False,
):
    """Get the mean amplitude in a specific time window.

    Parameters
    ----------
    ch_type : str | None
        The channel type to use. If None, the first available channel type from
        the following list is used: 'mag', 'grad', 'planar1', 'planar2', 'eeg'.
    tmin : float | None
        The beginning of the time window in seconds. If None the beginning of
        the data is used.
    tmax : float | None
        The end of the time window in seconds. If None the end of the data is
        used.
    mode : str
        How to combine multiple channels. The following options are available:
        'abs' : Take the absolute value of each channel and then average.
        'mean' : Average across channels.
        'max' : Take the maximum across channels.
        'mean_signed' : Average across channels, retaining sign.
        'median' : Take the median across channels.
        'percentile' : Take the specified percentile across channels. If
        percentile is 50, this is the same as 'median'.
    time_as_index : bool
        Whether to consider time as index or as float (default False).
    merge_grads : bool
        If True, merge gradiometer data into one value by taking the RMS
        (root mean square) for each pair of gradiometers. The RMS is taken
        over the pair of gradiometers before averaging across channels.
        This is only used for MEG data.
    return_amplitude : bool
        If True, return the amplitude values. If False, return the evoked
        instance (default False).
    strict : bool
        If True, raise an error if channels are missing. If False, ignore
        channels that are missing.

    Returns
    -------
    evoked : instance of Evoked
        The modified evoked instance. If return_amplitude is True, the
        amplitude values are returned instead.

    See Also
    --------
    mne.Evoked.get_peak : Get the time and value of the peak amplitude."""
    # check that the data is preloaded

    pass


def get_area(insta, ch_type=None, tmin=None, tmax=None, mode="abs", merge_grads=False):
    """Get the area under the curve in a specific time window.

    Parameters
    ----------
    ch_type : str | None
        The channel type to use. If None, the first available channel type from
        the following list is used: 'mag', 'grad', 'planar1', 'planar2', 'eeg'.
    tmin : float | None
        The beginning of the time window in seconds. If None the beginning of
        the data is used.
    tmax : float | None
        The end of the time window in seconds. If None the end of the data is
        used.
    mode : str
        How to combine multiple channels. The following options are available:
        'abs' : Take the absolute value of each channel and then average.
        'mean' : Average across channels.
        'max' : Take the maximum across channels.
        'mean_signed' : Average across channels, retaining sign.
        'median' : Take the median across channels.
        'percentile' : Take the specified percentile across channels. If
        percentile is 50, this is the same as 'median'.
    merge_grads : bool
        If True, merge gradiometer data into one value by taking the RMS
        (root mean square) for each pair of gradiometers. The RMS is taken
        over the pair of gradiometers before averaging across channels.
        This is only used for MEG data.
    return_area : bool
        If True, return the area values. If False, return the evoked instance
        (default False).

    Returns
    -------
    evoked : instance of Evoked
        The modified evoked instance. If return_area is True, the area values
        are returned instead.

    See Also
    --------
    mne.Evoked.get_peak : Get the time and value of the peak amplitude."""
    # check that the data is preloaded
    _check_option("mode", mode, ["intg", "abs", "neg", "pos"])
    data, ch_names = _erp_measure_setup(insta, ch_type, merge_grads)
    time_mask = _restrict_time_interval(tmin, tmax, insta.times, data)

    if mode == "abs":
        data = np.abs(data)
    elif mode == "neg":
        # Set positive values to zero
        data[data > 0] = 0
    elif mode == "pos":
        # Set negative values to zero
        data[data < 0] = 0
