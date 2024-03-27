# ruff: noqa
# flake8: noqa

from scipy import integrate
import pandas as pd
import numpy as np
from mne._fiff.pick import _pick_data_channels, _picks_to_idx, pick_info
from mne.channels.layout import _merge_ch_data, _pair_grad_sensors
from mne.utils import (
    _check_option,
    _time_mask,
)


def _get_peak(
    evoked, tmin=None, tmax=None, picks="all", mode="abs", average=False, strict=True
):
    """Helper function to get the peak amplitude and latency of an evoked response."""

    data = evoked.get_data(picks=picks)
    times = evoked.times
    mask = _time_mask(times, tmin, tmax, evoked.info["sfreq"])
    data_masked = data[:, mask]

    if average:
        data_masked = np.mean(data_masked, axis=0)

    if mode == "abs":
        data_masked = np.abs(data_masked)
    elif mode == "neg":
        if strict and not np.any(data_masked < 0):
            raise ValueError(
                "No negative values encountered. Cannot operate in neg mode."
            )
        data_masked = -data_masked
    elif mode == "pos":
        if strict and not np.any(data_masked > 0):
            raise ValueError(
                "No positive values encountered. Cannot operate in pos mode."
            )

    max_indices = np.argmax(data_masked, axis=1)
    peak_amplitudes = data[np.arange(data.shape[0]), max_indices + np.where(mask)[0][0]]
    peak_latencies = times[max_indices + np.where(mask)[0][0]]

    return peak_latencies, peak_amplitudes, data_masked, mask, times


def get_peak(
    evoked,
    tmin=None,
    tmax=None,
    picks="all",
    mode="abs",
    average=False,
    strict=True,
):
    """Get the peak amplitude and latency of an evoked response and return a
    DataFrame.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked response object.
    %(erp_evoked_tmin_tmax)s
    %(picks_all)s
    mode : str
    Specifies how the peak amplitude should be determined. Can be one of:
    - 'abs' : The peak amplitude is the maximum absolute value.
    - 'neg': The peak amplitude is the maximum negative value. If there are no
      negative values and `strict` is True, a ValueError is raised.
    - 'pos': The peak amplitude is the maximum positive value. If there are no
      positive values and `strict` is True, a ValueError is raised.
    Defaults to abs'.
    average : bool
        If True, the peak amplitude is computed by averaging the data across
        channels before finding the peak. Defaults to False.
    %(erp_strict)s

    Returns
    -------
    peak_df : pd.DataFrame
        A DataFrame with columns 'channels', 'latency', and 'amplitude'
        containing the peak amplitude and latency for each channel.
        (Will only contain one row 'with 'latency' and 'amplitude' if average=True)
    """

    _check_option("mode", mode, ["abs", "neg", "pos", "intg"])
    peak_latencies, peak_amplitudes, data_masked, mask, times = _get_peak(
        evoked, tmin, tmax, picks, mode, average, strict
    )

    peak_df = pd.DataFrame(
        {
            "channels": evoked.ch_names,
            "latency": peak_latencies,
            "amplitude": peak_amplitudes,
        }
    )
    if average:
        peak_df = peak_df.iloc[0]

    return peak_df


def get_area(
    evoked,
    tmin=None,
    tmax=None,
    picks="all",
    mode="abs",
    average=False,
):
    """
    Get the area under the curve of an evoked response within a given time window.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked response object.
    %(erp_evoked_tmin_tmax)s
    %(picks_all)s
    mode : str
        Specifies how the area should be computed. Can be one of:
        - 'abs': The absolute value of the data is used.
        - 'neg': Only negative values are considered.
        - 'pos': Only positive values are considered.
        - 'intg': The integral of the data is computed without rectification.
    Defaults to abs'.
    average : bool
        If True, the area is computed by averaging the data across channels
        before integration. Defaults to False.

    Returns
    -------
    area_df : pd.DataFrame
        A DataFrame with columns 'channels' and 'area' containing the area
        under the curve for each channel. (Will only contain one row with
        'area' if average=True)

    """
    _check_option("mode", mode, ["abs", "neg", "pos", "intg"])
    data = evoked.get_data(picks=picks)
    times = evoked.times
    mask = _time_mask(times, tmin, tmax, evoked.info["sfreq"])
    data_masked = data[:, mask]

    if average:
        data_masked = np.mean(data_masked, axis=0)
    if mode == "abs":
        data_masked = np.abs(data_masked)
    elif mode == "neg":
        data_masked = np.clip(data_masked, None, 0)
    elif mode == "pos":
        data_masked = np.clip(data_masked, 0, None)

    area = integrate.trapz(data_masked, times[mask], axis=1)

    if average:
        area = area[0]
    area_df = pd.DataFrame({"channels": evoked.ch_names, "area": area})

    return area_df


def get_frac_peak_latency(
    evoked,
    frac=0.5,
    tmin=None,
    tmax=None,
    picks="all",
    mode="abs",
    average=False,
    strict=False,
):
    """Get the latency at which the peak amplitude reaches a certain fraction of its
    maximum value.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked response object.
    frac : float
        The fraction of the peak amplitude at which to compute the latency.
        Defaults to 0.5.
    %(erp_evoked_tmin_tmax)s
    %(picks_all)s
    mode : str
        Specifies how the peak amplitude should be determined. Can be one of:
        - 'abs' : The peak amplitude is the maximum absolute value.
        - 'neg': The peak amplitude is the maximum negative value. If there are no
          negative values and `strict` is True, a ValueError is raised.
        - 'pos': The peak amplitude is the maximum positive value. If there are no
          positive values and `strict` is True, a ValueError is raised.
        Defaults to abs'.
    average : bool
        If True, the fractional peak latency is computed by averaging the data
        across channels before finding the latency. Defaults to False.
    strict : bool
        If True, raise an error if values are all positive when detecting
        a minimum (mode='neg'), or all negative when detecting a maximum
        (mode='pos'). Defaults to False.

    Returns
    -------
    frac_peak_df : pd.DataFrame
        A DataFrame with columns 'channels', 'fractional_peak_onset',
        'fractional_peak_offset', and 'amplitude' containing the latency at which...

    """
    _check_option("mode", mode, ["abs", "neg", "pos"])

    peak_latencies, peak_amplitudes, data_masked, mask, times = _get_peak(
        evoked, tmin, tmax, picks, mode, average, strict
    )
    frac_amplitudes = frac * peak_amplitudes[:, np.newaxis]

    # Find the first time point before the peak where the signal reaches the fractional threshold
    frac_peak_onset = np.argmax(data_masked >= frac_amplitudes, axis=1)
    frac_peak_onset_latency = times[mask][frac_peak_onset]

    # Find the first time point after the peak where the signal reaches the fractional threshold
    peak_idx = np.argmax(data_masked, axis=1)
    frac_peak_offset = np.array(
        [
            peak_idx[i] + np.argmin(data_masked[i, peak_idx[i] :] <= frac_amplitudes[i])
            for i in range(data_masked.shape[0])
        ]
    )
    frac_peak_offset_latency = times[mask][frac_peak_offset]

    frac_peak_df = pd.DataFrame(
        {
            "channels": evoked.ch_names,
            "fractional_peak_onset": frac_peak_onset_latency,
            "fractional_peak_offset": frac_peak_offset_latency,
            "amplitude": peak_amplitudes,
        }
    )

    if average:
        frac_peak_df = frac_peak_df.iloc[0]

    return frac_peak_df


def get_frac_area_latency(
    evoked,
    frac=0.5,
    tmin=None,
    tmax=None,
    picks="all",
    mode="abs",
    average=False,
    strict=False,
):
    """Get the latency at which the area under the curve reaches a certain fraction of its
    maximum value.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked response object.
    frac : float
        The fraction of the area at which to compute the latency. Defaults to 0.5.
    %(erp_evoked_tmin_tmax)s
    %(picks_all)s
    mode : str
        Specifies how the area should be computed. Can be one of:
        - 'abs': The absolute value of the data is used.
        - 'neg': Only negative values are considered.
        - 'pos': Only positive values are considered.
        - 'intg': The integral of the data is computed without rectification.
    Defaults to abs'.
    average : bool
        If True, the fractional area latency is computed by averaging the data
        across channels before finding the latency. Defaults to False.
    %(erp_strict)s


    Returns
    -------
    frac_area_df : pd.DataFrame
        A DataFrame with columns 'channels', 'fractional_area_latency',
        and 'area' containing the latency at which...
    """
    _check_option("mode", mode, ["abs", "neg", "pos", "intg"])
    data = evoked.get_data(picks=picks)
    times = evoked.times
    mask = _time_mask(times, tmin, tmax, evoked.info["sfreq"])
    data_masked = data[:, mask]
    times = times[mask]
    if average:
        data_masked = np.mean(data_masked, axis=0, keepdims=True)
    if mode == "abs":
        data_masked = np.abs(data_masked)
    elif mode == "neg":
        data_masked = np.clip(data_masked, None, 0)
    elif mode == "pos":
        data_masked = np.clip(data_masked, 0, None)
    area = np.trapz(data_masked, times, axis=1)
    frac_area = frac * area
    frac_area_latency = np.full(len(evoked.ch_names), np.nan)
    for ch in range(data_masked.shape[0]):
        idx = np.where(np.cumsum(data_masked[ch]) >= frac_area[ch])[0]
        if len(idx) > 0:
            frac_area_latency[ch] = times[idx[0]]
    frac_area_df = pd.DataFrame(
        {
            "channels": evoked.ch_names,
            "fractional_area_latency": frac_area_latency,
            "area": area,
        }
    )
    if average:
        frac_area_df = frac_area_df.iloc[0]
    return frac_area_df
