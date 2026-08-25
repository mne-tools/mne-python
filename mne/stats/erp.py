"""ERP-related statistics."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from scipy import integrate

from mne._fiff.pick import _picks_to_idx
from mne.utils import (
    _check_option,
    _check_pandas_installed,
    _time_mask,
    _validate_type,
    fill_doc,
    warn,
)


def compute_sme(epochs, start=None, stop=None):
    """Compute standardized measurement error (SME).

    The standardized measurement error :footcite:`LuckEtAl2021` can be used as a
    universal measure of data quality in ERP studies.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs containing the data for which to compute the SME.
    start : int | float | None
        Start time (in s) of the time window used for SME computation. If ``None``, use
        the start of the epoch.
    stop : int | float | None
        Stop time (in s) of the time window used for SME computation. If ``None``, use
        the end of the epoch.

    Returns
    -------
    sme : array, shape (n_channels,)
        SME in given time window for each channel.

    Notes
    -----
    Currently, only the mean value in the given time window is supported, meaning that
    the resulting SME is only valid in studies which quantify the amplitude of an ERP
    component as the mean within the time window (as opposed to e.g. the peak, which
    would require bootstrapping).

    References
    ----------
    .. footbibliography::

    Examples
    --------
    Given an :class:`~mne.Epochs` object, the SME for the entire epoch duration can be
    computed as follows:

        >>> compute_sme(epochs)  # doctest: +SKIP

    However, the SME is best used to estimate the precision of a specific ERP measure,
    specifically the mean amplitude of an ERP component in a time window of interest.
    For example, the SME for the mean amplitude of the P3 component in the 300-500 ms
    time window could be computed as follows:

        >>> compute_sme(epochs, start=0.3, stop=0.5)  # doctest: +SKIP

    Usually, it will be more informative to compute the SME for specific conditions
    separately. This can be done by selecting the epochs of interest as follows:

        >>> compute_sme(epochs["oddball"], 0.3, 0.5)  # doctest: +SKIP

    Note that the SME will be reported for each channel separately. If you are only
    interested in a single channel (or a subset of channels), select the channels
    before computing the SME:

        >>> compute_sme(epochs.pick("Pz"), 0.3, 0.5)  # doctest: +SKIP

    Selecting both conditions and channels is also possible:

        >>> compute_sme(epochs["oddball"].pick("Pz"), 0.3, 0.5)  # doctest: +SKIP

    In any case, the output will be a NumPy array with the SME value for each channel.
    """
    _validate_type(start, ("numeric", None), "start", "int or float")
    _validate_type(stop, ("numeric", None), "stop", "int or float")
    start = epochs.tmin if start is None else start
    stop = epochs.tmax if stop is None else stop
    if start < epochs.tmin:
        raise ValueError("start is out of bounds.")
    if stop > epochs.tmax:
        raise ValueError("stop is out of bounds.")

    data = epochs.get_data(tmin=start, tmax=stop)
    return data.mean(axis=2).std(axis=0) / np.sqrt(data.shape[0])


def _compute_peak(
    evoked, start=None, stop=None, picks="all", mode="abs", average=False, strict=True
):
    """Locate the peak shared by compute_peak and compute_frac_peak_latency."""
    data = evoked.get_data(picks=picks)
    picked_idx = _picks_to_idx(evoked.info, picks, "all", exclude=())
    ch_names = [evoked.ch_names[i] for i in picked_idx]
    times = evoked.times
    mask = _time_mask(times, start, stop, evoked.info["sfreq"])
    data_masked = data[:, mask]

    if average:
        data = np.mean(data, axis=0, keepdims=True)
        data_masked = np.mean(data_masked, axis=0, keepdims=True)
        ch_names = ["Average"]

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

    return peak_latencies, peak_amplitudes, data_masked, mask, times, ch_names


@fill_doc
def compute_peak(
    evoked,
    start=None,
    stop=None,
    picks="all",
    mode="abs",
    average=False,
    strict=True,
):
    """Compute the peak amplitude and latency of an evoked response.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked response object.
    %(erp_evoked_start_stop)s
    %(picks_all)s
    mode : str
        Specifies how the peak amplitude should be determined. Can be one of:

        ``'abs'``
            The peak amplitude is the maximum absolute value.
        ``'neg'``
            The peak amplitude is the maximum negative value. If there are
            no negative values and ``strict`` is True, a ValueError is raised.
        ``'pos'``
            The peak amplitude is the maximum positive value. If there are
            no positive values and ``strict`` is True, a ValueError is raised.

        Defaults to ``'abs'``.
    average : bool
        If True, the peak amplitude is computed by averaging the data across
        channels before finding the peak. Defaults to False.
    %(erp_strict)s

    Returns
    -------
    peak_df : pandas.DataFrame
        A DataFrame with columns 'channel', 'latency', and 'amplitude'
        containing the peak amplitude and latency for each channel. If
        ``average=True``, contains a single row whose 'channel' value is
        ``'Average'``.
    """
    pd = _check_pandas_installed(strict=True)
    _check_option("mode", mode, ["abs", "neg", "pos"])
    peak_latencies, peak_amplitudes, _, _, _, channel = _compute_peak(
        evoked, start, stop, picks, mode, average, strict
    )

    peak_df = pd.DataFrame(
        {
            "channel": channel,
            "latency": peak_latencies,
            "amplitude": peak_amplitudes,
        }
    )

    return peak_df


@fill_doc
def compute_area(
    evoked,
    start=None,
    stop=None,
    picks="all",
    mode="abs",
    average=False,
):
    """
    Compute the area under the curve of an evoked response within a given time window.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked response object.
    %(erp_evoked_start_stop)s
    %(picks_all)s
    mode : str
        Specifies how the area should be computed. Can be one of:

        ``'abs'``
            The absolute value of the data is used.
        ``'neg'``
            Only negative values are considered.
        ``'pos'``
            Only positive values are considered.
        ``'intg'``
            The integral of the data is computed without rectification.

        Defaults to ``'abs'``.
    average : bool
        If True, the area is computed by averaging the data across channels
        before integration. Defaults to False.

    Returns
    -------
    area_df : pandas.DataFrame
        A DataFrame with columns 'channel' and 'area' containing the area
        under the curve for each channel. If ``average=True``, contains a
        single row whose 'channel' value is ``'Average'``.
    """
    pd = _check_pandas_installed(strict=True)
    _check_option("mode", mode, ["abs", "neg", "pos", "intg"])
    data = evoked.get_data(picks=picks)
    picked_idx = _picks_to_idx(evoked.info, picks, "all", exclude=())
    channel = [evoked.ch_names[i] for i in picked_idx]
    times = evoked.times
    mask = _time_mask(times, start, stop, evoked.info["sfreq"])
    data_masked = data[:, mask]

    if average:
        data_masked = np.mean(data_masked, axis=0, keepdims=True)
        channel = ["Average"]
    if mode == "abs":
        data_masked = np.abs(data_masked)
    elif mode == "neg":
        data_masked = np.clip(data_masked, None, 0)
    elif mode == "pos":
        data_masked = np.clip(data_masked, 0, None)

    area = integrate.trapezoid(data_masked, times[mask], axis=1)
    area_df = pd.DataFrame({"channel": channel, "area": area})

    return area_df


@fill_doc
def compute_frac_peak_latency(
    evoked,
    frac=0.5,
    start=None,
    stop=None,
    picks="all",
    mode="abs",
    average=False,
    strict=True,
):
    """Compute the latency at which a fraction of the peak amplitude is reached.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked response object.
    frac : float
        The fraction of the peak amplitude at which to compute the latency.
        Defaults to 0.5.
    %(erp_evoked_start_stop)s
    %(picks_all)s
    mode : str
        Specifies how the peak amplitude should be determined. Can be one of:

        ``'abs'``
            The peak amplitude is the maximum absolute value.
        ``'neg'``
            The peak amplitude is the maximum negative value. If there are
            no negative values and ``strict`` is True, a ValueError is raised.
        ``'pos'``
            The peak amplitude is the maximum positive value. If there are
            no positive values and ``strict`` is True, a ValueError is raised.

        Defaults to ``'abs'``.
    average : bool
        If True, the fractional peak latency is computed by averaging the data
        across channels before finding the latency. Defaults to False.
    %(erp_strict)s

    Returns
    -------
    frac_peak_df : pandas.DataFrame
        A DataFrame with columns 'channel', 'fractional_peak_onset',
        'fractional_peak_offset', and 'amplitude' containing the latency at
        which the peak amplitude reaches the fractional threshold. If
        ``average=True``, contains a single row whose 'channel' value is
        ``'Average'``.
    """
    pd = _check_pandas_installed(strict=True)
    _check_option("mode", mode, ["abs", "neg", "pos"])

    _, peak_amplitudes, data_masked, mask, times, channel = _compute_peak(
        evoked, start, stop, picks, mode, average, strict
    )

    peak_idx = np.argmax(data_masked, axis=1)
    transformed_peak = data_masked[np.arange(data_masked.shape[0]), peak_idx]
    frac_amplitudes = frac * transformed_peak[:, np.newaxis]

    # Find the first time point before the peak where the signal reaches the
    # fractional threshold
    frac_peak_onset = np.argmax(data_masked >= frac_amplitudes, axis=1)
    frac_peak_onset_latency = times[mask][frac_peak_onset]

    # Find the first time point after the peak where the signal falls back to
    # the fractional threshold; NaN if it never does before the window ends
    frac_peak_offset_latency = np.full(data_masked.shape[0], np.nan)
    nan_channels = []
    for i in range(data_masked.shape[0]):
        below_threshold = np.where(data_masked[i, peak_idx[i] :] <= frac_amplitudes[i])[
            0
        ]
        if len(below_threshold) > 0:
            frac_peak_offset_latency[i] = times[mask][peak_idx[i] + below_threshold[0]]
        else:
            nan_channels.append(channel[i])
    if nan_channels:
        warn(
            f"The signal never fell back below the fractional threshold before "
            f"the end of the window for {len(nan_channels)} channel(s) "
            f"({', '.join(nan_channels)}); fractional_peak_offset is NaN for "
            "these channels."
        )

    frac_peak_df = pd.DataFrame(
        {
            "channel": channel,
            "fractional_peak_onset": frac_peak_onset_latency,
            "fractional_peak_offset": frac_peak_offset_latency,
            "amplitude": peak_amplitudes,
        }
    )

    return frac_peak_df


@fill_doc
def compute_frac_area_latency(
    evoked,
    frac=0.5,
    start=None,
    stop=None,
    picks="all",
    mode="abs",
    average=False,
):
    """Compute the latency at which a fraction of the total area is reached.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked response object.
    frac : float
        The fraction of the area at which to compute the latency. Defaults to 0.5.
    %(erp_evoked_start_stop)s
    %(picks_all)s
    mode : str
        Specifies how the area should be computed. Can be one of:

        ``'abs'``
            The absolute value of the data is used.
        ``'neg'``
            Only negative values are considered.
        ``'pos'``
            Only positive values are considered.
        ``'intg'``
            The integral of the data is computed without rectification.

        Defaults to ``'abs'``.
    average : bool
        If True, the fractional area latency is computed by averaging the data
        across channels before finding the latency. Defaults to False.

    Returns
    -------
    frac_area_df : pandas.DataFrame
        A DataFrame with columns 'channel', 'fractional_area_latency',
        and 'area' containing the latency at which the area under the curve
        reaches the fractional threshold. If ``average=True``, contains a
        single row whose 'channel' value is ``'Average'``.

    Notes
    -----
    With ``mode='intg'`` the running signed area is not guaranteed to
    increase monotonically, so for a channel whose positive and negative
    portions nearly cancel, the reported latency may not correspond to any
    visually meaningful point in the waveform. Only a channel whose total
    area is *exactly* zero is guarded against (yielding ``NaN``); a total
    area that is merely small relative to the channel's overall activity is
    not. The earliest sample satisfying the fractional threshold is
    returned.
    """
    pd = _check_pandas_installed(strict=True)
    _check_option("mode", mode, ["abs", "neg", "pos", "intg"])
    data = evoked.get_data(picks=picks)
    picked_idx = _picks_to_idx(evoked.info, picks, "all", exclude=())
    channel = [evoked.ch_names[i] for i in picked_idx]
    times = evoked.times
    mask = _time_mask(times, start, stop, evoked.info["sfreq"])
    data_masked = data[:, mask]
    times = times[mask]
    if average:
        data_masked = np.mean(data_masked, axis=0, keepdims=True)
        channel = ["Average"]
    if mode == "abs":
        data_masked = np.abs(data_masked)
    elif mode == "neg":
        data_masked = np.clip(data_masked, None, 0)
    elif mode == "pos":
        data_masked = np.clip(data_masked, 0, None)

    cum_area = integrate.cumulative_trapezoid(data_masked, times, axis=1, initial=0)
    area = cum_area[:, -1]

    frac_area_latency = np.full(data_masked.shape[0], np.nan)
    nan_channels = []
    for ch in range(data_masked.shape[0]):
        if area[ch] == 0:
            # Nothing accumulated; no latency can be defined
            nan_channels.append(channel[ch])
            continue
        # Normalize
        idx = np.where(cum_area[ch] / area[ch] >= frac)[0]
        if len(idx) > 0:
            frac_area_latency[ch] = times[idx[0]]
    if nan_channels:
        warn(
            f"No area was accumulated for {len(nan_channels)} channel(s) "
            f"({', '.join(nan_channels)}); fractional_area_latency is NaN "
            "for these channels."
        )

    frac_area_df = pd.DataFrame(
        {
            "channel": channel,
            "fractional_area_latency": frac_area_latency,
            "area": area,
        }
    )
    return frac_area_df
