# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from scipy.interpolate import UnivariateSpline

from ...io import BaseRaw
from ...utils import _validate_type, verbose
from ..nirs import _validate_nirs_info


def _compute_window(seg_length, dt_short, dt_long, fs):
    """Compute window size for spline baseline correction.

    Parameters
    ----------
    seg_length : int
        Length of the segment in samples.
    dt_short : float
        Short time interval in seconds.
    dt_long : float
        Long time interval in seconds.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    wind : int
        Window size in samples (at least 1).
    """
    if seg_length < dt_short * fs:
        wind = seg_length
    elif seg_length < dt_long * fs:
        wind = int(np.floor(dt_short * fs))
    else:
        wind = int(np.floor(seg_length / 10))
    return max(int(wind), 1)


@verbose
def motion_correct_spline(raw, p=0.01, tIncCh=None, *, verbose=None):
    """Apply spline interpolation motion correction to fNIRS data.

    For each detected motion-artifact segment the signal is detrended with a
    smoothing spline, then consecutive segments are baseline-shifted so that
    they connect smoothly.  Based on Homer3 v1.80.2
    ``hmrR_tInc_baselineshift_Ch_Nirs.m`` :footcite:`HuppertEtAl2009` and the
    cedalion reimplementation.

    Parameters
    ----------
    raw : instance of Raw
        The raw fNIRS data (optical density or hemoglobin).
    p : float
        Smoothing parameter for the spline.  Smaller values yield a spline
        that follows the data more closely.  Default is ``0.01``.
    tIncCh : array-like of bool, shape (n_picks, n_times) | None
        Per-channel motion-artifact mask.  ``True`` = clean sample,
        ``False`` = motion artifact.  When ``None`` the entire recording is
        treated as a single motion-artifact segment and only spline detrending
        is applied (no baseline shifting).
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        Data with spline motion correction applied (copy).

    Notes
    -----
    ``n_picks`` is the number of fNIRS channels returned by
    ``_validate_nirs_info``.

    There is a shorter alias ``mne.preprocessing.nirs.spline`` that
    can be used instead of this function.

    References
    ----------
    .. footbibliography::
    """
    _validate_type(raw, BaseRaw, "raw")
    raw = raw.copy().load_data()
    picks = _validate_nirs_info(raw.info)

    if not len(picks):
        raise RuntimeError(
            "Spline motion correction should be run on optical density "
            "or hemoglobin data."
        )

    n_times = raw._data.shape[1]
    fs = raw.info["sfreq"]
    t = np.arange(n_times) / fs

    dt_short = 0.3  # seconds
    dt_long = 3.0  # seconds

    if tIncCh is None:
        tIncCh = np.ones((len(picks), n_times), dtype=bool)
    tIncCh = np.asarray(tIncCh, dtype=bool)

    for ch_idx, pick in enumerate(picks):
        channel = raw._data[pick].copy()
        mask = tIncCh[ch_idx]  # True = good, False = motion

        # Only process if there are actual motion-artifact samples
        lst_ma = np.where(~mask)[0]
        if len(lst_ma) == 0:
            continue

        temp = np.diff(mask.astype(int))
        lst_ms = np.where(temp == -1)[0]  # good→bad  (motion start)
        lst_mf = np.where(temp == 1)[0]  # bad→good  (motion end)

        if len(lst_ms) == 0:
            lst_ms = np.asarray([0])
        if len(lst_mf) == 0:
            lst_mf = np.asarray([n_times - 1])
        if lst_ms[0] > lst_mf[0]:
            lst_ms = np.insert(lst_ms, 0, 0)
        if lst_ms[-1] > lst_mf[-1]:
            lst_mf = np.append(lst_mf, n_times - 1)

        nb_ma = len(lst_ms)
        lst_ml = lst_mf - lst_ms

        dod_spline = channel.copy()

        # ---- Step 1: detrend each motion segment with a spline ----
        for ii in range(nb_ma):
            idx_seg = np.arange(lst_ms[ii], lst_mf[ii])
            if len(idx_seg) > 3:
                spl = UnivariateSpline(t[idx_seg], channel[idx_seg], s=p * len(idx_seg))
                dod_spline[idx_seg] = channel[idx_seg] - spl(t[idx_seg])

        # ---- Step 2: baseline-shift segments to align continuity ----

        # First MA segment
        idx_seg = np.arange(lst_ms[0], lst_mf[0])
        if len(idx_seg) > 0:
            seg_curr_len = lst_ml[0]
            wind_curr = _compute_window(seg_curr_len, dt_short, dt_long, fs)

            if lst_ms[0] > 0:
                seg_prev_len = lst_ms[0]
                wind_prev = _compute_window(seg_prev_len, dt_short, dt_long, fs)
                mean_prev = np.mean(
                    dod_spline[max(0, idx_seg[0] - wind_prev) : idx_seg[0]]
                )
                mean_curr = np.mean(dod_spline[idx_seg[0] : idx_seg[0] + wind_curr])
                dod_spline[idx_seg] = dod_spline[idx_seg] - mean_curr + mean_prev
            else:
                seg_next_len = (
                    (lst_ms[1] - lst_mf[0]) if nb_ma > 1 else (n_times - lst_mf[0])
                )
                wind_next = _compute_window(seg_next_len, dt_short, dt_long, fs)
                mean_next = np.mean(dod_spline[idx_seg[-1] : idx_seg[-1] + wind_next])
                mean_curr = np.mean(
                    dod_spline[max(0, idx_seg[-1] - wind_curr) : idx_seg[-1]]
                )
                dod_spline[idx_seg] = dod_spline[idx_seg] - mean_curr + mean_next

        # Intermediate non-MA and MA segments
        for kk in range(nb_ma - 1):
            # Non-motion segment between MA[kk] and MA[kk+1]
            idx_seg = np.arange(lst_mf[kk], lst_ms[kk + 1])
            seg_prev_len = lst_ml[kk]
            seg_curr_len = len(idx_seg)

            wind_prev = _compute_window(seg_prev_len, dt_short, dt_long, fs)
            wind_curr = _compute_window(seg_curr_len, dt_short, dt_long, fs)

            mean_prev = np.mean(dod_spline[max(0, idx_seg[0] - wind_prev) : idx_seg[0]])
            mean_curr = np.mean(channel[idx_seg[0] : idx_seg[0] + wind_curr])
            dod_spline[idx_seg] = channel[idx_seg] - mean_curr + mean_prev

            # Next MA segment
            idx_seg = np.arange(lst_ms[kk + 1], lst_mf[kk + 1])
            seg_prev_len = seg_curr_len
            seg_curr_len = lst_ml[kk + 1]

            wind_prev = _compute_window(seg_prev_len, dt_short, dt_long, fs)
            wind_curr = _compute_window(seg_curr_len, dt_short, dt_long, fs)

            mean_prev = np.mean(dod_spline[max(0, idx_seg[0] - wind_prev) : idx_seg[0]])
            mean_curr = np.mean(dod_spline[idx_seg[0] : idx_seg[0] + wind_curr])
            dod_spline[idx_seg] = dod_spline[idx_seg] - mean_curr + mean_prev

        # Last non-MA segment (after the final motion segment)
        if lst_mf[-1] < n_times:
            idx_seg = np.arange(lst_mf[-1], n_times)
            seg_prev_len = lst_ml[-1]
            seg_curr_len = len(idx_seg)

            wind_prev = _compute_window(seg_prev_len, dt_short, dt_long, fs)
            wind_curr = _compute_window(seg_curr_len, dt_short, dt_long, fs)

            mean_prev = np.mean(dod_spline[max(0, idx_seg[0] - wind_prev) : idx_seg[0]])
            mean_curr = np.mean(channel[idx_seg[0] : idx_seg[0] + wind_curr])
            dod_spline[idx_seg] = channel[idx_seg] - mean_curr + mean_prev

        raw._data[pick] = dod_spline

    return raw


# provide a short alias
spline = motion_correct_spline
