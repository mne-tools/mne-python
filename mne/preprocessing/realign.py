# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import pearsonr

from ..io import BaseRaw
from ..utils import _validate_type, logger, verbose, warn


@verbose
def realign_raw(raw, other, t_raw, t_other, *, verbose=None):
    """Realign two simultaneous recordings.

    Due to clock drift, recordings at a given same sample rate made by two
    separate devices simultaneously can become out of sync over time. This
    function uses event times captured by both acquisition devices to resample
    ``other`` to match ``raw``.

    Parameters
    ----------
    raw : instance of Raw
        The first raw instance.
    other : instance of Raw
        The second raw instance. It will be resampled to match ``raw``.
    t_raw : array-like, shape (n_events,)
        The times of shared events in ``raw`` relative to ``raw.times[0]`` (0).
        Typically these could be events on some TTL channel such as::

            find_events(raw)[:, 0] / raw.info["sfreq"] - raw.first_time
    t_other : array-like, shape (n_events,)
        The times of shared events in ``other`` relative to ``other.times[0]``.
    %(verbose)s

    Notes
    -----
    This function operates inplace. It will:

    1. Estimate the zero-order (start offset) and first-order (clock drift)
       correction.
    2. Crop the start of ``raw`` or ``other``, depending on which started
       recording first.
    3. Resample ``other`` to match ``raw`` based on the clock drift.
    4. Realign the onsets and durations in ``other.annotations``.
    5. Crop the end of ``raw`` or ``other``, depending on which stopped
       recording first (and the clock drift rate).

    This function is primarily designed to work on recordings made at the same
    sample rate, but it can also operate on recordings made at different
    sample rates to resample and deal with clock drift simultaneously.

    .. versionadded:: 0.22
    """
    _validate_type(raw, BaseRaw, "raw")
    _validate_type(other, BaseRaw, "other")
    t_raw = np.array(t_raw, float)
    t_other = np.array(t_other, float)
    if t_raw.ndim != 1 or t_raw.shape != t_other.shape:
        raise ValueError(
            "t_raw and t_other must be 1D with the same shape, "
            f"got shapes {t_raw.shape} and {t_other.shape}"
        )
    if len(t_raw) < 20:
        warn("Fewer than 20 times passed, results may be unreliable")

    # 1. Compute correction factors
    poly = Polynomial.fit(x=t_other, y=t_raw, deg=1)
    converted = poly.convert(domain=(-1, 1))
    [zero_ord, first_ord] = converted.coef
    logger.info(
        f"Zero order coefficient: {zero_ord} \nFirst order coefficient: {first_ord}"
    )
    r, p = pearsonr(t_other, t_raw)
    msg = f"Linear correlation computed as R={r:0.3f} and p={p:0.2e}"
    if p > 0.05 or r <= 0:
        raise ValueError(msg + ", cannot resample safely")
    if p > 1e-6:
        warn(msg + ", results may be unreliable")
    else:
        logger.info(msg)
    dr_ms_s = 1000 * abs(1 - first_ord)
    logger.info(
        f"Drift rate: {1000 * dr_ms_s:0.1f} μs/s "
        f"(total drift over {raw.times[-1]:0.1f} s recording: "
        f"{raw.times[-1] * dr_ms_s:0.1f} ms)"
    )

    # 2. Crop start of recordings to match
    if zero_ord > 0:  # need to crop start of raw to match other
        logger.info(f"Cropping {zero_ord:0.3f} s from the start of raw")
        raw.crop(zero_ord, None)
        t_raw -= zero_ord
    elif zero_ord < 0:  # need to crop start of other to match raw
        t_crop = -zero_ord / first_ord
        logger.info(f"Cropping {t_crop:0.3f} s from the start of other")
        other.crop(t_crop, None)
        t_other -= t_crop

    # 3. Resample data using the first-order term
    nan_ch_names = [
        ch for ch in other.info["ch_names"] if np.isnan(other.get_data(picks=ch)).any()
    ]
    if len(nan_ch_names) > 0:  # Issue warning if any channel in other has nan values
        warn(
            f"Channel(s) {', '.join(nan_ch_names)} in `other` contain NaN values. "
            "Resampling these channels will result in the whole channel being NaN. "
            "(If realigning eye-tracking data, consider using interpolate_blinks and "
            "passing interpolate_gaze=True)"
        )
    logger.info("Resampling other")
    sfreq_new = raw.info["sfreq"] * first_ord
    other.load_data().resample(sfreq_new)
    with other.info._unlock():
        other.info["sfreq"] = raw.info["sfreq"]

    # 4. Realign the onsets and durations in other.annotations
    # Must happen before end cropping to avoid losing annotations
    logger.info("Correcting annotations in other")
    other.annotations.onset *= first_ord
    other.annotations.duration *= first_ord

    # 5. Crop the end of one of the recordings if necessary
    delta = raw.times[-1] - other.times[-1]
    msg = f"Cropping {abs(delta):0.3f} s from the end of "
    if delta > 0:
        logger.info(msg + "raw")
        raw.crop(0, other.times[-1])
    elif delta < 0:
        logger.info(msg + "other")
        other.crop(0, raw.times[-1])
