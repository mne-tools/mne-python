# Authors: Scott Huberty <seh33@uw.edu>
#
# License: BSD-3-Clause

import numpy as np

from ...io import BaseRaw
from ...io.constants import FIFF
from ...utils import logger, _check_preload, _validate_type, warn


def interpolate_blinks(raw, buffer=0.05, match="BAD_blink", interpolate_gaze=False):
    """Interpolate eyetracking signals during blinks.

    This function uses the timing of blink annotations to estimate missing
    data. Operates in place.

    Parameters
    ----------
    raw : instance of Raw
        The raw data with at least one ``'pupil'`` or ``'eyegaze'`` channel.
    buffer : float | array-like of float, shape ``(2,))``
        The time in seconds before and after a blink to consider invalid and
        include in the segment to be interpolated over. Default is ``0.05`` seconds
        (50 ms). If array-like, the first element is the time before the blink and the
        second element is the time after the blink to consider invalid, for example,
        ``(0.025, .1)``.
    match : str | list of str
        The description of annotations to interpolate over. If a list, the data within
        all annotations that match any of the strings in the list will be interpolated
        over. Defaults to ``'BAD_blink'``.
    interpolate_gaze : bool
        If False, only apply interpolation to ``'pupil channels'``. If True, interpolate
        over ``'eyegaze'`` channels as well. Defaults to False, because eye position can
        change in unpredictable ways during blinks.

    Returns
    -------
    self : instance of Raw
        Returns the modified instance.

    Notes
    -----
    .. versionadded:: 1.5
    """
    _check_preload(raw, "interpolate_blinks")
    _validate_type(raw, BaseRaw, "raw")
    _validate_type(buffer, (float, tuple, list, np.ndarray), "buffer")
    _validate_type(match, (str, tuple, list, np.ndarray), "match")

    # determine the buffer around blinks to include in the interpolation
    buffer = np.array(buffer, dtype=float)
    if buffer.size == 1:
        buffer = np.array([buffer, buffer])

    if isinstance(match, str):
        match = [match]

    # get the blink annotations
    blink_annots = [annot for annot in raw.annotations if annot["description"] in match]
    if not blink_annots:
        warn("No annotations matching {} found. Aborting.".format(match))
        return raw
    _interpolate_blinks(raw, buffer, blink_annots, interpolate_gaze=interpolate_gaze)

    # remove bad from the annotation description
    for desc in match:
        if desc.startswith("BAD_"):
            logger.info("Removing 'BAD_' from {}.".format(desc))
            raw.annotations.rename({desc: desc.replace("BAD_", "")})
    return raw


def _interpolate_blinks(raw, buffer, blink_annots, interpolate_gaze):
    """Interpolate eyetracking signals during blinks in-place."""
    logger.info("Interpolating missing data during blinks...")
    pre_buffer, post_buffer = buffer
    # iterate over each eyetrack channel and interpolate the blinks
    for ci, ch_info in enumerate(raw.info["chs"]):
        if interpolate_gaze:  # interpolate over all eyetrack channels
            if ch_info["kind"] != FIFF.FIFFV_EYETRACK_CH:
                continue
        else:  # interpolate over pupil channels only
            if ch_info["coil_type"] != FIFF.FIFFV_COIL_EYETRACK_PUPIL:
                continue
        # Create an empty boolean mask
        mask = np.zeros_like(raw.times, dtype=bool)
        for annot in blink_annots:
            if "ch_names" not in annot or not annot["ch_names"]:
                msg = f"Blink annotation missing values for 'ch_names' key: {annot}"
                raise ValueError(msg)
            start = annot["onset"] - pre_buffer
            end = annot["onset"] + annot["duration"] + post_buffer
            if ch_info["ch_name"] not in annot["ch_names"]:
                continue  # skip if the channel is not in the blink annotation
            # Update the mask for times within the current blink period
            mask |= (raw.times >= start) & (raw.times <= end)
        blink_indices = np.where(mask)[0]
        non_blink_indices = np.where(~mask)[0]

        # Linear interpolation
        interpolated_samples = np.interp(
            raw.times[blink_indices],
            raw.times[non_blink_indices],
            raw._data[ci, non_blink_indices],
        )
        # Replace the samples at the blink_indices with the interpolated values
        raw._data[ci, blink_indices] = interpolated_samples
