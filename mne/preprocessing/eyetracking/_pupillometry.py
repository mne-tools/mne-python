"""functions for pupillometry analysis.

Authors: Scott Huberty <seh33@uw.edu>
License: BSD-3-Clause
"""

import numpy as np

from ...io import BaseRaw
from ...utils import logger, _check_preload, _validate_type


def interpolate_blinks(raw, buffer=0.025):
    """Interpolate pupil size during blinks.

    This function uses the timing of blink annotations to estimate missing
    pupil size data.

    Parameters
    ----------
    raw : instance of Raw
        The data.
    buffer : float | array-like of float, shape ``(2,))``
        The time in seconds before and after a blink to consider invalid and
        include in the segment to be interpolated over. Default is ``0.025`` seconds
        (25 ms). if array-like, the first element is the time before the blink and the
        second element is the time after the blink to consider invalid, for example,
        ``(0.025, .1)``.

    Returns
    -------
    self : instance of Raw
        The data with interpolated blinks.
    """
    from scipy.interpolate import interp1d

    _check_preload(raw, "interpolate_blinks")
    _validate_type(raw, BaseRaw, "raw")
    _validate_type(buffer, (float, tuple, list, np.ndarray), "buffer")
    # determine the buffer around blinks to include in the interpolation
    buffer = np.array(buffer)
    if buffer.size == 1:
        buffer = np.array([buffer, buffer])

    # get the blink annotations
    blink_annots = [
        annot for annot in raw.annotations if annot["description"].startswith("blink")
    ]
    if not blink_annots:
        logger.info("No blink annotations found. Aborting.")
        return raw
    blink_starts = np.array([annot["onset"] for annot in blink_annots]) - buffer[0]
    blink_ends = (
        np.array([annot["onset"] + annot["duration"] for annot in blink_annots])
        + buffer[1]
    )

    # iterate over each pupil channel and interpolate the blinks
    for i, ch_type in enumerate(raw.get_channel_types()):
        if ch_type != "pupil":
            continue  # skip non-pupil channels
        # Create an empty boolean mask
        mask = np.zeros_like(raw.times, dtype=bool)
        # Iterate over each blink period
        for start, end in zip(blink_starts, blink_ends):
            # Update the mask for times within the current blink period
            mask = np.logical_or(
                mask, np.logical_and(raw.times >= start, raw.times <= end)
            )
        blink_indices = np.where(mask)[0]
        non_blink_indices = np.where(~mask)[0]

        # Create the interpolation object
        interpolator = interp1d(
            non_blink_indices, raw._data[i, non_blink_indices], kind="linear"
        )
        # Interpolate the blink periods
        interpolated_pupil_sizes = interpolator(blink_indices)

        # Replace the pupil size at the blink_indices with the interpolated values
        raw._data[i, blink_indices] = interpolated_pupil_sizes
    return raw
