# Authors: Dominik Welke <dominik.welke@mailbox.org>
#
# License: BSD-3-Clause


import numpy as np

from ...io import BaseRaw
from ...io.constants import FIFF
from ...utils import logger, _check_preload, _validate_type, warn


def interpolate_blinks(raw, buffer=0.05, match="BAD_blink"):
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
        all annotations that match any of the strings in the list will be interpolated over.
        Defaults to ``'BAD_blink'``.

    Returns
    -------
    self : instance of Raw
        Returns the modified instance.
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
    _interpolate_blinks(raw, buffer, blink_annots)

    # remove bad from the annotation description
    for desc in match:
        if desc.startswith("BAD_"):
            logger.info("Removing 'BAD_' from {}.".format(desc))
            raw.annotations.rename({desc: desc.replace("BAD_", "")})
    return raw


def _interpolate_blinks(raw, buffer, blink_annots):
    """Interpolate eyetracking signals during blinks in-place."""
    from scipy.interpolate import interp1d

    logger.info("Interpolating missing data during blinks...")
    pre_buffer, post_buffer = buffer
    # iterate over each eyetrack channel and interpolate the blinks
    for i, ch_info in enumerate(raw.info["chs"]):
        if ch_info["kind"] != FIFF.FIFFV_EYETRACK_CH:
            continue
        # Create an empty boolean mask
        mask = np.zeros_like(raw.times, dtype=bool)
        for annot in blink_annots:
            if "ch_names" not in annot or not annot["ch_names"]:
                msg = "blink annotation missing 'ch_names' key. got: {}".format(annot)
                raise ValueError(msg)
            start = annot["onset"] - pre_buffer
            end = annot["onset"] + annot["duration"] + post_buffer
            ch_names = annot["ch_names"]
            if ch_info["ch_name"] not in ch_names:
                continue  # skip if the channel is not in the blink annotation
            # Update the mask for times within the current blink period
            mask |= (raw.times >= start) & (raw.times <= end)
        blink_indices = np.where(mask)[0]
        non_blink_indices = np.where(~mask)[0]

        # Create the interpolation object
        interpolator = interp1d(
            non_blink_indices, raw._data[i, non_blink_indices], kind="linear"
        )
        # Interpolate the blink periods
        interpolated_samples = interpolator(blink_indices)

        # Replace the samples at the blink_indices with the interpolated values
        raw._data[i, blink_indices] = interpolated_samples


# specific function to set eyetrack channels
def set_channel_types_eyetrack(inst, mapping):
    """Define sensor type for eyetrack channels.

    This function can set all eye tracking specific information:
    channel type, unit, eye (and x/y component; only for gaze channels)

    Supported channel types:
    ``'eyegaze'`` and ``'pupil'``

    Supported units:
    ``'au'``, ``'px'``, ``'deg'``, ``'rad'`` (for eyegaze)
    ``'au'``, ``'mm'``, ``'m'`` (for pupil)

    Parameters
    ----------
    inst : instance of Raw, Epochs, or Evoked
        The data instance.
    mapping : dict
        A dictionary mapping a channel to a list/tuple including
        channel type, unit, eye, [and x/y component] (all as str),  e.g.,
        ``{'l_x': ('eyegaze', 'deg', 'left', 'x')}`` or
        ``{'r_pupil': ('pupil', 'au', 'right')}``.

    Returns
    -------
    inst : instance of Raw | Epochs | Evoked
        The instance, modified in place.

    Notes
    -----
    ``inst.set_channel_types()`` to ``'eyegaze'`` or ``'pupil'``
    works as well, but cannot correctly set unit, eye and x/y component.

    Data will be stored in SI units:
    if your data comes in ``deg`` (visual angle) it will be converted to
    ``rad``, if it is in ``mm`` it will be converted to ``m``.
    """
    ch_names = inst.info["ch_names"]

    # allowed
    valid_types = ["eyegaze", "pupil"]  # ch_type
    valid_units = {
        "px": ["px", "pixel"],
        "rad": ["rad", "radian", "radians"],
        "deg": ["deg", "degree", "degrees"],
        "m": ["m", "meter", "meters"],
        "mm": ["mm", "millimeter", "millimeters"],
        "au": [None, "none", "au", "arbitrary"],
    }
    valid_units["all"] = [item for sublist in valid_units.values() for item in sublist]
    valid_eye = {"l": ["left", "l"], "r": ["right", "r"]}
    valid_eye["all"] = [item for sublist in valid_eye.values() for item in sublist]
    valid_xy = {"x": ["x", "h", "horizontal"], "y": ["y", "v", "vertical"]}
    valid_xy["all"] = [item for sublist in valid_xy.values() for item in sublist]

    # loop over channels
    for ch_name, ch_desc in mapping.items():
        if ch_name not in ch_names:
            raise ValueError(
                "This channel name (%s) doesn't exist in " "info." % ch_name
            )
        c_ind = ch_names.index(ch_name)

        # set ch_type and unit
        ch_type = ch_desc[0].lower()
        if ch_type not in valid_types:
            raise ValueError(
                "ch_type must be one of {}. "
                "Got '{}' instead.".format(valid_types, ch_type)
            )
        if ch_type == "eyegaze":
            coil_type = FIFF.FIFFV_COIL_EYETRACK_POS
        elif ch_type == "pupil":
            coil_type = FIFF.FIFFV_COIL_EYETRACK_PUPIL
        inst.info["chs"][c_ind]["coil_type"] = coil_type
        inst.info["chs"][c_ind]["kind"] = FIFF.FIFFV_EYETRACK_CH

        ch_unit = None if (ch_desc[1] is None) else ch_desc[1].lower()
        if ch_unit not in valid_units["all"]:
            raise ValueError(
                "unit must be one of {}. Got '{}' instead.".format(
                    valid_units["all"], ch_unit
                )
            )
        if ch_unit in valid_units["px"]:
            unit_new = FIFF.FIFF_UNIT_PX
        elif ch_unit in valid_units["rad"]:
            unit_new = FIFF.FIFF_UNIT_RAD
        elif ch_unit in valid_units["deg"]:  # convert deg to rad (SI)
            inst = inst.apply_function(_convert_deg_to_rad, picks=ch_name)
            unit_new = FIFF.FIFF_UNIT_RAD
        elif ch_unit in valid_units["m"]:
            unit_new = FIFF.FIFF_UNIT_M
        elif ch_unit in valid_units["mm"]:  # convert mm to m (SI)
            inst = inst.apply_function(_convert_mm_to_m, picks=ch_name)
            unit_new = FIFF.FIFF_UNIT_M
        elif ch_unit in valid_units["au"]:
            unit_new = FIFF.FIFF_UNIT_NONE
        inst.info["chs"][c_ind]["unit"] = unit_new

        # set eye (and x/y-component)
        loc = np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        )

        ch_eye = ch_desc[2].lower()
        if ch_eye not in valid_eye["all"]:
            raise ValueError(
                "eye must be one of {}. Got '{}' instead.".format(
                    valid_eye["all"], ch_eye
                )
            )
        if ch_eye in valid_eye["l"]:
            loc[3] = -1
        elif ch_eye in valid_eye["r"]:
            loc[3] = 1

        if ch_type == "eyegaze":
            ch_xy = ch_desc[3].lower()
            if ch_xy not in valid_xy["all"]:
                raise ValueError(
                    "x/y must be one of {}. Got '{}' instead.".format(
                        valid_xy["all"], ch_xy
                    )
                )
            if ch_xy in valid_xy["x"]:
                loc[4] = -1
            elif ch_xy in valid_xy["y"]:
                loc[4] = 1

        inst.info["chs"][c_ind]["loc"] = loc

    return inst


def _convert_mm_to_m(array):
    return array * 0.001


def _convert_deg_to_rad(array):
    return array * np.pi / 180.0
