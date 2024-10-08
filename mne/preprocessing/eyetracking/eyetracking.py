# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


import numpy as np

from ..._fiff.constants import FIFF
from ...epochs import BaseEpochs
from ...evoked import Evoked
from ...io import BaseRaw
from ...utils import _check_option, _validate_type, logger, warn
from .calibration import Calibration
from .utils import _check_calibration


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
            raise ValueError(f"This channel name ({ch_name}) doesn't exist in info.")
        c_ind = ch_names.index(ch_name)

        # set ch_type and unit
        ch_type = ch_desc[0].lower()
        if ch_type not in valid_types:
            raise ValueError(
                f"ch_type must be one of {valid_types}. Got '{ch_type}' instead."
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


def convert_units(inst, calibration, to="radians"):
    """Convert Eyegaze data from pixels to radians of visual angle or vice versa.

    .. warning::
        Currently, depending on the units (pixels or radians), eyegaze channels may not
        be reported correctly in visualization functions like :meth:`mne.io.Raw.plot`.
        They will be shown  correctly in :func:`mne.viz.eyetracking.plot_gaze`.
        See :gh:`11879` for more information.

    .. Important::
       There are important considerations to keep in mind when using this function,
       see the Notes section below.

    Parameters
    ----------
    inst : instance of Raw, Epochs, or Evoked
        The Raw, Epochs, or Evoked instance with eyegaze channels.
    calibration : Calibration
        Instance of  Calibration, containing information about the screen size
        (in meters), viewing distance (in meters), and the screen resolution
        (in pixels).
    to : str
        Must be either ``"radians"`` or ``"pixels"``, indicating the desired unit.

    Returns
    -------
    inst : instance of Raw | Epochs | Evoked
        The Raw, Epochs, or Evoked instance, modified in place.

    Notes
    -----
    There are at least two important considerations to keep in mind when using this
    function:

    1. Converting between on-screen pixels and visual angle is not a linear
       transformation. If the visual angle subtends less than approximately ``.44``
       radians (``25`` degrees), the conversion could be considered to be approximately
       linear. However, as the visual angle increases, the conversion becomes
       increasingly non-linear. This may lead to unexpected results after converting
       between pixels and visual angle.

    * This function assumes that the head is fixed in place and aligned with the center
      of the screen, such that gaze to the center of the screen results in a visual
      angle of ``0`` radians.

    .. versionadded:: 1.7
    """
    _validate_type(inst, (BaseRaw, BaseEpochs, Evoked), "inst")
    _validate_type(calibration, Calibration, "calibration")
    _check_option("to", to, ("radians", "pixels"))
    _check_calibration(calibration)

    # get screen parameters
    screen_size = calibration["screen_size"]
    screen_resolution = calibration["screen_resolution"]
    dist = calibration["screen_distance"]

    # loop through channels and convert units
    converted_chs = []
    for ch_dict in inst.info["chs"]:
        if ch_dict["coil_type"] != FIFF.FIFFV_COIL_EYETRACK_POS:
            continue
        unit = ch_dict["unit"]
        name = ch_dict["ch_name"]

        if ch_dict["loc"][4] == -1:  # x-coordinate
            size = screen_size[0]
            res = screen_resolution[0]
        elif ch_dict["loc"][4] == 1:  # y-coordinate
            size = screen_size[1]
            res = screen_resolution[1]
        else:
            raise ValueError(
                f"loc array not set properly for channel '{name}'. Index 4 should"
                f"  be -1 or 1, but got {ch_dict['loc'][4]}"
            )
        # check unit, convert, and set new unit
        if to == "radians":
            if unit != FIFF.FIFF_UNIT_PX:
                raise ValueError(
                    f"Data must be in pixels in order to convert to radians."
                    f" Got {unit} for {name}"
                )
            inst.apply_function(_pix_to_rad, picks=name, size=size, res=res, dist=dist)
            ch_dict["unit"] = FIFF.FIFF_UNIT_RAD
        elif to == "pixels":
            if unit != FIFF.FIFF_UNIT_RAD:
                raise ValueError(
                    f"Data must be in radians in order to convert to pixels."
                    f" Got {unit} for {name}"
                )
            inst.apply_function(_rad_to_pix, picks=name, size=size, res=res, dist=dist)
            ch_dict["unit"] = FIFF.FIFF_UNIT_PX
        converted_chs.append(name)
    if converted_chs:
        logger.info(f"Converted {converted_chs} to {to}.")
        if to == "radians":
            # check if any values are greaater than .44 radians
            # (25 degrees) and warn user
            data = inst.get_data(picks=converted_chs)
            if np.any(np.abs(data) > 0.52):
                warn(
                    "Some visual angle values subtend greater than .52 radians "
                    "(30 degrees), meaning that the conversion between pixels "
                    "and visual angle may be very non-linear. Take caution when "
                    "interpreting these values. Max visual angle value in data:"
                    f" {np.nanmax(data):0.2f} radians.",
                    UserWarning,
                )
    else:
        warn("Could not find any eyegaze channels. Doing nothing.", UserWarning)
    return inst


def _pix_to_rad(data, size, res, dist):
    """Convert pixel coordinates to radians of visual angle.

    Parameters
    ----------
    data : array-like, shape (n_samples,)
        A vector of pixel coordinates.
    size : float
        The width or height of the screen, in meters.
    res : int
        The screen resolution in pixels, along the x or y axis.
    dist : float
        The viewing distance from the screen, in meters.

    Returns
    -------
    rad : ndarray, shape (n_samples)
        the data in radians.
    """
    # Center the data so that 0 radians will be the center of the screen
    data -= res / 2
    # How many meters is the pixel width or height
    px_size = size / res
    # Convert to radians
    return np.arctan((data * px_size) / dist)


def _rad_to_pix(data, size, res, dist):
    """Convert radians of visual angle to pixel coordinates.

    See the parameters section of _pix_to_rad for more information.

    Returns
    -------
    pix : ndarray, shape (n_samples)
        the data in pixels.
    """
    # How many meters is the pixel width or height
    px_size = size / res
    # 1. calculate length of opposite side of triangle (in meters)
    # 2. convert meters to pixel coordinates
    # 3. add half of screen resolution to uncenter the pixel data (0,0 is top left)
    return np.tan(data) * dist / px_size + res / 2
