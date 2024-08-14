# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ...utils import _validate_type
from .calibration import Calibration


def _check_calibration(
    calibration, want_keys=("screen_size", "screen_resolution", "screen_distance")
):
    missing_keys = []
    for key in want_keys:
        if calibration.get(key, None) is None:
            missing_keys.append(key)

    if missing_keys:
        raise KeyError(
            "Calibration object must have the following keys with valid values:"
            f" {', '.join(missing_keys)}"
        )
    else:
        return True


def get_screen_visual_angle(calibration):
    """Calculate the radians of visual angle that the participant screen subtends.

    Parameters
    ----------
    calibration : Calibration
        An instance of Calibration. Must have valid values for ``"screen_size"`` and
        ``"screen_distance"`` keys.

    Returns
    -------
    visual angle in radians : ndarray, shape (2,)
        The visual angle of the monitor width and height, respectively.
    """
    _validate_type(calibration, Calibration, "calibration")
    _check_calibration(calibration, want_keys=("screen_size", "screen_distance"))
    size = np.array(calibration["screen_size"])
    return 2 * np.arctan(size / (2 * calibration["screen_distance"]))
