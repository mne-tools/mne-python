"""Eye tracking specific preprocessing functions."""

# Authors: Dominik Welke <dominik.welke@mailbox.org>
#
# License: BSD-3-Clause

from .eyetracking import set_channel_types_eyetrack
from .calibration import Calibration, read_eyelink_calibration
from ._pupillometry import interpolate_blinks
