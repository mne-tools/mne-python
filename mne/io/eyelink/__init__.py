"""Module for loading Eye-Tracker data."""

# Authors: Dominik Welke <dominik.welke@web.de>
#          Scott Huberty <seh33@uw.edu>
#
# License: BSD-3-Clause

from .eyelink import read_raw_eyelink, read_eyelink_calibration
from .calibration import Calibration
