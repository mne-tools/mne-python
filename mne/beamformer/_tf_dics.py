"""5D time-frequency beamforming based on DICS
"""

# Authors: Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

import warnings

import numpy as np

import logging
logger = logging.getLogger('mne')

from ..time_frequency import compute_epochs_csd
from ..source_estimate import SourceEstimate
from .. import verbose
from . import dics_source_power


