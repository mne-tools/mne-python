import numpy as np
from scipy import linalg

import logging
logger = logging.getLogger('mne')

import mne

from ..fiff.pick import pick_channels_cov
from ..forward import _subject_from_forward
from ..cov import compute_whitener
from ..source_estimate import SourceEstimate
from ._lcmv import _prepare_beamformer_input


