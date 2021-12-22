"""Preprocessing with artifact detection, SSP, and ICA."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD-3-Clause

from .flat import annotate_flat
from .maxfilter import apply_maxfilter
from .ssp import compute_proj_ecg, compute_proj_eog
from .eog import find_eog_events, create_eog_epochs
from .ecg import find_ecg_events, create_ecg_epochs
from .ica import (ICA, ica_find_eog_events, ica_find_ecg_events,
                  get_score_funcs, read_ica, corrmap, read_ica_eeglab)
from .otp import oversampled_temporal_projection
from ._peak_finder import peak_finder
from .infomax_ import infomax
from .stim import fix_stim_artifact
from .maxwell import (maxwell_filter, find_bad_channels_maxwell,
                      compute_maxwell_basis)
from .realign import realign_raw
from .xdawn import Xdawn
from ._csd import compute_current_source_density
from . import nirs
from .artifact_detection import (annotate_movement, compute_average_dev_head_t,
                                 annotate_muscle_zscore, annotate_break)
from ._regress import regress_artifact
from ._fine_cal import (compute_fine_calibration,  read_fine_calibration,
                        write_fine_calibration)
from .annotate_nan import annotate_nan
from .interpolate import equalize_bads
from . import ieeg
from ._css import cortical_signal_suppression
