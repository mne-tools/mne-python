"""Preprocessing with artifact detection, SSP, and ICA."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

from .flat import mark_flat
from .maxfilter import apply_maxfilter
from .ssp import compute_proj_ecg, compute_proj_eog
from .eog import find_eog_events, create_eog_epochs
from .ecg import find_ecg_events, create_ecg_epochs
from .ica import (ICA, ica_find_eog_events, ica_find_ecg_events,
                  get_score_funcs, read_ica, run_ica, corrmap)
from .otp import oversampled_temporal_projection
from ._peak_finder import peak_finder
from .bads import find_outliers
from .infomax_ import infomax
from .stim import fix_stim_artifact
from .maxwell import maxwell_filter
from .xdawn import Xdawn
from ._csd import compute_current_source_density
from . import nirs
