"""Preprocessing with artifact detection, SSP, and ICA."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

from .maxfilter import apply_maxfilter
from .ssp import compute_proj_ecg, compute_proj_eog
from .eog import find_eog_events, create_eog_epochs
from .ecg import find_ecg_events, create_ecg_epochs
from .ica import (ICA, ica_find_eog_events, ica_find_ecg_events,
                  get_score_funcs, read_ica, run_ica, corrmap)
from .bads import find_outliers
from .infomax_ import infomax
from .stim import fix_stim_artifact
from .maxwell import maxwell_filter
from .xdawn import Xdawn
