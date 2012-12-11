"""Preprocessing module"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from .maxfilter import apply_maxfilter
from .ssp import compute_proj_ecg, compute_proj_eog
from .eog import find_eog_events
from .ecg import find_ecg_events
from .ica import ICA, ica_find_eog_events, ica_find_ecg_events, score_funcs, \
                 read_ica
