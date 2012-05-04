"""Preprocessing module"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from . maxfilter import apply_maxfilter
from . ssp import compute_proj_ecg, compute_proj_eog
