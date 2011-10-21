# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

__version__ = '0.1.git'

from .constants import FIFF
from .open import fiff_open
from .evoked import Evoked, read_evoked, write_evoked
from .raw import Raw, read_raw_segment, read_raw_segment_times, \
                 start_writing_raw, write_raw_buffer, finish_writing_raw
from .pick import pick_types, pick_channels, pick_types_evoked
from .compensator import get_current_comp
from .proj import compute_spatial_vectors
