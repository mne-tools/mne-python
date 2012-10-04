"""FIF module for IO with .fif files"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from .constants import FIFF
from .open import fiff_open
from .evoked import Evoked, read_evoked, write_evoked
from .raw import Raw, read_raw_segment, read_raw_segment_times, \
                 start_writing_raw, write_raw_buffer, finish_writing_raw, \
                 concatenate_raws
from .pick import pick_types, pick_channels, pick_types_evoked, \
                  pick_channels_regexp, pick_channels_forward, \
                  pick_types_forward, pick_channels_cov

from .compensator import get_current_comp
from .proj import compute_spatial_vectors, proj_equal, \
                  make_eeg_average_ref_proj
from .cov import read_cov, write_cov
