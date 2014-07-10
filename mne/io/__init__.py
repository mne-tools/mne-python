"""FIF module for IO with .fif files"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from .open import fiff_open, show_fiff, _fiff_get_fid
from .meas_info import read_fiducials, write_fiducials, read_info, write_info

from .proj import proj_equal, make_eeg_average_ref_proj
from . import array
from . import base
from . import brainvision
from . import bti
from . import constants
from . import edf
from . import egi
from . import fiff
from . import kit
from . import pick

from .array import RawArray
from .brainvision import read_raw_brainvision
from .bti import read_raw_bti
from .edf import read_raw_edf
from .egi import read_raw_egi
from .kit import read_raw_kit

# for backward compatibility
from .fiff import RawFIFF
from .fiff import RawFIFF as Raw
from .base import concatenate_raws, get_chpi_positions, set_eeg_reference
