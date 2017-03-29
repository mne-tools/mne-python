"""IO module for reading raw data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from .open import fiff_open, show_fiff, _fiff_get_fid
from .meas_info import (read_fiducials, write_fiducials, read_info, write_info,
                        _empty_info, _merge_info, _force_update_info, Info,
                        anonymize_info)

from .proj import make_eeg_average_ref_proj, Projection
from .tag import _loc_to_coil_trans, _coil_trans_to_loc, _loc_to_eeg_loc
from .base import BaseRaw

from . import array
from . import base
from . import brainvision
from . import bti
from . import cnt
from . import ctf
from . import constants
from . import edf
from . import egi
from . import fiff
from . import kit
from . import nicolet
from . import eeglab
from . import pick

from .array import RawArray
from .brainvision import read_raw_brainvision
from .bti import read_raw_bti
from .cnt import read_raw_cnt
from .ctf import read_raw_ctf
from .edf import read_raw_edf
from .egi import read_raw_egi
from .kit import read_raw_kit, read_epochs_kit
from .fiff import read_raw_fif
from .nicolet import read_raw_nicolet
from .artemis123 import read_raw_artemis123
from .eeglab import read_raw_eeglab, read_epochs_eeglab

# for backward compatibility
from .fiff import Raw
from .fiff import Raw as RawFIF
from .base import concatenate_raws
from .reference import (set_eeg_reference, set_bipolar_reference,
                        add_reference_channels)
