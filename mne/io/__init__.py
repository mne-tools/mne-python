"""IO module for reading raw data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#
# License: BSD-3-Clause

from .open import fiff_open, show_fiff, _fiff_get_fid
from .meas_info import (read_fiducials, write_fiducials, read_info, write_info,
                        _empty_info, _merge_info, _force_update_info, Info,
                        anonymize_info, _writing_info_hdf5)

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
from . import nirx
from . import boxy
from . import persyst
from . import eeglab
from . import pick
from . import nihon

from .array import RawArray
from .besa import read_evoked_besa
from .brainvision import read_raw_brainvision
from .bti import read_raw_bti
from .cnt import read_raw_cnt
from .ctf import read_raw_ctf
from .curry import read_raw_curry
from .edf import read_raw_edf, read_raw_bdf, read_raw_gdf
from .egi import read_raw_egi, read_evokeds_mff
from .kit import read_raw_kit, read_epochs_kit
from .fiff import read_raw_fif
from .nedf import read_raw_nedf
from .nicolet import read_raw_nicolet
from .artemis123 import read_raw_artemis123
from .eeglab import read_raw_eeglab, read_epochs_eeglab
from .eximia import read_raw_eximia
from .hitachi import read_raw_hitachi
from .nirx import read_raw_nirx
from .boxy import read_raw_boxy
from .snirf import read_raw_snirf
from .persyst import read_raw_persyst
from .fieldtrip import (read_raw_fieldtrip, read_epochs_fieldtrip,
                        read_evoked_fieldtrip)
from .nihon import read_raw_nihon
from ._read_raw import read_raw

# for backward compatibility
from .fiff import Raw
from .fiff import Raw as RawFIF
from .base import concatenate_raws
from .reference import (set_eeg_reference, set_bipolar_reference,
                        add_reference_channels)
