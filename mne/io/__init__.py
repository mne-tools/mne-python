"""IO module for reading raw data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from .open import fiff_open, show_fiff, _fiff_get_fid
from .meas_info import (read_fiducials, write_fiducials, read_info, write_info,
                        _empty_info, _merge_info, _force_update_info, Info,
                        anonymize_info, _stamp_to_dt)

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
from . import eeglab
from . import pick

from .array import RawArray
from .brainvision import read_raw_brainvision
from .bti import read_raw_bti
from .cnt import read_raw_cnt
from .ctf import read_raw_ctf
from .curry import read_raw_curry
from .edf import read_raw_edf, read_raw_bdf, read_raw_gdf
from .egi import read_raw_egi
from .kit import read_raw_kit, read_epochs_kit
from .fiff import read_raw_fif
from .nicolet import read_raw_nicolet
from .artemis123 import read_raw_artemis123
from .eeglab import read_raw_eeglab, read_epochs_eeglab
from .eximia import read_raw_eximia
from .nirx import read_raw_nirx
from .fieldtrip import (read_raw_fieldtrip, read_epochs_fieldtrip,
                        read_evoked_fieldtrip)

# for backward compatibility
from .fiff import Raw
from .fiff import Raw as RawFIF
from .base import concatenate_raws
from .reference import (set_eeg_reference, set_bipolar_reference,
                        add_reference_channels)

__all__ = [
    _coil_trans_to_loc, _empty_info, _fiff_get_fid, _force_update_info,
    _loc_to_coil_trans, _loc_to_eeg_loc, _merge_info, _stamp_to_dt,
    BaseRaw, Info, Projection, Raw, RawArray, RawFIF, add_reference_channels,
    anonymize_info, array, base, brainvision, bti, cnt, concatenate_raws,
    constants, ctf, edf, eeglab, egi, fiff, fiff_open, kit,
    make_eeg_average_ref_proj, nicolet, pick, read_epochs_eeglab,
    read_epochs_fieldtrip, read_epochs_kit, read_evoked_fieldtrip,
    read_fiducials, read_info, read_raw_artemis123, read_raw_bdf,
    read_raw_brainvision, read_raw_bti, read_raw_cnt, read_raw_ctf,
    read_raw_curry, read_raw_edf, read_raw_eeglab, read_raw_egi,
    read_raw_eximia, read_raw_fieldtrip, read_raw_fif, read_raw_gdf,
    read_raw_kit, read_raw_nicolet, set_bipolar_reference, set_eeg_reference,
    show_fiff, write_fiducials, write_info, read_raw_nirx,
]
