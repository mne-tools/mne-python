"""IO module for reading raw data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#
# License: BSD-3-Clause

import sys as _sys

from .base import BaseRaw, concatenate_raws, match_channel_orders

from . import array
from . import base
from . import brainvision
from . import bti
from . import cnt
from . import ctf
from . import edf
from . import egi
from . import fiff
from . import fil
from . import kit
from . import nicolet
from . import nirx
from . import boxy
from . import persyst
from . import eeglab
from . import nihon
from . import nsx

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
from .fiff import read_raw_fif, Raw
from .fil import read_raw_fil
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
from .fieldtrip import read_raw_fieldtrip, read_epochs_fieldtrip, read_evoked_fieldtrip
from .nihon import read_raw_nihon
from .nsx import read_raw_nsx
from ._read_raw import read_raw
from .eyelink import read_raw_eyelink

# Backward compat since these were in the public API before switching to _fiff
# (and _empty_info is convenient to keep here for tests and is private)
from .._fiff.meas_info import (
    read_info,
    read_fiducials,
    write_fiducials,
    _empty_info,
    Info as _info,
)
from .._fiff.open import show_fiff
from .._fiff.pick import get_channel_type_constants  # moved up a level

# These we will remove in 1.6
from .._fiff import (
    _dep_msg,
)

# After merge, we should move these files, as the diff is horrible if we do it
# in one PR. So for now we hack `sys.modules` to make everything happy
from . import _constants as constants
from . import _pick as pick

# These three we will remove in 1.6
from . import _proj as proj
from . import _meas_info as meas_info
from . import _reference as reference

_sys.modules.setdefault("mne.io.meas_info", meas_info)
_sys.modules.setdefault("mne.io.proj", proj)
_sys.modules.setdefault("mne.io.reference", reference)
_sys.modules.setdefault("mne.io.constants", constants)
_sys.modules.setdefault("mne.io.pick", pick)


def __getattr__(name):
    """Try getting attribute from fiff submodule."""
    from ..utils import warn

    if name in ("meas_info", "proj", "reference"):
        warn(f"mne.io.{name} {_dep_msg}", FutureWarning)
        return importlib.import_module(f"mne.io.{name}")
    elif name in (
        "set_eeg_reference",
        "set_bipolar_reference",
        "add_reference_channels",
    ):
        warn(
            f"mne.io.{name} is deprecated and will be removed in 1.6, "
            "use mne.{name} instead",
            FutureWarning,
        )
        return getattr(reference, name)
    elif name == "RawFIF":
        warn(
            "RawFIF is deprecated and will be removed in 1.6, use Raw instead",
            FutureWarning,
        )
        return Raw
    elif name == "Info":
        from .._fiff.meas_info import Info

        warn(
            "mne.io.Info is deprecated and will be removed in 1.6, "
            "use mne.Info instead",
            FutureWarning,
        )
        return Info
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module {__name__} has no attribute {name}") from None
