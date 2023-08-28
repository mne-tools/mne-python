"""IO module for reading raw data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#
# License: BSD-3-Clause

import lazy_loader as lazy

__getattr_lz__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "constants",
        "pick",
        # Remove these three in 1.6 along with their .py files
        "proj",
        "meas_info",
        "reference",
    ],
    submod_attrs={
        "base": ["BaseRaw", "concatenate_raws", "match_channel_orders"],
        "array": ["RawArray"],
        "besa": ["read_evoked_besa"],
        "brainvision": ["read_raw_brainvision"],
        "bti": ["read_raw_bti"],
        "cnt": ["read_raw_cnt"],
        "ctf": ["read_raw_ctf"],
        "curry": ["read_raw_curry"],
        "edf": ["read_raw_edf", "read_raw_bdf", "read_raw_gdf"],
        "egi": ["read_raw_egi", "read_evokeds_mff"],
        "kit": ["read_raw_kit", "read_epochs_kit"],
        "fiff": ["read_raw_fif", "Raw"],
        "fil": ["read_raw_fil"],
        "nedf": ["read_raw_nedf"],
        "nicolet": ["read_raw_nicolet"],
        "artemis123": ["read_raw_artemis123"],
        "eeglab": ["read_raw_eeglab", "read_epochs_eeglab"],
        "eximia": ["read_raw_eximia"],
        "hitachi": ["read_raw_hitachi"],
        "nirx": ["read_raw_nirx"],
        "boxy": ["read_raw_boxy"],
        "snirf": ["read_raw_snirf"],
        "persyst": ["read_raw_persyst"],
        "fieldtrip": [
            "read_raw_fieldtrip",
            "read_epochs_fieldtrip",
            "read_evoked_fieldtrip",
        ],
        "nihon": ["read_raw_nihon"],
        "nsx": ["read_raw_nsx"],
        "_read_raw": ["read_raw"],
        "eyelink": ["read_raw_eyelink"],
        "_fiff_wrap": [
            "read_info",
            "write_info",
            "anonymize_info",
            "read_fiducials",
            "write_fiducials",
            "show_fiff",
            "get_channel_type_constants",
        ],
    },
)


# Remove in 1.6 and change __getattr_lz__ to __getattr__
from ..utils import warn as _warn
from .._fiff.reference import (
    set_eeg_reference as _set_eeg_reference,
    set_bipolar_reference as _set_bipolar_reference,
    add_reference_channels as _add_referenc_channels,
)
from .._fiff.meas_info import Info as _Info


def __getattr__(name):
    """Try getting attribute from fiff submodule."""
    if name in (
        "set_eeg_reference",
        "set_bipolar_reference",
        "add_reference_channels",
    ):
        _warn(
            f"mne.io.{name} is deprecated and will be removed in 1.6, "
            "use mne.{name} instead",
            FutureWarning,
        )
        return locals()[f"_{name}"]
    elif name == "RawFIF":
        _warn(
            "RawFIF is deprecated and will be removed in 1.6, use Raw instead",
            FutureWarning,
        )
        name = "Raw"
    elif name == "Info":
        _warn(
            "mne.io.Info is deprecated and will be removed in 1.6, "
            "use mne.Info instead",
            FutureWarning,
        )
        return _Info
    return __getattr_lz__(name)
