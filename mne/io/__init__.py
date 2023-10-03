"""IO module for reading raw data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#
# License: BSD-3-Clause

import lazy_loader as lazy

__getattr_lz__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

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
        return globals()[f"_{name}"]
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
