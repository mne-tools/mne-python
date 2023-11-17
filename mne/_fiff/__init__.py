"""Private module for FIF basic I/O routines."""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# All imports should be done directly to submodules, so we don't import
# anything here or use lazy_loader.

# This warn import (made private as _warn) is just for the temporary
# _io_dep_getattr and can be removed in 1.6 along with _dep_msg and _io_dep_getattr.
from ..utils import warn as _warn


_dep_msg = (
    "is deprecated will be removed in 1.6, use documented public API instead. "
    "If no appropriate public API exists, please open an issue on GitHub."
)


def _io_dep_getattr(name, mod):
    import importlib

    fiff_mod = importlib.import_module(f"mne._fiff.{mod}")
    obj = getattr(fiff_mod, name)
    _warn(f"mne.io.{mod}.{name} {_dep_msg}", FutureWarning)
    return obj
