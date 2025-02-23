"""Private module for FIF basic I/O routines."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# All imports should be done directly to submodules, so we don't import
# anything here


_dep_msg = (
    "is deprecated will be removed in 1.6, use documented public API instead. "
    "If no appropriate public API exists, please open an issue on GitHub."
)


# Helper for keeping some attributes en mne/io/*.py
def _io_dep_getattr(name, mod):
    import importlib
    from ..utils import warn

    fiff_mod = importlib.import_module(f"mne._fiff.{mod}")
    obj = getattr(fiff_mod, name)
    warn(f"mne.io.{mod}.{name} {_dep_msg}", FutureWarning)
    return obj
