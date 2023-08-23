"""Private module for FIF basic I/O routines."""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

# All imports should be done directly to submodules, so we don't import
# anything here or use lazy_loader.


# Helpers for keeping some attributes in mne/io/*.py, remove in 1.6
_dep_msg = (
    "is deprecated will be removed in 1.6, use documented public API instead. "
    "If no appropriate public API exists, please open an issue on GitHub."
)


def _io_dep_getattr(name, mod):
    import importlib
    from ..utils import warn

    fiff_mod = importlib.import_module(f"mne._fiff.{mod}")
    obj = getattr(fiff_mod, name)
    warn(f"mne.io.{mod}.{name} {_dep_msg}", FutureWarning)
    return obj
