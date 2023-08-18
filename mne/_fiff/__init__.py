"""Private module for FIF basic I/O routines."""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

# All imports should be done directly to submodules, so we don't import
# anything here


_dep_msg = (
    "is deprecated will be removed in 1.6, use documented public API instead. "
    "If no appropriate public API exists, please open an issue on GitHub."
)


# Helper for keeping some attributes en mne/io/*.py
def _io_dep_getattr(name, mod, public_names=()):
    import importlib
    from ..utils import warn

    fiff_mod = importlib.import_module(f"mne._fiff.{mod}")
    obj = getattr(fiff_mod, name)
    if name not in public_names:
        warn(f"mne.io.{mod}.{name} {_dep_msg}", FutureWarning)
    return obj
