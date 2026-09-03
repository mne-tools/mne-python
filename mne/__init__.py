# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""MNE software for MEG and EEG data analysis."""
# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.devN' where N is an integer.
#
import lazy_loader as lazy

try:
    from importlib.metadata import version

    __version__ = version("mne")
except Exception:
    __version__ = "0.0.0"

(__getattr__, __dir__, __all__) = lazy.attach_stub(__name__, __file__)

# initialize logging
from .utils import _no_filelock, set_log_level, set_log_file

# _no_filelock: this single config read happens on every ``import mne``, and locking
# it would mean importing filelock (~20 ms of asyncio + sqlite3) up front
with _no_filelock():
    set_log_level(None, False)
    set_log_file()
