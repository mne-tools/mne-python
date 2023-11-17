# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from .._fiff import _io_dep_getattr
from .._fiff.constants import FIFF

__all__ = ["FIFF"]


def __getattr__(name):
    try:
        return globals()[name]
    except KeyError:
        pass
    return _io_dep_getattr(name, "constants")
