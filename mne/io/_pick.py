# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause


from .._fiff import _io_dep_getattr
from .._fiff.pick import _picks_to_idx

__all__ = ["_picks_to_idx"]


def __getattr__(name):
    try:
        return globals()[name]
    except KeyError:
        pass
    return _io_dep_getattr(name, "pick")
