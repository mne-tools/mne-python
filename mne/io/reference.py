# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


from .._fiff import _io_dep_getattr


def __getattr__(name):
    return _io_dep_getattr(name, "reference")
