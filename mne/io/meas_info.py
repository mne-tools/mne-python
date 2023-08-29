# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause


from .._fiff import _io_dep_getattr


def __getattr__(name):
    return _io_dep_getattr(name, "meas_info")
