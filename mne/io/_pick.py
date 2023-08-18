# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause


from .._fiff import _io_dep_getattr
from .._fiff.pick import (
    _picks_to_idx,
    _DATA_CH_TYPES_ORDER_DEFAULT,
    _DATA_CH_TYPES_SPLIT,
    channel_indices_by_type,
)

__all__ = [
    "_picks_to_idx",
    # mne-qt-browser
    "_DATA_CH_TYPES_ORDER_DEFAULT",
    "_DATA_CH_TYPES_SPLIT",
    "channel_indices_by_type",
]


def __getattr__(name):
    try:
        return globals()[name]
    except KeyError:
        pass
    return _io_dep_getattr(name, "pick")
