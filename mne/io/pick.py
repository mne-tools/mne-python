# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause


from .._fiff import _io_dep_getattr
from .._fiff.pick import (
    _DATA_CH_TYPES_ORDER_DEFAULT,
    _DATA_CH_TYPES_SPLIT,
    _picks_to_idx,
)

__all__ = [
    # mne-bids, autoreject, mne-connectivity, mne-realtime, mne-nirs, mne-realtime
    "_picks_to_idx",
    # mne-qt-browser
    "_DATA_CH_TYPES_ORDER_DEFAULT",
    "_DATA_CH_TYPES_SPLIT",
]


def __getattr__(name):
    try:
        return globals()[name]
    except KeyError:
        pass
    return _io_dep_getattr(name, "pick")
