from ..channels import (make_eeg_layout, make_grid_layout, read_layout,
                        find_layout)
from ..channels import Layout as _Layout

from ..utils import deprecated as dep

msg = ('The module ``mne.layouts`` is deprecated and will be removed in '
       'MNE-Python 0.10. Please import ``{0}`` from ``mne.channels``')


dep(msg.format('Layout'))
class Layout(_Layout):
    """"""  # needed to inherit doc string
    __doc__ += _Layout.__doc__
    pass


make_eeg_layout = dep(msg.format('make_eeg_layout'))(make_eeg_layout)
make_grid_layout = dep(msg.format('make_grid_layout'))(make_grid_layout)
read_layout = dep(msg.format('read_layout'))(read_layout)
find_layout = dep(msg.format('find_layout'))(find_layout)
