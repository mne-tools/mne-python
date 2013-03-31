"""FIF module for IO with .fif files"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)
from ..utils import deprecated

_io = ('The fiff module will no longer be supported in v0.7.'
       ' Please use the io module instead.')

from ..io.fiff.constants import FIFF
from ..io.fiff.open import fiff_open, show_fiff
from ..io.fiff.evoked import Evoked, read_evoked, write_evoked
from ..io.fiff.raw import Raw, start_writing_raw, write_raw_buffer, \
                 finish_writing_raw, concatenate_raws
from ..io.fiff.pick import pick_types, pick_channels, pick_types_evoked, \
                pick_channels_regexp, pick_channels_forward, \
                pick_types_forward, pick_channels_cov, \
                pick_channels_evoked


for k, v  in globals().items():
    if k == 'deprecated':
        continue
    if 'fiff' in getattr(v,  '__module__', '') and hasattr(v, '__name__'):
        globals()[k] = deprecated(_io)(v)
