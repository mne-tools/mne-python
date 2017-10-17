# Authors: Clemens Brunner <clemens.brunner@gmail.com>
#
# License: BSD (3-clause)

from os.path import splitext

from .artemis123 import read_raw_artemis123
from .brainvision import read_raw_brainvision
from .bti import read_raw_bti  # TODO: which extension?
from .cnt import read_raw_cnt
from .ctf import read_raw_ctf
from .edf import read_raw_edf
from .egi import read_raw_egi
from .kit import read_raw_kit
from .fiff import read_raw_fif
from .nicolet import read_raw_nicolet
from .eeglab import read_raw_eeglab


def read_raw(input_fname, **kwargs):
    _, ext = splitext(input_fname)

    if ext == '.bin':
        return read_raw_artemis123(input_fname, **kwargs)
    elif ext == '.vhdr':
        return read_raw_brainvision(input_fname, **kwargs)
    elif ext == '.cnt':
        return read_raw_cnt(input_fname, **kwargs)
    elif ext == '.ds':
        return read_raw_ctf(input_fname, **kwargs)
    elif ext in ['.edf', '.bdf', '.gdf']:
        return read_raw_edf(input_fname, **kwargs)
    elif ext == '.mff':
        return read_raw_egi(input_fname, **kwargs)
    elif ext == '.sqd':
        return read_raw_kit(input_fname, **kwargs)
    elif ext == '.fif':
        return read_raw_fif(input_fname, **kwargs)
    elif ext == '.data':
        return read_raw_nicolet(input_fname, **kwargs)
    elif ext == '.set':
        return read_raw_eeglab(input_fname, **kwargs)
    else:
        raise ValueError('File type {} not supported.'.format(ext))
