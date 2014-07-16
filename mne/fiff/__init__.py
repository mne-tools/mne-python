"""FIF module for IO with .fif files"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from ..utils import deprecated

from ..io.open import fiff_open, show_fiff, _fiff_get_fid
from ..evoked import (Evoked, read_evoked, write_evoked, read_evokeds,
                      write_evokeds)
from ..io.meas_info import read_fiducials, write_fiducials, read_info, write_info
from ..io.pick import (pick_types, pick_channels, pick_types_evoked,
                       pick_channels_regexp, pick_channels_forward,
                       pick_types_forward, pick_channels_cov,
                       pick_channels_evoked, pick_info, _has_kit_refs)

from ..io.proj import proj_equal, make_eeg_average_ref_proj
from ..cov import _read_cov, _write_cov
from ..io import array
from ..io import base
from ..io import brainvision
from ..io import bti
from ..io import edf
from ..io import egi
from ..io import fiff
from ..io import kit

# for backward compatibility
from ..io.fiff import RawFIFF
from ..io.fiff import RawFIFF as Raw
from ..io.base import concatenate_raws, get_chpi_positions, set_eeg_reference

def _deprecate_io(obj, name):
    return deprecated('Use mne.io.%s as mne.fiff.%s is deprecated and will be '
                      'removed in v0.9.' % (name, name))(obj)

def _deprecate_mne(obj, name):
    return deprecated('Use mne.%s as mne.fiff.%s is deprecated and will be '
                      'removed in v0.9.' % (name, name))(obj)


# our decorator overwrites the class, so we need to wrap :(
class Evoked(Evoked):
    pass


class Raw(Raw):
    pass


Evoked = _deprecate_io(Evoked, 'Evoked')
Raw = _deprecate_io(Raw, 'Raw')
read_evoked = _deprecate_io(read_evoked, 'read_evoked')
read_evokeds = _deprecate_io(read_evokeds, 'read_evokeds')
write_evoked = _deprecate_io(write_evoked, 'write_evoked')
write_evokeds = _deprecate_io(write_evokeds, 'write_evokeds')
read_fiducials = _deprecate_io(read_fiducials, 'read_fiducials')
write_fiducials = _deprecate_io(write_fiducials, 'write_fiducials')
read_info = _deprecate_io(read_info, 'read_info')
write_info = _deprecate_io(write_info, 'write_info')
proj_equal = _deprecate_io(proj_equal, 'proj_equal')
make_eeg_average_ref_proj = _deprecate_io(make_eeg_average_ref_proj, 'make_eeg_average_ref_proj')
read_cov = _deprecate_io(_read_cov, 'read_cov')
write_cov = _deprecate_io(_write_cov, 'write_cov')
concatenate_raws = _deprecate_io(concatenate_raws, 'concatenate_raws')
get_chpi_positions = _deprecate_io(get_chpi_positions, 'get_chpi_positions')
set_eeg_reference = _deprecate_io(set_eeg_reference, 'set_eeg_reference')

pick_types = _deprecate_mne(pick_types, 'pick_types')
pick_channels = _deprecate_mne(pick_channels, 'pick_channels')
pick_types_evoked = _deprecate_mne(pick_types_evoked, 'pick_types_evoked')
pick_channels_regexp = _deprecate_mne(pick_channels_regexp, 'pick_channels_regexp')
pick_channels_forward = _deprecate_mne(pick_channels_forward, 'pick_channels_forward')
pick_types_forward = _deprecate_mne(pick_types_forward, 'pick_types_forward')
pick_channels_cov = _deprecate_mne(pick_channels_cov, 'pick_channels_cov')
pick_channels_evoked = _deprecate_mne(pick_channels_evoked, 'pick_channels_evoked')
pick_info = _deprecate_mne(pick_info, 'pick_info')
