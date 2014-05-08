"""FIF module for IO with .fif files"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from ..utils import deprecated

from ..io.open import fiff_open, show_fiff, _fiff_get_fid
from ..io.evoked import (Evoked, read_evoked, write_evoked, read_evokeds,
                     write_evokeds)
from ..io.meas_info import read_fiducials, write_fiducials, read_info, write_info
from ..io.pick import (pick_types, pick_channels, pick_types_evoked,
                   pick_channels_regexp, pick_channels_forward,
                   pick_types_forward, pick_channels_cov,
                   pick_channels_evoked, pick_info, _has_kit_refs)

from ..io.proj import proj_equal, make_eeg_average_ref_proj
from ..io.cov import read_cov, write_cov
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

def _deprecate(obj, name):
    return deprecated('Use mne.io.%s as mne.fiff.%s is deprecated and will be '
                      'removed in v0.9.' % (name, name))(obj)

Evoked = _deprecate(Evoked, 'Evoked')
Raw = _deprecate(Raw, 'Raw')
read_evoked = _deprecate(read_evoked, 'read_evoked')
read_evokeds = _deprecate(read_evokeds, 'read_evokeds')
write_evoked = _deprecate(write_evoked, 'write_evoked')
write_evokeds = _deprecate(write_evokeds, 'write_evokeds')
read_fiducials = _deprecate(read_fiducials, 'read_fiducials')
write_fiducials = _deprecate(write_fiducials, 'write_fiducials')
read_info = _deprecate(read_info, 'read_info')
write_info = _deprecate(write_info, 'write_info')
pick_types = _deprecate(pick_types, 'pick_types')
pick_channels = _deprecate(pick_channels, 'pick_channels')
pick_types_evoked = _deprecate(pick_types_evoked, 'pick_types_evoked')
pick_channels_regexp = _deprecate(pick_channels_regexp, 'pick_channels_regexp')
pick_channels_forward = _deprecate(pick_channels_forward, 'pick_channels_forward')
pick_types_forward = _deprecate(pick_types_forward, 'pick_types_forward')
pick_channels_cov = _deprecate(pick_channels_cov, 'pick_channels_cov')
pick_channels_evoked = _deprecate(pick_channels_evoked, 'pick_channels_evoked')
pick_info = _deprecate(pick_info, 'pick_info')
proj_equal = _deprecate(proj_equal, 'proj_equal')
make_eeg_average_ref_proj = _deprecate(make_eeg_average_ref_proj, 'make_eeg_average_ref_proj')
read_cov = _deprecate(read_cov, 'read_cov')
write_cov = _deprecate(write_cov, 'write_cov')
concatenate_raws = _deprecate(concatenate_raws, 'concatenate_raws')
get_chpi_positions = _deprecate(get_chpi_positions, 'get_chpi_positions')
set_eeg_reference = _deprecate(set_eeg_reference, 'set_eeg_reference')
