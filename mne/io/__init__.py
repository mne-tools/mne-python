"""FIF module for IO with .fif files"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from .open import fiff_open, show_fiff, _fiff_get_fid
from .meas_info import read_fiducials, write_fiducials, read_info, write_info

from .proj import make_eeg_average_ref_proj
from .tag import _loc_to_coil_trans, _coil_trans_to_loc, _loc_to_eeg_loc
from .base import _BaseRaw

from . import array
from . import base
from . import brainvision
from . import bti
from . import constants
from . import edf
from . import egi
from . import fiff
from . import kit
from . import pick

from .array import RawArray
from .brainvision import read_raw_brainvision
from .bti import read_raw_bti
from .edf import read_raw_edf
from .egi import read_raw_egi
from .kit import read_raw_kit, read_epochs_kit
from .fiff import read_raw_fif

# for backward compatibility
from .fiff import RawFIF
from .fiff import RawFIF as Raw
from .base import concatenate_raws
from .reference import (set_eeg_reference, set_bipolar_reference,
                        add_reference_channels)
from ..utils import deprecated


@deprecated('mne.io.get_chpi_positions is deprecated and will be removed in '
            'v0.11, please use mne.get_chpi_positions')
def get_chpi_positions(raw, t_step=None, verbose=None):
    """Extract head positions

    Note that the raw instance must have CHPI channels recorded.

    Parameters
    ----------
    raw : instance of Raw | str
        Raw instance to extract the head positions from. Can also be a
        path to a Maxfilter log file (str).
    t_step : float | None
        Sampling interval to use when converting data. If None, it will
        be automatically determined. By default, a sampling interval of
        1 second is used if processing a raw data. If processing a
        Maxfilter log file, this must be None because the log file
        itself will determine the sampling interval.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    translation : ndarray, shape (N, 3)
        Translations at each time point.
    rotation : ndarray, shape (N, 3, 3)
        Rotations at each time point.
    t : ndarray, shape (N,)
        The time points.

    Notes
    -----
    The digitized HPI head frame y is related to the frame position X as:

        Y = np.dot(rotation, X) + translation

    Note that if a Maxfilter log file is being processed, the start time
    may not use the same reference point as the rest of mne-python (i.e.,
    it could be referenced relative to raw.first_samp or something else).
    """
    from ..chpi import get_chpi_positions
    return get_chpi_positions(raw, t_step, verbose)
