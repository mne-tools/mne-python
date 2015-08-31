"""Conversion tool from OpenBCI to FIF"""

# Authors: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import os
import time
import re
import warnings

import numpy as np

from ...utils import verbose, logger
from ..constants import FIFF
from ..meas_info import create_info
from ..base import _BaseRaw, _check_update_montage
from ..reference import add_reference_channels

from ...externals.six import StringIO, u
from ...externals.six.moves import configparser


class RawBCI(_BaseRaw):
    """Raw object from BCI file

    Parameters
    ----------
    input_fname : str
        Path to the BCI file.
    montage : str | None | instance of Montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. Values should correspond to the vhdr file.
        Default is ('HEOGL', 'HEOGR', 'VEOGb').
    misc : list or tuple
        Names of channels or list of indices that should be designated
        MISC channels. Values should correspond to the electrodes
        in the vhdr file. Default is None.
    reference : None | str
        Name of the electrode which served as the reference in the recording.
        If a name is provided, a corresponding channel is added and its data
        is set to 0. This is useful for later re-referencing. The name should
        correspond to a name in elp_names. Data must be preloaded.
    scale : float
        The scaling factor for EEG data. Units for MNE are in volts.
        BCI data are typically stored in microvolts. Default scale
        factor is 1e-6.
    sfreq : int
        The sampling frequency of the data. BCI defaults are 250 Hz.
    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    @verbose
    def __init__(self, input_fname, montage=None, eog=None, misc=[-3, -2, -1],
                 reference=None, scale=1e-6, sfreq=250, preload=True,
                 verbose=None):

        if not eog:
            eog = list()
        if not misc:
            misc = list()
        data = np.genfromtxt(input_fname, delimiter=',', comments='%',
                             skip_footer=1)
        missing_data = np.diff(data[:, 0])
        ch_names = ['EEG %03d' % num for num in range(1, data.shape[-1])]
        ch_types = ['eeg'] * len(ch_names)
        if misc:
            misc_names = ['MISC %03d' % ii for ii in range(len(misc))]
            misc_types = ['misc'] * len(misc)
            for ii, mi in enumerate(misc):
                ch_names[mi] = misc_names[ii]
                ch_types[mi] = misc_types[ii]
        if eog:
            eog_names = ['EOG %03d' % ii for ii in range(len(eog))]
            misc_types = ['eog'] * len(eog)
            for ii, ei in enumerate(eog):
                ch_names[ei] = eog_names[ii]
                ch_types[ei] = eog_types[ii]

        # fix it for eog and misc marking
        info = create_info(ch_names, sfreq, ch_types, montage)
        super(RawBCI, self).__init__(info, last_samps=last_sampes,
                                     filenames=[input_fname], verbose=verbose)
        self.data = data


def read_raw_bci(input_fname, montage=None, eog=None, misc=None, reference=None,
                 scale=1e-6, sfreq=250, preload=True, verbose=None):
    """Raw object from BCI file

    Parameters
    ----------
    input_fname : str
        Path to the BCI file.
    montage : str | None | instance of Montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. Values should correspond to the vhdr file.
        Default is ('HEOGL', 'HEOGR', 'VEOGb').
    misc : list or tuple
        Names of channels or list of indices that should be designated
        MISC channels. Values should correspond to the electrodes
        in the vhdr file. Default is None.
    reference : None | str
        Name of the electrode which served as the reference in the recording.
        If a name is provided, a corresponding channel is added and its data
        is set to 0. This is useful for later re-referencing. The name should
        correspond to a name in elp_names. Data must be preloaded.
    scale : float
        The scaling factor for EEG data. Units for MNE are in volts.
        BCI data are typically stored in microvolts. Default scale
        factor is 1e-6.
    sfreq : int
        The sampling frequency of the data. BCI defaults are 250 Hz.
    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : Instance of RawBCI
        A Raw object containing OpenBCI data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    raw = RawBCI(input_fname=input_fname, montage=montage, eog=eog,
                 misc=misc, reference=reference, scale=scale, sfreq=sfreq,
                 preload=preload, verbose=verbose)
    return raw
