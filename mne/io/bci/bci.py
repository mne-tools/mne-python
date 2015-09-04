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

    Notes
    -----
    .. versionadded:: 0.10.0

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
        """
        Dealing with the missing data
        -----------------------------
        When recording with OpenBCI over Bluetooth, it is possible for some of
        the data packets, samples, to not be recorded. This does not happen
        often but it poses a problem for maintaining proper sampling periods.
        OpenBCI data format combats this by providing a counter on the sample to
        know which ones are missing.

        Solution
        --------
        Interpolate the missing samples by resampling the surrounding samples.
        1. Find where the missing samples are.
        2. Deal with the counter reset (resets after cycling a byte).
        3. Resample given the diffs.
        4. Insert resampled data in the array using the diff indices
           (index + 1).
        """
        diff = np.abs(np.diff(data[:, 0]))
        # counter goes from 0 to 255, maxdiff is 255, make diff one like others.
        diff = np.mod(diff, 254)
        missing_idx = np.where(diff != 1)[0]
        missing_samps = diff[missing_idx]
        missing_nsamps = np.sum(missing_samps)
        missing_cumsum = np.insert(np.cumsum(missing_samps), 0, 0)[:-1]
        missing_data = np.empty((missing_nsamps, data.shape[-1]), dtype=float)
        for idx, nn, ii in zip(missing_idx, missing_samps, missing_cumsum):
            missing_data[ii:ii+nn] = np.mean(data[(idx-1, idx), :])
        idx = list()
        for ii in missing_idx:
            idx.append([ii] * ii)
        idx = np.hstack(idx)
        asdf
        data = np.insert(data, idx, missing_data)
        first_samps = [0]
        last_samps = [data.shape[0]]
        # since there is a counter column, don't need to add 1 to range
        ch_names = ['EEG %03d' % num for num in range(1, data.shape[-1])]
        ch_types = ['eeg'] * len(ch_names)
        if misc:
            misc_names = ['MISC %03d' % ii for ii in range(1, len(misc) + 1)]
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
        super(RawBCI, self).__init__(info, last_samps=last_samps,
                                     filenames=[input_fname], verbose=verbose)
        self._data = data


def read_raw_bci(input_fname, montage=None, eog=None, misc=[-3, -2, -1],
                 reference=None, scale=1e-6, sfreq=250, preload=True, verbose=None):
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

    Notes
    -----
    .. versionadded:: 0.10.0

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    raw = RawBCI(input_fname=input_fname, montage=montage, eog=eog,
                 misc=misc, reference=reference, scale=scale, sfreq=sfreq,
                 preload=preload, verbose=verbose)
    return raw
