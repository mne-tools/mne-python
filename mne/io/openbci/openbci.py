"""Conversion tool from OpenBCI to FIF"""

# Authors: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import warnings
import numpy as np

from ...utils import verbose, logger
from ..meas_info import create_info
from ..base import _BaseRaw


class RawOpenBCI(_BaseRaw):
    """Raw object from OpenBCI file

    Parameters
    ----------
    input_fname : str
        Path to the OpenBCI file.
    montage : str | None | instance of Montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. Default is None.
    misc : list or tuple
        List of indices that should be designated MISC channels.
        Default is (-3, -2, -1), which are the accelerator sensors.
    stim_channel : int | None
        The channel index (starting at 0).
        If None (default), there will be no stim channel added.
    scale : float
        The scaling factor for EEG data. Units for MNE are in volts.
        OpenBCI data are typically stored in microvolts. Default scale
        factor is 1e-6.
    sfreq : int
        The sampling frequency of the data. OpenBCI defaults are 250 Hz.
    missing_tol : int
        The tolerance for interpolating missing samples. Default is 1. If the
        number of contiguous missing samples is greater than tolerance, then
        values are marked as NaN.
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
    def __init__(self, input_fname, montage=None, eog=None,
                 misc=(-3, -2, -1), stim_channel=None, scale=1e-6, sfreq=250,
                 missing_tol=1, preload=True, verbose=None):

        bci_info = {'missing_tol': missing_tol, 'stim_channel': stim_channel}
        if not eog:
            eog = list()
        if not misc:
            misc = list()
        nsamps, nchan = self._get_data_dims(input_fname)

        last_samps = [nsamps - 1]
        ch_names = ['EEG %03d' % num for num in range(1, nchan + 1)]
        ch_types = ['eeg'] * nchan
        if misc:
            misc_names = ['MISC %03d' % ii for ii in range(1, len(misc) + 1)]
            misc_types = ['misc'] * len(misc)
            for ii, mi in enumerate(misc):
                ch_names[mi] = misc_names[ii]
                ch_types[mi] = misc_types[ii]
        if eog:
            eog_names = ['EOG %03d' % ii for ii in range(len(eog))]
            eog_types = ['eog'] * len(eog)
            for ii, ei in enumerate(eog):
                ch_names[ei] = eog_names[ii]
                ch_types[ei] = eog_types[ii]
        if stim_channel:
            ch_names[stim_channel] = 'STI 014'
            ch_types[stim_channel] = 'stim'

        # fix it for eog and misc marking
        info = create_info(ch_names, sfreq, ch_types, montage)
        super(RawOpenBCI, self).__init__(info, last_samps=last_samps,
                                         raw_extras=[bci_info],
                                         filenames=[input_fname],
                                         preload=False, verbose=verbose)
        # load data
        if preload:
            self.preload = preload
            logger.info('Reading raw data from %s...' % input_fname)
            self._data, _ = self._read_segment()

    def _read_segment_file(self, data, idx, offset, fi, start, stop,
                           cals, mult):
        """Read a chunk of raw data"""
        input_fname = self._filenames[fi]
        data_ = np.genfromtxt(input_fname, delimiter=',', comments='%',
                              skip_footer=1)
        """
        Dealing with the missing data
        -----------------------------
        When recording with OpenBCI over Bluetooth, it is possible for some of
        the data packets, samples, to not be recorded. This does not happen
        often but it poses a problem for maintaining proper sampling periods.
        OpenBCI data format combats this by providing a counter on the sample
        to know which ones are missing.

        Solution
        --------
        Interpolate the missing samples by resampling the surrounding samples.
        1. Find where the missing samples are.
        2. Deal with the counter reset (resets after cycling a byte).
        3. Resample given the diffs.
        4. Insert resampled data in the array using the diff indices
           (index + 1).
        5. If number of missing samples is greater than the missing_tol, Values
           are replaced with np.nan.
        """
        # counter goes from 0 to 255, maxdiff is 255.
        # make diff one like others.
        missing_tol = self._raw_extras[fi]['missing_tol']
        diff = np.abs(np.diff(data_[:, 0]))
        diff = np.mod(diff, 254) - 1
        missing_idx = np.where(diff != 0)[0]
        missing_samps = diff[missing_idx].astype(int)

        if missing_samps.size:
            missing_nsamps = np.sum(missing_samps, dtype=int)
            missing_cumsum = np.insert(np.cumsum(missing_samps), 0, 0)[:-1]
            missing_data = np.empty((missing_nsamps, data_.shape[-1]),
                                    dtype=float)
            insert_idx = list()
            for idx_, nn, ii in zip(missing_idx, missing_samps,
                                    missing_cumsum):
                missing_data[ii:ii + nn] = np.mean(data_[(idx_, idx_ + 1), :])
                if nn > missing_tol:
                    missing_data[ii:ii + nn] *= np.nan
                    warnings.warn('The number of missing samples exceeded the '
                                  'missing_tol threshold.')
                insert_idx.append([idx_] * nn)
            insert_idx = np.hstack(insert_idx)
            data_ = np.insert(data_, insert_idx, missing_data, axis=0)
        # data_ dimensions are samples by channels. transpose for MNE.
        data_ = data_[start:stop, 1:].T
        data[:, offset:offset + stop - start] = \
            np.dot(mult, data_[idx]) if mult is not None else data_[idx]

    def _get_data_dims(self, input_fname):
        """Briefly scan the data file for info"""
        # raw data formatting is nsamps by nchans + counter
        data = np.genfromtxt(input_fname, delimiter=',', comments='%',
                             skip_footer=1)
        diff = np.abs(np.diff(data[:, 0]))
        diff = np.mod(diff, 254) - 1
        missing_idx = np.where(diff != 0)[0]
        missing_samps = diff[missing_idx].astype(int)
        nsamps, nchan = data.shape
        # add the missing samples
        nsamps += sum(missing_samps)
        # remove the tracker column
        nchan -= 1
        del data

        return nsamps, nchan


def read_raw_openbci(input_fname, montage=None, eog=None, misc=(-3, -2, -1),
                     stim_channel=None, scale=1e-6, sfreq=250, missing_tol=1,
                     preload=True, verbose=None):
    """Raw object from OpenBCI file

    Parameters
    ----------
    input_fname : str
        Path to the OpenBCI file.
    montage : str | None | instance of Montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. Default is None.
    misc : list or tuple
        List of indices that should be designated MISC channels.
        Default is (-3, -2, -1), which are the accelerator sensors.
    stim_channel : str | int | None
        The channel name or channel index (starting at 0).
        -1 corresponds to the last channel (default).
        If None, there will be no stim channel added.
    scale : float
        The scaling factor for EEG data. Units for MNE are in volts.
        OpenBCI data are typically stored in microvolts. Default scale
        factor is 1e-6.
    sfreq : int
        The sampling frequency of the data. OpenBCI defaults are 250 Hz.
    missing_tol : int
        The tolerance for interpolating missing samples. Default is 1. If the
        number of contiguous missing samples is greater than tolerance, then
        values are marked as NaN.
    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : Instance of RawOpenBCI
        A Raw object containing OpenBCI data.

    Notes
    -----
    .. versionadded:: 0.10.0

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    raw = RawOpenBCI(input_fname=input_fname, montage=montage, eog=eog,
                     misc=misc, stim_channel=stim_channel, scale=scale,
                     sfreq=sfreq, missing_tol=missing_tol, preload=preload,
                     verbose=verbose)
    return raw
