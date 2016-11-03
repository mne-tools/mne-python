# Author: Luke Bloy <bloyl@chop.edu>
#
# License: BSD (3-clause)

import numpy as np
from os import path
import datetime
import calendar

from ...utils import logger
from ..utils import _read_segments_file, _find_channels, _create_chs
from ..base import _BaseRaw, _check_update_montage
from ..meas_info import _empty_info
from ..constants import FIFF


def read_raw_artemis123(input_fname, preload=False, verbose=None):
    """Read Artemis123 data as raw object.

    Note: This reader takes data files with the extension ``.bin`` as an
    input. The header file with the same file name stem and an extension
    ``.`` is expected to be found in the same directory.

    Parameters
    ----------
    input_fname : str
        Path to the data file.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : Instance of Raw
        A Raw object containing the data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawArtemis123(input_fname, preload=preload, verbose=verbose)


def _get_artemis123_info(fname):
    """Function for extracting info from artemis123 header files."""
    fname = path.splitext(fname)[0]
    header = fname + '.txt'

    logger.info('Reading header...')
    header_info = dict()
    with open(header, 'r') as fid:
        pass
    #     for line in fid:
    #         var, value = line.split('=')
    #         if var == 'elec_names':
    #             value = value[1:-2].split(',')  # strip brackets
    #         elif var == 'conversion_factor':
    #             value = float(value)
    #         elif var != 'start_ts':
    #             value = int(value)
    #         header_info[var] = value
    #
    # ch_names = header_info['elec_names']
    # if eog == 'auto':
    #     eog = _find_channels(ch_names, 'EOG')
    # if ecg == 'auto':
    #     ecg = _find_channels(ch_names, 'ECG')
    # if emg == 'auto':
    #     emg = _find_channels(ch_names, 'EMG')
    # date, time = header_info['start_ts'].split()
    # date = date.split('-')
    # time = time.split(':')
    # sec, msec = time[2].split('.')
    # date = datetime.datetime(int(date[0]), int(date[1]), int(date[2]),
    #                          int(time[0]), int(time[1]), int(sec), int(msec))
    info = _empty_info(5000)
    # info.update({'filename': fname,
    #              'meas_date': calendar.timegm(date.utctimetuple()),
    #              'description': None, 'buffer_size_sec': 1.})
    #
    # if ch_type == 'eeg':
    #     ch_coil = FIFF.FIFFV_COIL_EEG
    #     ch_kind = FIFF.FIFFV_EEG_CH
    # elif ch_type == 'seeg':
    #     ch_coil = FIFF.FIFFV_COIL_EEG
    #     ch_kind = FIFF.FIFFV_SEEG_CH
    # else:
    #     raise TypeError("Channel type not recognized. Available types are "
    #                     "'eeg' and 'seeg'.")
    # cals = np.repeat(header_info['conversion_factor'] * 1e-6, len(ch_names))
    # info['chs'] = _create_chs(ch_names, cals, ch_coil, ch_kind, eog, ecg, emg,
    #                           misc)
    # info['highpass'] = 0.
    # info['lowpass'] = info['sfreq'] / 2.0
    # info._update_redundant()
    return info, header_info


class RawArtemis123(_BaseRaw):
    """Raw object from Artemis123 file.

    Parameters
    ----------
    input_fname : str
        Path to the Artemis123 data file (ending in ``'.bin'``).
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    def __init__(self, input_fname, preload=False, verbose=None):
        input_fname = path.abspath(input_fname)
        info, header_info = _get_artemis123_info(input_fname)
        # last_samps = [header_info['num_samples'] - 1]
        last_samps = None
        super(RawArtemis123, self).__init__(
            info, preload, filenames=[input_fname], raw_extras=[header_info],
            last_samps=last_samps, orig_format='int',
            verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        _read_segments_file(self, data, idx, fi, start, stop, cals, mult)
