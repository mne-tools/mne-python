# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import numpy as np
from os import path
import datetime
import calendar

from ...utils import logger, fill_doc
from ..utils import _read_segments_file, _find_channels, _create_chs
from ..base import BaseRaw
from ..meas_info import _empty_info
from ..constants import FIFF


@fill_doc
def read_raw_nicolet(input_fname, ch_type, eog=(),
                     ecg=(), emg=(), misc=(), preload=False, verbose=None):
    """Read Nicolet data as raw object.

    Note: This reader takes data files with the extension ``.data`` as an
    input. The header file with the same file name stem and an extension
    ``.head`` is expected to be found in the same directory.

    Parameters
    ----------
    input_fname : str
        Path to the data file.
    ch_type : str
        Channel type to designate to the data channels. Supported data types
        include 'eeg', 'seeg'.
    eog : list | tuple | 'auto'
        Names of channels or list of indices that should be designated
        EOG channels. If 'auto', the channel names beginning with
        ``EOG`` are used. Defaults to empty tuple.
    ecg : list or tuple | 'auto'
        Names of channels or list of indices that should be designated
        ECG channels. If 'auto', the channel names beginning with
        ``ECG`` are used. Defaults to empty tuple.
    emg : list or tuple | 'auto'
        Names of channels or list of indices that should be designated
        EMG channels. If 'auto', the channel names beginning with
        ``EMG`` are used. Defaults to empty tuple.
    misc : list or tuple
        Names of channels or list of indices that should be designated
        MISC channels. Defaults to empty tuple.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        A Raw object containing the data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawNicolet(input_fname, ch_type, eog=eog, ecg=ecg,
                      emg=emg, misc=misc, preload=preload, verbose=verbose)


def _get_nicolet_info(fname, ch_type, eog, ecg, emg, misc):
    """Extract info from Nicolet header files."""
    fname = path.splitext(fname)[0]
    header = fname + '.head'

    logger.info('Reading header...')
    header_info = dict()
    with open(header, 'r') as fid:
        for line in fid:
            var, value = line.split('=')
            if var == 'elec_names':
                value = value[1:-2].split(',')  # strip brackets
            elif var == 'conversion_factor':
                value = float(value)
            elif var != 'start_ts':
                value = int(value)
            header_info[var] = value

    ch_names = header_info['elec_names']
    if eog == 'auto':
        eog = _find_channels(ch_names, 'EOG')
    if ecg == 'auto':
        ecg = _find_channels(ch_names, 'ECG')
    if emg == 'auto':
        emg = _find_channels(ch_names, 'EMG')

    date, time = header_info['start_ts'].split()
    date = date.split('-')
    time = time.split(':')
    sec, msec = time[2].split('.')
    date = datetime.datetime(int(date[0]), int(date[1]), int(date[2]),
                             int(time[0]), int(time[1]), int(sec), int(msec))
    info = _empty_info(header_info['sample_freq'])
    info['meas_date'] = (calendar.timegm(date.utctimetuple()), 0)

    if ch_type == 'eeg':
        ch_coil = FIFF.FIFFV_COIL_EEG
        ch_kind = FIFF.FIFFV_EEG_CH
    elif ch_type == 'seeg':
        ch_coil = FIFF.FIFFV_COIL_EEG
        ch_kind = FIFF.FIFFV_SEEG_CH
    else:
        raise TypeError("Channel type not recognized. Available types are "
                        "'eeg' and 'seeg'.")
    cals = np.repeat(header_info['conversion_factor'] * 1e-6, len(ch_names))
    info['chs'] = _create_chs(ch_names, cals, ch_coil, ch_kind, eog, ecg, emg,
                              misc)
    info['highpass'] = 0.
    info['lowpass'] = info['sfreq'] / 2.0
    info._update_redundant()
    return info, header_info


class RawNicolet(BaseRaw):
    """Raw object from Nicolet file.

    Parameters
    ----------
    input_fname : str
        Path to the Nicolet file.
    ch_type : str
        Channel type to designate to the data channels. Supported data types
        include 'eeg', 'seeg'.
    eog : list | tuple | 'auto'
        Names of channels or list of indices that should be designated
        EOG channels. If 'auto', the channel names beginning with
        ``EOG`` are used. Defaults to empty tuple.
    ecg : list or tuple | 'auto'
        Names of channels or list of indices that should be designated
        ECG channels. If 'auto', the channel names beginning with
        ``ECG`` are used. Defaults to empty tuple.
    emg : list or tuple | 'auto'
        Names of channels or list of indices that should be designated
        EMG channels. If 'auto', the channel names beginning with
        ``EMG`` are used. Defaults to empty tuple.
    misc : list or tuple
        Names of channels or list of indices that should be designated
        MISC channels. Defaults to empty tuple.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    def __init__(self, input_fname, ch_type, eog=(),
                 ecg=(), emg=(), misc=(), preload=False,
                 verbose=None):  # noqa: D102
        input_fname = path.abspath(input_fname)
        info, header_info = _get_nicolet_info(input_fname, ch_type, eog, ecg,
                                              emg, misc)
        last_samps = [header_info['num_samples'] - 1]
        super(RawNicolet, self).__init__(
            info, preload, filenames=[input_fname], raw_extras=[header_info],
            last_samps=last_samps, orig_format='int',
            verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        _read_segments_file(self, data, idx, fi, start, stop, cals, mult)
