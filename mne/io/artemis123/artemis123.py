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

    #key names for artemis channel info...
    chanKeys = ['name','scaling','FLL_Gain','FLL_Mode','FLL_HighPass','FLL_AutoReset','FLL_ResetLock']

    header_info = dict()
    header_info['filterHist'] = [];
    header_info['comments']   = ''
    header_info['chanels']    = [];

    with open(header, 'r') as fid:
      #section flag
      # 0 - None
      # 1 - main header
      # 2 - channel header
      # 3 - comments 
      # 4 - length
      # 5 - filtering History
      sectionFlag = 0
      for line in fid:
        #skip emptylines
        if not line.strip() or ( sectionFlag ==2 and line.startswith('DAQ Map') ):
          continue

        #set sectionFlag
        if line.startswith('<end'):
          sectionFlag = 0
        elif line.startswith("<start main header>"):
          sectionFlag = 1
        elif line.startswith("<start per channel header>"):
          sectionFlag = 2
        elif line.startswith("<start comments>"):
          sectionFlag = 3
        elif line.startswith("<start length>"):
          sectionFlag = 4
        elif line.startswith("<start filtering history>"):
          sectionFlag = 5
        else:
          #parse header info lines
          if sectionFlag == 1: #part of main header - lines are name value pairs
            values = line.strip().split('\t')
            if len(values) == 1:
              values.append('')
            header_info[values[0]] = values[1]
          elif sectionFlag == 2: #part of channel header - lines are Channel Info
            values = line.strip().split('\t')
            if len(values) != 7:
              pass #error
            tmp = dict()
            for k,v in zip(chanKeys,values):
              tmp[k] = v
            header_info['chanels'].append(tmp)
          elif sectionFlag == 3:
            header_info['comments'] = '%s%s'%(header_info['comments'],line.strip())
          elif sectionFlag == 4:
            header_info['num_samples'] = int(line.strip())
          elif sectionFlag == 5:
            header_info['filterHist'].append(line.strip())

    #########################################################################################
    #build mne info struct
    info = _empty_info(float(header_info['Rate Out']))

    #Attempt to get time/date from fname
    try:
        date = datetime.datetime.strptime(path.basename(fname).split('_')[2],'%Y-%m-%d-%Hh-%Mm')
    except:
        date = None

    #TODO expand on the descriptiong as a compbination of header fields and comments
    desc = None
    info = _empty_info(float(header_info['Rate Out']))
    info.update({'filename': fname,
                'meas_date': calendar.timegm(date.utctimetuple()),
                'description': None, 'buffer_size_sec': 1.})
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
        last_samps = [header_info['num_samples'] - 1]
        super(RawArtemis123, self).__init__(
            info, preload, filenames=[input_fname], raw_extras=[header_info],
            last_samps=last_samps, orig_format='int',
            verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        _read_segments_file(self, data, idx, fi, start, stop, cals, mult)
