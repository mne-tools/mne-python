# Author: Luke Bloy <bloyl@chop.edu>
#
# License: BSD (3-clause)

import numpy as np
import os.path as op
import datetime
import calendar

from .utils import _load_mne_locs
from ...utils import logger, warn
from ..utils import _read_segments_file
from ..base import BaseRaw
from ..meas_info import _empty_info
from ..constants import FIFF


def read_raw_artemis123(input_fname, preload=False, verbose=None):
    """Read Artemis123 data as raw object.

    Parameters
    ----------
    input_fname : str
        Path to the data file (extension ``.bin``). The header file with the
        same file name stem and an extension ``.txt`` is expected to be found
        in the same directory.
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
    fname = op.splitext(op.abspath(fname))[0]
    header = fname + '.txt'

    logger.info('Reading header...')

    # key names for artemis channel info...
    chan_keys = ['name', 'scaling', 'FLL_Gain', 'FLL_Mode', 'FLL_HighPass',
                 'FLL_AutoReset', 'FLL_ResetLock']

    header_info = dict()
    header_info['filter_hist'] = []
    header_info['comments'] = ''
    header_info['channels'] = []

    with open(header, 'r') as fid:
        # section flag
        # 0 - None
        # 1 - main header
        # 2 - channel header
        # 3 - comments
        # 4 - length
        # 5 - filtering History
        sectionFlag = 0
        for line in fid:
            # skip emptylines or header line for channel info
            if ((not line.strip()) or
               (sectionFlag == 2 and line.startswith('DAQ Map'))):
                continue

            # set sectionFlag
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
                # parse header info lines
                # part of main header - lines are name value pairs
                if sectionFlag == 1:
                    values = line.strip().split('\t')
                    if len(values) == 1:
                        values.append('')
                    header_info[values[0]] = values[1]
                # part of channel header - lines are Channel Info
                elif sectionFlag == 2:
                    values = line.strip().split('\t')
                    if len(values) != 7:
                        raise IOError('Error parsing line \n\t:%s\n' % line +
                                      'from file %s' % header)
                    tmp = dict()
                    for k, v in zip(chan_keys, values):
                        tmp[k] = v
                    header_info['channels'].append(tmp)
                elif sectionFlag == 3:
                    header_info['comments'] = '%s%s' \
                        % (header_info['comments'], line.strip())
                elif sectionFlag == 4:
                    header_info['num_samples'] = int(line.strip())
                elif sectionFlag == 5:
                    header_info['filter_hist'].append(line.strip())

    for k in ['Temporal Filter Active?', 'Decimation Active?',
              'Spatial Filter Active?']:
        if(header_info[k] != 'FALSE'):
            warn('%s - set to but is not supported' % k)
    if(header_info['filter_hist']):
        warn('Non-Empty Filter histroy found, BUT is not supported' % k)

    # build mne info struct
    info = _empty_info(float(header_info['Rate Out']))

    # Attempt to get time/date from fname
    # Artemis123 files saved from the scanner observe the following
    # naming convention 'Artemis_Data_YYYY-MM-DD-HHh-MMm_[chosen by user].bin'
    try:
        date = datetime.datetime.strptime(
            op.basename(fname).split('_')[2], '%Y-%m-%d-%Hh-%Mm')
        meas_date = calendar.timegm(date.utctimetuple())
    except Exception:
        meas_date = None

    # build subject info
    subject_info = {'id': header_info['Subject ID']}

    # build description
    desc = ''
    for k in ['Purpose', 'Notes']:
        desc += '{} : {}\n'.format(k, header_info[k])
    desc += 'Comments : {}'.format(header_info['comments'])

    info = _empty_info(float(header_info['Rate Out']))
    info.update({'meas_date': meas_date,
                 'description': desc, 'buffer_size_sec': 1.,
                 'subject_info': subject_info,
                 'proj_name': header_info['Project Name']})

    # Channel Names by type
    ref_mag_names = ['REF_001', 'REF_002', 'REF_003',
                     'REF_004', 'REF_005', 'REF_006']

    ref_grad_names = ['REF_007', 'REF_008', 'REF_009',
                      'REF_010', 'REF_011', 'REF_012']

    # load mne loc dictionary
    loc_dict = _load_mne_locs()
    info['chs'] = []
    info['bads'] = []

    for i, chan in enumerate(header_info['channels']):
        # build chs struct
        t = {'cal': float(chan['scaling']), 'ch_name': chan['name'],
             'logno': i + 1, 'scanno': i + 1, 'range': 1.0,
             'unit_mul': FIFF.FIFF_UNITM_NONE,
             'coord_frame': FIFF.FIFFV_COORD_DEVICE}
        # REF_018 has a zero cal which can cause problems. Let's set it to
        # a value of another ref channel to make writers/readers happy.
        if t['cal'] == 0:
            t['cal'] = 4.716e-10
            info['bads'].append(t['ch_name'])
        t['loc'] = loc_dict.get(chan['name'], np.zeros(12))

        if (chan['name'].startswith('MEG')):
            t['coil_type'] = FIFF.FIFFV_COIL_ARTEMIS123_GRAD
            t['kind'] = FIFF.FIFFV_MEG_CH
            # While gradiometer units are T/m, the meg sensors referred to as
            # gradiometers report the field difference between 2 pick-up coils.
            # Therefore the units of the measurements should be T
            # *AND* the baseline (difference between pickup coils)
            # should not be used in leadfield / forwardfield computations.
            t['unit'] = FIFF.FIFF_UNIT_T
            t['unit_mul'] = FIFF.FIFF_UNITM_F

        # 3 axis referance magnetometers
        elif (chan['name'] in ref_mag_names):
            t['coil_type'] = FIFF.FIFFV_COIL_ARTEMIS123_REF_MAG
            t['kind'] = FIFF.FIFFV_REF_MEG_CH
            t['unit'] = FIFF.FIFF_UNIT_T
            t['unit_mul'] = FIFF.FIFF_UNITM_F

        # reference gradiometers
        elif (chan['name'] in ref_grad_names):
            t['coil_type'] = FIFF.FIFFV_COIL_ARTEMIS123_REF_GRAD
            t['kind'] = FIFF.FIFFV_REF_MEG_CH
            # While gradiometer units are T/m, the meg sensors referred to as
            # gradiometers report the field difference between 2 pick-up coils.
            # Therefore the units of the measurements should be T
            # *AND* the baseline (difference between pickup coils)
            # should not be used in leadfield / forwardfield computations.
            t['unit'] = FIFF.FIFF_UNIT_T
            t['unit_mul'] = FIFF.FIFF_UNITM_F

        # other reference channels are unplugged and should be ignored.
        elif (chan['name'].startswith('REF')):
            t['coil_type'] = FIFF.FIFFV_COIL_NONE
            t['kind'] = FIFF.FIFFV_MISC_CH
            t['unit'] = FIFF.FIFF_UNIT_V
            info['bads'].append(t['ch_name'])

        elif (chan['name'].startswith(('AUX', 'TRG', 'MIO'))):
            t['coil_type'] = FIFF.FIFFV_COIL_NONE
            t['unit'] = FIFF.FIFF_UNIT_V
            if (chan['name'].startswith('TRG')):
                t['kind'] = FIFF.FIFFV_STIM_CH
            else:
                t['kind'] = FIFF.FIFFV_MISC_CH
        else:
            raise ValueError('Channel does not match expected' +
                             ' channel Types:"%s"' % chan['name'])

        # incorporate mulitplier (unit_mul) into calibration
        t['cal'] *= 10 ** t['unit_mul']
        t['unit_mul'] = FIFF.FIFF_UNITM_NONE

        # append this channel to the info
        info['chs'].append(t)
        if chan['FLL_ResetLock'] == 'TRUE':
            info['bads'].append(t['ch_name'])

    # reduce info['bads'] to unique set
    info['bads'] = list(set(info['bads']))
    info._update_redundant()
    return info, header_info


class RawArtemis123(BaseRaw):
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

    def __init__(self, input_fname, preload=False, verbose=None):  # noqa: D102
        info, header_info = _get_artemis123_info(input_fname)
        last_samps = [header_info['num_samples'] - 1]
        super(RawArtemis123, self).__init__(
            info, preload, filenames=[input_fname], raw_extras=[header_info],
            last_samps=last_samps, orig_format=np.float32,
            verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        _read_segments_file(self, data, idx, fi, start,
                            stop, cals, mult, dtype='>f4')
