# Author: Luke Bloy <bloyl@chop.edu>
#
# License: BSD (3-clause)

import numpy as np
import os.path as op
import datetime
import calendar

from .utils import _load_mne_locs, _read_pos
from ...utils import logger, warn, verbose
from ..utils import _read_segments_file
from ..base import BaseRaw
from ..meas_info import _empty_info
from .._digitization import _make_dig_points
from ..constants import FIFF
from ...chpi import _fit_device_hpi_positions, _fit_coil_order_dev_head_trans
from ...transforms import get_ras_to_neuromag_trans, apply_trans, Transform


@verbose
def read_raw_artemis123(input_fname, preload=False, verbose=None,
                        pos_fname=None, add_head_trans=True):
    """Read Artemis123 data as raw object.

    Parameters
    ----------
    input_fname : str
        Path to the data file (extension ``.bin``). The header file with the
        same file name stem and an extension ``.txt`` is expected to be found
        in the same directory.
    %(preload)s
    %(verbose)s
    pos_fname : str or None (default None)
        If not None, load digitized head points from this file
    add_head_trans : bool (default True)
        If True attempt to perform initial head localization. Compute initial
        device to head coordinate transform using HPI coils. If no
        HPI coils are in info['dig'] hpi coils are assumed to be in canonical
        order of fiducial points (nas, rpa, lpa).

    Returns
    -------
    raw : instance of Raw
        A Raw object containing the data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawArtemis123(input_fname, preload=preload, verbose=verbose,
                         pos_fname=pos_fname, add_head_trans=add_head_trans)


def _get_artemis123_info(fname, pos_fname=None):
    """Generate info struct from artemis123 header file."""
    fname = op.splitext(fname)[0]
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
        warn('Non-Empty Filter history found, BUT is not supported' % k)

    # build mne info struct
    info = _empty_info(float(header_info['DAQ Sample Rate']))

    # Attempt to get time/date from fname
    # Artemis123 files saved from the scanner observe the following
    # naming convention 'Artemis_Data_YYYY-MM-DD-HHh-MMm_[chosen by user].bin'
    try:
        date = datetime.datetime.strptime(
            op.basename(fname).split('_')[2], '%Y-%m-%d-%Hh-%Mm')
        meas_date = (calendar.timegm(date.utctimetuple()), 0)
    except Exception:
        meas_date = None

    # build subject info must be an integer (as per FIFF)
    try:
        subject_info = {'id': int(header_info['Subject ID'])}
    except ValueError:
        subject_info = {'id': 0}

    # build description
    desc = ''
    for k in ['Purpose', 'Notes']:
        desc += '{} : {}\n'.format(k, header_info[k])
    desc += 'Comments : {}'.format(header_info['comments'])

    info.update({'meas_date': meas_date,
                 'description': desc,
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

    # HPI information
    # print header_info.keys()
    hpi_sub = dict()
    # Don't know what event_channel is don't think we have it HPIs are either
    # always on or always off.
    # hpi_sub['event_channel'] = ???
    hpi_sub['hpi_coils'] = [dict(), dict(), dict(), dict()]
    hpi_coils = [dict(), dict(), dict(), dict()]
    drive_channels = ['MIO_001', 'MIO_003', 'MIO_009', 'MIO_011']
    key_base = 'Head Tracking %s %d'

    # set default HPI frequencies
    if info['sfreq'] == 1000:
        default_freqs = [140, 150, 160, 40]
    else:
        default_freqs = [700, 750, 800, 40]

    for i in range(4):
        # build coil structure
        hpi_coils[i]['number'] = i + 1
        hpi_coils[i]['drive_chan'] = drive_channels[i]
        this_freq = header_info.pop(key_base % ('Frequency', i + 1),
                                    default_freqs[i])
        hpi_coils[i]['coil_freq'] = this_freq

        # check if coil is on
        if header_info[key_base % ('Channel', i + 1)] == 'OFF':
            hpi_sub['hpi_coils'][i]['event_bits'] = [0]
        else:
            hpi_sub['hpi_coils'][i]['event_bits'] = [256]

    info['hpi_subsystem'] = hpi_sub
    info['hpi_meas'] = [{'hpi_coils': hpi_coils}]

    # read in digitized points if supplied
    if pos_fname is not None:
        info['dig'] = _read_pos(pos_fname)
    else:
        info['dig'] = []

    info._update_redundant()
    return info, header_info


class RawArtemis123(BaseRaw):
    """Raw object from Artemis123 file.

    Parameters
    ----------
    input_fname : str
        Path to the Artemis123 data file (ending in ``'.bin'``).
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, input_fname, preload=False, verbose=None,
                 pos_fname=None, add_head_trans=True):  # noqa: D102
        from scipy.spatial.distance import cdist

        fname, ext = op.splitext(input_fname)
        if ext == '.txt':
            input_fname = fname + '.bin'
        elif ext != '.bin':
            raise RuntimeError('Valid artemis123 files must end in "txt"' +
                               ' or ".bin".')

        if not op.exists(input_fname):
            raise RuntimeError('%s - Not Found' % input_fname)

        info, header_info = _get_artemis123_info(input_fname,
                                                 pos_fname=pos_fname)

        last_samps = [header_info.get('num_samples', 1) - 1]

        super(RawArtemis123, self).__init__(
            info, preload, filenames=[input_fname], raw_extras=[header_info],
            last_samps=last_samps, orig_format=np.float32,
            verbose=verbose)
        self.info['hpi_results'] = []

        if add_head_trans:
            n_hpis = 0
            for d in info['hpi_subsystem']['hpi_coils']:
                if d['event_bits'] == [256]:
                    n_hpis += 1
            if n_hpis < 3:
                warn('%d HPIs active. At least 3 needed to perform' % n_hpis +
                     'head localization\n *NO* head localization performed')
            else:
                # Localized HPIs using the 1st seconds of data.
                hpi_dev, hpi_g = _fit_device_hpi_positions(self,
                                                           t_win=[0, 0.25])
                if pos_fname is not None:
                    logger.info('No Digitized cHPI locations found.\n' +
                                'Assuming cHPIs are placed at cardinal ' +
                                'fiducial locations. (Nasion, LPA, RPA')

                    # Digitized HPI points are needed.
                    hpi_head = np.array([d['r']
                                         for d in self.info.get('dig', [])
                                         if d['kind'] == FIFF.FIFFV_POINT_HPI])

                    if (len(hpi_head) != len(hpi_dev)):
                        mesg = ("number of digitized (%d) and " +
                                "active (%d) HPI coils are " +
                                "not the same.")
                        raise RuntimeError(mesg % (len(hpi_head),
                                                   len(hpi_dev)))

                    # compute initial head to dev transform and hpi ordering
                    head_to_dev_t, order = \
                        _fit_coil_order_dev_head_trans(hpi_dev, hpi_head)

                    # set the device to head transform
                    self.info['dev_head_t'] = \
                        Transform(FIFF.FIFFV_COORD_DEVICE,
                                  FIFF.FIFFV_COORD_HEAD, head_to_dev_t)

                    dig_dists = cdist(hpi_head, hpi_head)
                    dev_dists = cdist(hpi_dev, hpi_dev)
                    tmp_dists = np.abs(dig_dists - dev_dists)
                    dist_limit = tmp_dists.max() * 1.1

                else:
                    logger.info('Assuming Cardinal HPIs')
                    nas = hpi_dev[0]
                    lpa = hpi_dev[2]
                    rpa = hpi_dev[1]
                    t = get_ras_to_neuromag_trans(nas, lpa, rpa)
                    self.info['dev_head_t'] = \
                        Transform(FIFF.FIFFV_COORD_DEVICE,
                                  FIFF.FIFFV_COORD_HEAD, t)

                    # transform fiducial points
                    nas = apply_trans(t, nas)
                    lpa = apply_trans(t, lpa)
                    rpa = apply_trans(t, rpa)

                    hpi = [nas, rpa, lpa]
                    self.info['dig'] = _make_dig_points(nasion=nas, lpa=lpa,
                                                        rpa=rpa, hpi=hpi)
                    order = np.array([0, 1, 2])
                    dist_limit = 0.005

                # fill in hpi_results
                hpi_result = dict()

                # add HPI points in device coords...
                dig = []
                for idx, point in enumerate(hpi_dev):
                    dig.append({'r': point, 'ident': idx + 1,
                                'kind': FIFF.FIFFV_POINT_HPI,
                                'coord_frame': FIFF.FIFFV_COORD_DEVICE})
                hpi_result['dig_points'] = dig

                # attach Transform
                hpi_result['coord_trans'] = self.info['dev_head_t']

                # 1 based indexing
                hpi_result['order'] = order + 1
                hpi_result['used'] = np.arange(3) + 1
                hpi_result['dist_limit'] = dist_limit
                hpi_result['good_limit'] = 0.98

                # Warn for large discrepancies between digitized and fit
                # cHPI locations
                if hpi_result['dist_limit'] > 0.005:
                    warn('Large difference between digitized geometry' +
                         ' and HPI geometry. Max coil to coil difference' +
                         ' is %0.2f cm\n' % (100. * tmp_dists.max()) +
                         'beware of *POOR* head localization')

                # store it
                self.info['hpi_results'] = [hpi_result]

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        _read_segments_file(self, data, idx, fi, start,
                            stop, cals, mult, dtype='>f4')
