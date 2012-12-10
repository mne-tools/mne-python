#!/usr/bin/env python

# Authors: Denis A. Engemann  <d.engemann@fz-juelich.de>
#         Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#         Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#         Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#         Yuval Harpaz <yuvharpaz@gmail.com>
#
#         simplified bsd-3 license

from itertools import count
import time
import os.path as op
import sys
from datetime import datetime
import numpy as np

from mne import verbose
from mne.fiff.constants import Bunch, FIFF
from mne.fiff.raw import Raw, pick_types

import logging
logger = logging.getLogger('mne')

FIFF_INFO_CHS_FIELDS = ('loc', 'ch_name', 'unit_mul', 'coil_trans',
    'coord_frame', 'coil_type', 'range', 'unit', 'cal', 'eeg_loc',
    'scanno', 'kind', 'logno')

FIFF_INFO_CHS_DEFAULTS = (np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                          dtype=np.float32), None, 0, None, 0, 0, 1.0,
                          107, 1.0, None, None, 402, None)

FIFF_INFO_DIG_FIELDS = ("kind", "ident", "r", "coord_frame")
FIFF_INFO_DIG_DEFAULTS = (None, None, None, FIFF.FIFFV_COORD_HEAD)

BTI = Bunch()

BTI.HDR_FILEINFO = 'FILEINFO'
BTI.HDR_DATAFILE = 'DATAFILE'
BTI.HDR_EPOCH_INFORMATION = 'EPOCH INFORMATION'
BTI.HDR_LONGEST_EPOCH = 'LONGEST EPOCH'
BTI.HDR_FIXED_EVENTS = 'FIXED EVENTS'
BTI.HDR_CH_CAL = 'CHANNEL SENSITIVITIES'
BTI.HDR_CH_NAMES = 'CHANNEL LABELS'
BTI.HDR_CH_GROUPS = 'CHANNEL GROUPS'
BTI.HDR_CH_TRANS = 'CHANNEL XFM'

BTI.T_ROT_VV = ((0, -1, 0, 0), (1, 0, 0, 0), (0, 0, 1, 0), (1, 1, 1, 1))
BTI.T_IDENT = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (1, 1, 1, 1))
BTI.T_ROT_IX = slice(0, 3), slice(0, 3)
BTI.T_TRANS_IX = slice(0, 3), slice(3, 4)
BTI.T_SCA_IX = slice(3, 4), slice(0, 4)


def read_bti_ascii(bti_hdr_fname):
    """4-D ascii export header parser

    Use this function to read ASCII headers drawn from the export_data
    command to Python. Cave! this is tested for late 4D BTI systems and the
    MSI software on Linux and Solaris. At the moment it can't be precluded
    that other versions of the MSI software require further parsing options.
    Feedback on this is highly appreciated.

    Parameters
    ----------
    bti_hdr_fname : str
        Absolute path to the bti ascii header.

    Returns
    -------
    info : dict
        bti data info as Python dictionary

    """
    with open(bti_hdr_fname, 'r') as f:
        info = (l for l in f if not l.startswith('#'))

    _pre_parsed = {}
    this_key = None
    for line in info:
        if line.isupper() and line.endswith(':'):
            this_key = line.strip(':')
            _pre_parsed[this_key] = []
        else:
            _pre_parsed[this_key].append(line)

    info = {}
    for field, params in _pre_parsed.iteritems():
        if field in (BTI.HDR_FILEINFO, BTI.HDR_CH_NAMES, BTI.HDR_DATAFILE):
            if field == BTI.HDR_DATAFILE:
                sep = ' : '
            elif field == BTI.HDR_FILEINFO:
                sep = ':'
            else:
                sep = None

            mapping = (e.strip().split(sep) for e in params)
            mapping = [(k.strip(), v.strip()) for k, v in mapping]

            if field == BTI.HDR_CH_NAMES:
                info[field] = mapping
            else:
                info[field] = dict(mapping)

        elif field == BTI.HDR_CH_GROUPS:
            ch_groups = {}
            for p in params:
                if p.endswith('channels'):
                    ch_groups['CHANNELS'] = int(p.strip().split(' ')[0])
                elif 'MEG' in p:
                    ch_groups['MEG'] = int(p.strip().split(' ')[0])
                elif 'REFERENCE' in p:
                    ch_groups['REF'] = int(p.strip().split(' ')[0])
                elif 'EEG' in p:
                    ch_groups['EEG'] = int(p.strip().split(' ')[0])
                elif 'TRIGGER' in p:
                    ch_groups['TRIGGER'] = int(p.strip().split(' ')[0])
                elif 'UTILITY' in p:
                    ch_groups['UTILITY'] = int(p.strip().split(' ')[0])
            info[BTI.HDR_CH_GROUPS] = ch_groups

        elif field == BTI.HDR_CH_CAL:
            ch_cal = []
            ch_fields = ['ch_name', 'group', 'cal', 'unit']
            for p in params:
                this_ch_info = p.strip().split()
                ch_cal.append(dict(zip(ch_fields, this_ch_info)))
            info[BTI.HDR_CH_CAL] = ch_cal

        elif field == BTI.HDR_CH_TRANS:
            sensor_trans = {}
            idx = 0
            for p in params:
                if "|" in p:
                    k, d, _ = p.strip().split("|")
                    if k.strip().isalnum():
                        current_chan = info[BTI.HDR_CH_NAMES][idx][0]
                        sensor_trans[current_chan] = d.strip()
                        idx += 1
                    else:
                        sensor_trans[current_chan] += ', ' + d.strip()
        info[BTI.HDR_CH_TRANS] = sensor_trans

    tsl, duration = _pre_parsed['LONGEST EPOCH'][0].split(', ')
    info['FILEINFO']['Time slices'] = tsl.split(': ')[1]
    info['FILEINFO']['Total duration'] = duration.strip()

    return info


def _get_m_to_nm(adjust=None, translation=(0.0, 0.02, 0.11)):
    """ Get the general Magnes3600WH to Neuromag coordinate transform

    Parameters
    ----------
    adjust : int | None
        Degrees to tilt x-axis for sensor frame misalignment.
        If None, no adjustment will be applied.
    translation : array-like
        The translation to place the origin of coordinate system
        to the center of the head.

    Returns
    -------
    m_nm_t : ndarray
        4 x 4 rotation, translation, scaling matrix.

    """
    flip_t = np.array(BTI.T_ROT_VV, np.float64)
    adjust_t = np.array(BTI.T_IDENT, np.float64)
    adjust = 0 if adjust is None else adjust
    deg = np.deg2rad(float(adjust))
    adjust_t[[1, 2], [1, 2]] = np.cos(deg)
    adjust_t[[1, 2], [2, 1]] = -np.sin(deg), np.sin(deg)
    m_nm_t = np.ones([4, 4])
    m_nm_t[BTI.T_ROT_IX] = np.dot(flip_t[BTI.T_ROT_IX],
                                  adjust_t[BTI.T_ROT_IX])
    m_nm_t[BTI.T_TRANS_IX] = np.matrix(translation).T

    return m_nm_t


def _read_system_trans(fname):
    """ Read global 4-D system transform
    """
    trans = [l.strip().split(', ') for l in open(fname) if l.startswith(' ')]
    return np.array(trans, dtype=float).reshape([4, 4])


def _m_to_nm_coil_trans(ch_t, bti_t, nm_t, nm_default_scale=True):
    """ transforms 4D coil position to fiff / Neuromag
    """
    ch_t = np.array(ch_t.split(', '), dtype=float).reshape([4, 4])

    nm_coil_trans = _apply_t(_inverse_t(ch_t, bti_t), nm_t)

    if nm_default_scale:
        nm_coil_trans[3, :3] = 0.

    return nm_coil_trans


def _inverse_t(x, t, rot=BTI.T_ROT_IX, trans=BTI.T_TRANS_IX,
               scal=BTI.T_SCA_IX):
    """ Apply inverse transform
    """
    x = x.copy()
    x[scal] *= t[scal]
    x[rot] = np.dot(t[rot].T, x[rot])
    x[trans] -= t[trans]
    x[trans] = np.dot(t[rot].T, x[trans])

    return x


def _apply_t(x, t, rot=BTI.T_ROT_IX, trans=BTI.T_TRANS_IX, scal=BTI.T_SCA_IX):
    """ Apply transform
    """
    x = x.copy()
    x[rot] = np.dot(t[rot], x[rot])
    x[trans] = np.dot(t[rot], x[trans])
    x[trans] += t[trans]
    x[scal] *= t[scal]

    return x


def _merge_t(t1, t2):
    """ Merge two transforms """
    t = np.array(BTI.T_IDENT, dtype=np.float32)
    t[BTI.T_ROT_IX] = np.dot(t1[BTI.T_ROT_IX], t2[BTI.T_ROT_IX])
    t[BTI.T_TRANS_IX] = np.dot(t1[BTI.T_ROT_IX], t2[BTI.T_TRANS_IX])
    t[BTI.T_TRANS_IX] += t1[BTI.T_TRANS_IX]

    return t


def read_head_shape(head_shape_fname, use_hpi=False):
    """ Read digitation points from MagnesWH3600 and transform to Neuromag
        coordinates.

    Parameters
    ----------
    head_shape_fname : str
        The absolute path to the ascii-exported head shape file.
    use_hpi : bool
        Whether to treat hpi coils as digitization points or not. If False,
        Hpi coils will be discarded.

    Returns
    -------
    dig : list
        A list of dictionaries including the dig points and additional info
    t : ndarray
        The 4 x 4 matrix describing the Magnes3600WH head to Neuromag head
        transformation.
    """

    target = fiducials = []
    dig_points = []
    for line in open(head_shape_fname):
        if line.startswith('Digitization Points'):
            target = dig_points
        if line.startswith(' '):
            target += [np.array(line.strip().split(), dtype=np.float32)]

    fp = np.array(fiducials, dtype=np.float32)  # fiducial points

    dp = np.sum(fp[2] * (fp[0] - fp[1]))
    tmp1, tmp2 = np.sum(fp[2] ** 2), np.sum((fp[0] - fp[1]) ** 2)
    dcos = -dp / np.sqrt(tmp1 * tmp2)
    dsin = np.sqrt(1. - dcos * dcos)
    dt = dp / np.sqrt(tmp2)

    fiducials_nm = np.ones([len(fp), 3])

    for idx, f in enumerate(fp):
        fiducials_nm[idx, 0] = dcos * f[0] - dsin * f[1] + dt
        fiducials_nm[idx, 1] = dsin * f[0] + dcos * f[1]
        fiducials_nm[idx, 2] = f[2]

    # adjust order of fiducials to Neuromag
    fiducials_nm[[1, 2]] = fiducials_nm[[2, 1]]

    t = np.array(BTI.T_IDENT, dtype=np.float32)
    t[0, 0] = dcos
    t[0, 1] = -dsin
    t[1, 0] = dsin
    t[1, 1] = dcos
    t[0, 3] = dt

    dpnts = np.array(dig_points, dtype=np.float32).T
    dig_points_nm = np.dot(t[BTI.T_ROT_IX], dpnts).T
    dig_points_nm += t[BTI.T_TRANS_IX].T

    all_points = np.r_[fiducials_nm, dig_points_nm]
    fiducials_idents = range(1, 4) + range(1, (len(fp) + 1) - 3)
    dig = []
    for idx in xrange(all_points.shape[0]):
        point_info = dict((k, v) for k, v in zip(FIFF_INFO_DIG_FIELDS,
                          FIFF_INFO_DIG_DEFAULTS))
        point_info['r'] = all_points[idx]
        if idx < 3:
            point_info['kind'] = FIFF.FIFFV_POINT_CARDINAL
            point_info['ident'] = fiducials_idents[idx]
        if 2 < idx < len(fiducials) and use_hpi:
            point_info['kind'] = FIFF.FIFFV_POINT_HPI
            point_info['ident'] = fiducials_idents[idx]
        elif idx > 4:
            point_info['kind'] = FIFF.FIFFV_POINT_EXTRA
            point_info['ident'] = (idx + 1) - len(fiducials_idents)

        if 2 < idx < len(fiducials) and not use_hpi:
            pass
        else:
            dig.append(point_info)

    return dig, t


def _rename_channels(names):
    """Renames appropriately ordered list of channel names

    Parameters
    ----------
    names : list of str
        Lists of 4-D channel names in ascending order

    Returns
    -------
    generator object :
        Generator object for iteration over MEG channels with Neuromag names.
    """
    ref_mag, ref_grad, eog, ext = ((lambda: count(1))() for i in range(4))
    for i, name in enumerate(names, 1):
        if name.startswith('A'):
            name = 'MEG %3.3d' % i
        elif name == 'RESPONSE':
            name = 'SRI 013'
        elif name == 'TRIGGER':
            name = 'STI 014'
        elif name.startswith('EOG'):
            name = 'EOG %3.3d' % eog.next()
        elif name == 'ECG':
            name = 'ECG 001'
        elif name == 'UACurrent':
            name = 'UTL 001'
        elif name.startswith('M'):
            name = 'RFM %3.3d' % ref_mag.next()
        elif name.startswith('G'):
            name = 'RFG %3.3d' % ref_grad.next()
        elif name.startswith('X'):
            name = 'EXT %3.3d' % ext.next()

        yield name


class RawBTi(Raw):
    """ Raw object from 4-D Neuroimaging ascii exported MagnesWH3600 data

    Parameters
    ----------
    hdr_fname : str
        absolute path to asii header as drawn from msi 'export_data'
    data_fname : str
        absolute path to asii data as drawn from msi 'export_data'
    head_shape_fname : str
        absolute path to asii headshape as drawn from bti 'print_hs_file'.
    dev_head_t : ndarray | str | None
        The device to head transform. If None, an identity matrix is being
        used. If str, an BTI ascii file as drawn from the command
        'sensor_transformer' is expected and transformed to the Neuromag
        coordinate system. If ndarray, the transform is expected to be
        aligned with the Neuromag coordinate system.
    data : bool | array-like
        if array-like custom data matching the header info to be used
        instead of the data from data_fname
    adjust : int | None
        Degrees to tilt x-axis for sensor frame misalignment.
        If None, no adjustment will be applied.
    translation : array-like
        The translation to place the origin of coordinate system
        to the center of the head.
    sep : str
        seperator used for dates.
    use_hpi : bool
        Whether to treat hpi coils as digitization points or not. If
        False, HPI coils will be discarded.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Attributes & Methods
    --------------------
    See documentation for mne.fiff.Raw

    """
    @verbose
    def __init__(self, hdr_fname, data_fname, head_shape_fname, dev_head_t=None,
                 data=None, sep='-', adjust=2, translation=(0.0, 0.02, 0.11),
                 use_hpi=False, verbose=True):

        logger.info('Opening 4-D header file %s...' % hdr_fname)
        self.hdr = read_bti_ascii(hdr_fname)
        self._root, self._hdr_name = op.split(hdr_fname)
        self._data_file = data_fname
        self.head_shape_fname = head_shape_fname
        self._dev_head_t = dev_head_t
        self.sep = sep
        self._use_hpi = use_hpi
        self.bti_to_nm = _get_m_to_nm(adjust, translation)

        logger.info('Creating Neuromag info structure ...')
        info = self._create_raw_info()

        # get the scaling right
        pick_mag = pick_types(info, meg='mag', eeg=False, misc=False,
                              eog=False, ecg=False)
        self._data[pick_mag] *= 1e-15  # put data in Tesla

        self.first_samp, self.last_samp = 0, self._data.shape[1] - 1
        cals = np.zeros(info['nchan'])
        for k in range(info['nchan']):
            cals[k] = info['chs'][k]['range'] * info['chs'][k]['cal']

        self.cals = cals
        self.rawdir = None
        self.proj = None
        self.comp = None
        self.verbose = verbose
        self.fid = None
        self._preloaded = True
        self._times = np.arange(self.first_samp, \
                                self.last_samp + 1) / info['sfreq']
        self._projectors = [None]
        self._projector_hashes = [None]
        self.info = info

        logger.info('Reading raw data from %s...' % data_fname)
        self._data = self._read_4D_data() if not data else data
        assert len(self._data) == len(self.info['ch_names'])
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs' % (
                   self.first_samp, self.last_samp,
                   float(self.first_samp) / info['sfreq'],
                   float(self.last_samp) / info['sfreq']))
        logger.info('Ready.')

        for attr in self.__dict__():
            if attr not in Raw.__dict():
                del attr

    @verbose
    def _create_raw_info(self):
        """ Fills list of dicts for initializing empty fiff with 4D data
        """
        info = {}
        sep = self.sep
        d = datetime.strptime(self.hdr['DATAFILE']['Session'],
                              '%d' + sep + '%y' + sep + '%m %H:%M')

        sec = time.mktime(d.timetuple())
        info['projs'] = []
        info['comps'] = []
        info['meas_date'] = np.array([sec, 0], dtype=np.int32)
        info['sfreq'] = float(self.hdr['FILEINFO']['Sample Frequency'][:-2])
        info['nchan'] = int(self.hdr['CHANNEL GROUPS']['CHANNELS'])
        ch_names = np.array([(e[0], i) for i, e in
                                enumerate(self.hdr['CHANNEL LABELS'], 1)])

        info['ch_names'] = list(_rename_channels(ch_names[:, 0].tolist()))
        ch_mapping = dict(zip(* info['ch_names']))

        sensor_trans = dict((ch_mapping[k], v) for k, v in
                             self.hdr['CHANNEL XFM'].items())
        info['bads'] = []  # TODO

        fspec = self.info.get('DATAFILE', 'PDF')
        if fspec is not None:
            fspec = fspec.split(',')[2].split('ord')[0]
            ffreqs = fspec.replace('fwsbp', '').split('-')
        else:
            logger.info('Cannot fint any filter specification' \
                  '\n No filter info will be set.')
            ffreqs = 0, 1000

        info['highpass'], info['lowpass'] = ffreqs
        info['acq_pars'], info['acq_stim'] = None, None
        info['filename'] = None
        info['ctf_head_t'] = None
        info['dev_ctf_t'] = []
        info['filenames'] = []
        chs = []

        # get 4-D head_dev_t needed for appropriate
        if isinstance(self._dev_head_t, str):
            logger.info('... Reading device to head transform from %s.' %
                        self._dev_head_t)
            bti_sys_trans = _read_system_trans(self._dev_head_t)

        elif hasattr(self._dev_head_t, 'shape'):
            logger.info('... Setting device from matrix.')
            if self._dev_head_t.shape != (4, 4):
                raise ValueError('A transformation matrix of shape 4 x 4 is'
                                 ' expected. Found shape %i x %i instead.'
                                 % self._dev_head_t.shape)
            bti_sys_trans = self._dev_head_t
        else:
            logger.info('... No device to head transform specified. '
                        'Using identity transform instead')
            bti_sys_trans = np.array(BTI.T_IDENT, np.float32)

        logger.info('... Setting channel info structure.')
        for idx, chan in enumerate(info['ch_names'], 1):
            chan_info = dict((k, v) for k, v in zip(FIFF_INFO_CHS_FIELDS,
                             FIFF_INFO_CHS_DEFAULTS))
            chan_info['ch_name'] = chan
            chan_info['logno'] = idx
            chan_info['scanno'] = idx

            if any([chan.startswith(k) for k in ('MEG', 'RFG', 'RFM')]):
                t = _m_to_nm_coil_trans(sensor_trans[chan], bti_sys_trans,
                                        self.bti_to_nm)
                chan_info['coil_trans'] = t
                chan_info['loc'] = np.roll(t.copy().T, 1, 0)[:, :3].flatten()
                chan_info['logno'] = idx

            if chan.startswith('MEG'):
                chan_info['kind'] = FIFF.FIFFV_MEG_CH
                chan_info['coil_type'] = FIFF.FIFFV_COIL_MAGNES_MAG
                chan_info['coord_frame'] = FIFF.FIFFV_COORD_HEAD
                chan_info['unit'] = FIFF.FIFF_UNIT_T

            elif chan.startswith('RFM'):
                chan_info['kind'] = FIFF.FIFFV_REF_MEG_CH
                chan_info['coil_type'] = FIFF.FIFFV_COIL_POINT_MAGNETOMETER
                chan_info['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
                chan_info['unit'] = FIFF.FIFF_UNIT_T

            elif chan.startswith('RFG'):
                chan_info['kind'] = FIFF.FIFFV_REF_MEG_CH
                chan_info['coil_type'] = FIFF.FIFFV_COIL_AXIAL_GRAD_5CM
                chan_info['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
                chan_info['unit'] = FIFF.FIFF_UNIT_T_M

            elif chan == 'STI 014':
                chan_info['kind'] = FIFF.FIFFV_STIM_CH
            elif chan.startswith('EOG'):
                chan_info['kind'] = FIFF.FIFFV_EOG_CH
            elif chan == 'ECG 001':
                chan_info['kind'] = FIFF.FIFFV_ECG_CH
            elif chan == 'RSP 001':
                chan_info['kind'] = FIFF.FIFFV_RESP_CH
            elif chan.startswith('EXT'):
                chan_info['kind'] = FIFF.FIFFV_MISC_CH
            elif chan.startswith('UTL'):
                chan_info['kind'] = FIFF.FIFFV_MISC_CH

            chs.append(chan_info)

        info['chs'] = chs
        info['meas_id'] = None
        info['file_id'] = None

        identity = np.array(BTI.T_IDENT, np.float32)
        if self.head_shape_fname is not None:
            logger.info('... Reading digitization points from %s' %
                        self.head_shape_fname)
            info['dig'], m_h_nm_h = read_head_shape(self.head_shape_fname,
                                                     self._use_hpi)
            nm_to_m_sensor = _inverse_t(identity, self.bti_to_nm)
            nm_sensor_m_head = _merge_t(bti_sys_trans, nm_to_m_sensor)
            nm_dev_head_t = _merge_t(m_h_nm_h, nm_sensor_m_head)
            nm_dev_head_t[3, :3] = 0.
        else:
            logger.info('Warning. No head shape file provided. '
                        'Sensor data will stay in head coordinate frame')
            nm_dev_head_t = identity

        info['dev_head_t'] = {}
        info['dev_head_t']['from'] = FIFF.FIFFV_COORD_DEVICE
        info['dev_head_t']['to'] = FIFF.FIFFV_COORD_HEAD
        info['dev_head_t']['trans'] = nm_dev_head_t

        return info

    def _read_4D_data(self, count=-1, dtype=np.float32):
        """ Reads data from binary string file (dumped: time slices x channels)
        """
        ntsl = int(self.hdr['FILEINFO']['Time slices'].replace(' slices', ''))
        cnt, dtp = count, dtype
        with open(self._data_file, 'rb') as f:
            shape = (ntsl, self.info['nchan'])
            data = np.fromfile(file=f, dtype=dtp, ount=cnt).reshape(shape)

        return data.T


if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-i', '--hdr_fname', dest='hdr_fname',
                    help='Input data header file ', metavar='FILE')
    parser.add_option('-d', '--data_fname', dest='data_fname',
                    help='Input data file ', metavar='FILE')
    parser.add_option('-s', '--head_shape_fname', dest='head_shape_fname',
                    help='Headshape file name', metavar='FILE')
    parser.add_option('-t', '--dev_head_t_fname', dest='dev_head_t_fname',
                      help='Device head transform file name', metavar='FILE')
    parser.add_option('-o', '--out_fname', dest='out_fname', default='as_data_fname')
    parser.add_option('-r', '--rotation_x', dest='rotation_x', type='float',
                    help='Compensatory rotation about Neuromag x axis, deg',
                    default=2.0)
    parser.add_option('-T', '--translation', dest='translation', type='str',
                    help='Default translation, meter',
                    default=(0.00, 0.020, 0.11))
    parser.add_option('-u', '--use_hpi', dest='use_hpi',
                    help='Use all or onlye the first three HPI coils',
                    default=False)
    parser.add_option('-S', '--seperator', dest='seperator', type='str',
                    help='seperator used for date parsing', default='-')
    parser.add_option('-v', '--verbose', dest='verbose',
                    help='Print single processing steps to command line',
                    default=True)

    options, args = parser.parse_args()

    data_fname = options.data_fname
    hdr_fname = options.hdr_fname
    head_shape_fname = options.head_shape_fname
    dev_head_t_fname = options.dev_head_t_fname
    out_fname = options.out_fname
    rotation_x = options.rotation_x
    translation = options.translation
    seperator = options.seperator
    use_hpi = options.use_hpi
    verbose = options.verbose

    if any([o is None for o in [data_fname,  hdr_fname]]):
        parser.print_help()
        sys.exit(-1)

    if out_fname == 'as_data_fname':
        out_fname = data_fname.split('.data')[0] + '_raw.fif'

    raw = RawBTi(hdr_fname=hdr_fname, data_fname=data_fname,
                 head_shape=head_shape_fname, dev_head_t=dev_head_t_fname,
                 adjust=rotation_x, trasnlation=translation, sep=seperator,
                 use_hpi=use_hpi, verbose=verbose)

    raw.save(out_fname)
    raw.close()
    sys.exit(0)
