"""Conversion tool from SQD to FIF

RawKIT class is adapted from Denis Engemann et al.'s mne_bti2fiff.py

"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import logging
import os
from os import SEEK_CUR
from struct import unpack
import time
import warnings

import numpy as np
from scipy.linalg import norm

from ...fiff import pick_types
from ...transforms.coreg import fit_matched_pts
from ...utils import verbose
from ..raw import Raw
from ..constants import FIFF
from ..meas_info import Info
from .constants import KIT, KIT_NY, KIT_AD
from .coreg import get_dig_points, read_elp, read_hsp, read_mrk, read_sns, \
                   get_neuromag_transform, transform_ALS_to_RAS

logger = logging.getLogger('mne')


class RawKIT(Raw):
    """Raw object from KIT SQD file adapted from bti/raw.py

    Parameters
    ----------
    input_fname : str
        Absolute path to the sqd file.
    mrk : str | array, shape = (5, 3)
        Absolute path to marker coils file, or array of points.
    elp : str | array, shape = (8, 3)
        Absolute path to elp digitizer laser points file, or array with points.
    hsp : str | array, shape = (n_pts, 3)
        Absolute path to elp digitizer head shape points file, or array with
        points.
    sns_fname : str
        Absolute path to sensor information file.
    stim : list of int | '<' | '>'
        Can be submitted as list of trigger channels.
        If a list is not specified, the default triggers extracted from
        misc channels will be used with specified directionality.
        '<' means that largest values assigned to the first channel
        in sequence.
        '>' means the largest trigger assigned to the last channel
        in sequence.
    slope : '+' | '-'
        '+' means a positive slope (low-to-high) on the event channel(s)
        is used to trigger an event.
        '-' means a negative slope (high-to-low) on the event channel(s)
        is used to trigger an event.
    stimthresh : float
        The threshold level for accepting voltage change as a trigger event.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.

    See Also
    --------
    mne.fiff.Raw : Documentation of attribute and methods.
    """
    @verbose
    def __init__(self, input_fname, mrk=None, elp=None, hsp=None,
                 stim='>', slope='-', stimthresh=1,
                 verbose=None, preload=False):

        logger.info('Extracting SQD Parameters from %s...' % input_fname)
        self._sqd_params = get_sqd_params(input_fname)
        self._sqd_params['slope'] = slope
        self._sqd_params['stimthresh'] = stimthresh
        self._sqd_params['fname'] = input_fname
        logger.info('Creating Raw.info structure...')

        # Raw attributes
        self.verbose = verbose
        self._preloaded = preload
        self.fids = list()
        self._projector = None
        self.first_samp = 0
        self.last_samp = self._sqd_params['nsamples'] - 1
        self.comp = None  # no compensation for KIT

        # Create raw.info dict for raw fif object with SQD data
        self.info = Info()
        self.info['meas_id'] = None
        self.info['file_id'] = None
        self.info['meas_date'] = int(time.time())
        self.info['projs'] = []
        self.info['comps'] = []
        self.info['lowpass'] = self._sqd_params['lowpass']
        self.info['highpass'] = self._sqd_params['highpass']
        self.info['sfreq'] = float(self._sqd_params['sfreq'])
        # meg channels plus synthetic channel
        self.info['nchan'] = self._sqd_params['nchan'] + 1
        self.info['bads'] = []
        self.info['acq_pars'], self.info['acq_stim'] = None, None
        self.info['filename'] = None
        self.info['ctf_head_t'] = None
        self.info['dev_ctf_t'] = []
        self.info['filenames'] = []
        self.info['dev_head_t'] = {}
        self.info['dev_head_t']['from'] = FIFF.FIFFV_COORD_DEVICE
        self.info['dev_head_t']['to'] = FIFF.FIFFV_COORD_HEAD

        if (mrk and elp and hsp):
            self.set_dig(mrk, elp, hsp)
        elif (mrk or elp or hsp):
            err = ("mrk, elp and hsp need to be provided as a group (all or "
                   "none)")
            raise ValueError(err)

        # Creates a list of dicts of meg channels for raw.info
        logger.info('Setting channel info structure...')
        ch_names = {}
        ch_names['MEG'] = ['MEG %03d' % ch for ch
                                in range(1, self._sqd_params['n_sens'] + 1)]
        ch_names['MISC'] = ['MISC %03d' % ch for ch
                                 in range(1, self._sqd_params['nmiscchan']
                                          + 1)]
        ch_names['STIM'] = ['STI 014']
        locs = read_sns(sns_fname=sns_fname)
        chan_locs = transform_ALS_to_RAS(locs[:, :3])
        chan_angles = locs[:, 3:]
        self.info['chs'] = []
        for idx, ch_info in enumerate(zip(ch_names['MEG'], chan_locs,
                                          chan_angles), 1):
            ch_name, ch_loc, ch_angles = ch_info
            chan_info = {}
            chan_info['cal'] = KIT.CALIB_FACTOR
            chan_info['logno'] = idx
            chan_info['scanno'] = idx
            chan_info['range'] = KIT.RANGE
            chan_info['unit_mul'] = KIT.UNIT_MUL
            chan_info['ch_name'] = ch_name
            chan_info['unit'] = FIFF.FIFF_UNIT_T
            chan_info['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
            if idx <= self._sqd_params['nmegchan']:
                chan_info['coil_type'] = FIFF.FIFFV_COIL_KIT_GRAD
                chan_info['kind'] = FIFF.FIFFV_MEG_CH
            else:
                chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
                chan_info['kind'] = FIFF.FIFFV_REF_MEG_CH

            # create three orthogonal vector
            # ch_angles[0]: theta, ch_angles[1]: phi
            ch_angles = np.radians(ch_angles)
            x = np.sin(ch_angles[0]) * np.cos(ch_angles[1])
            y = np.sin(ch_angles[0]) * np.sin(ch_angles[1])
            z = np.cos(ch_angles[0])
            vec_z = np.array([x, y, z])
            length = norm(vec_z)
            vec_z /= length
            vec_x = np.zeros(vec_z.size, dtype=np.float)
            if vec_z[1] < vec_z[2]:
                if vec_z[0] < vec_z[1]:
                    vec_x[0] = 1.0
                else:
                    vec_x[1] = 1.0
            elif vec_z[0] < vec_z[2]:
                vec_x[0] = 1.0
            else:
                vec_x[2] = 1.0
            vec_x -= np.sum(vec_x * vec_z) * vec_z
            length = norm(vec_x)
            vec_x /= length
            vec_y = np.cross(vec_z, vec_x)
            # transform to Neuromag like coordinate space
            vecs = np.vstack((vec_x, vec_y, vec_z))
            vecs = transform_ALS_to_RAS(vecs, unit='m')
            chan_info['loc'] = np.vstack((ch_loc, vecs)).ravel()
            self.info['chs'].append(chan_info)

        # label trigger and misc channels
        for idy, ch_name in enumerate(ch_names['MISC'] + ch_names['STIM'],
                                      self._sqd_params['n_sens']):
            chan_info = {}
            chan_info['cal'] = KIT.CALIB_FACTOR
            chan_info['logno'] = idy
            chan_info['scanno'] = idy
            chan_info['range'] = 1.0
            chan_info['unit'] = FIFF.FIFF_UNIT_V
            chan_info['unit_mul'] = 0  # default is 0 mne_manual p.273
            chan_info['ch_name'] = ch_name
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['loc'] = np.zeros(12)
            if ch_name.startswith('STI'):
                chan_info['unit'] = FIFF.FIFF_UNIT_NONE
                chan_info['kind'] = FIFF.FIFFV_STIM_CH
            else:
                chan_info['kind'] = FIFF.FIFFV_MISC_CH
            self.info['chs'].append(chan_info)
        self.info['ch_names'] = (ch_names['MEG'] + ch_names['MISC'] +
                                 ch_names['STIM'])

        # Acquire stim channels
        if isinstance(stim, str):
            picks = pick_types(self.info, meg=False, misc=True,
                               exclude=[])[:8]
            if stim == '<':
                stim = picks[::-1]
            elif stim == '>':
                stim = picks
            else:
                raise ValueError("stim needs to be list of int, '>' or '<', "
                                 "not %r" % stim)
        self._sqd_params['stim'] = stim

        if self._preloaded:
            logger.info('Reading raw data from %s...' % input_fname)
            self._data, _ = self._read_segment()
            assert len(self._data) == self.info['nchan']

            # Create a synthetic channel
            trig_chs = self._data[stim, :]
            if slope == '+':
                trig_chs = trig_chs > stimthresh
            elif slope == '-':
                trig_chs = trig_chs < stimthresh
            else:
                raise ValueError("slope needs to be '+' or '-'")
            trig_vals = np.array(2 ** np.arange(len(stim)), ndmin=2).T
            trig_chs = trig_chs * trig_vals
            stim_ch = trig_chs.sum(axis=0)
            self._data[-1, :] = stim_ch

            # Add time info
            self.first_samp, self.last_samp = 0, self._data.shape[1] - 1
            self._times = np.arange(self.first_samp, self.last_samp + 1,
                                    dtype=np.float64)
            self._times /= self.info['sfreq']
            logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs'
                        % (self.first_samp, self.last_samp,
                           float(self.first_samp) / self.info['sfreq'],
                           float(self.last_samp) / self.info['sfreq']))
        logger.info('Ready.')

    def __repr__(self):
        s = ('%r' % os.path.basename(self._sqd_params['fname']),
             "n_channels x n_times : %s x %s" % (len(self.info['ch_names']),
                                       self.last_samp - self.first_samp + 1))
        return "<RawKIT  |  %s>" % ', '.join(s)

    def read_stim_ch(self, buffer_size=1e5):
        """Read events from data

        Parameter
        ---------
        buffer_size : int
            The size of chunk to by which the data are scanned.

        Returns
        -------
        events : array, [samples]
           The event vector (1 x samples).
        """
        buffer_size = int(buffer_size)
        start = int(self.first_samp)
        stop = int(self.last_samp + 1)

        pick = pick_types(self.info, meg=False, stim=True, exclude=[])
        stim_ch = np.empty((1, stop), dtype=np.int)
        for b_start in range(start, stop, buffer_size):
            b_stop = b_start + buffer_size
            x, _ = self._read_segment(start=b_start, stop=b_stop, sel=pick)
            stim_ch[:, b_start:b_start + x.shape[1]] = x

        return stim_ch

    def _read_segment(self, start=0, stop=None, sel=None, verbose=None,
                      projector=None):
        """Read a chunk of raw data

        Parameters
        ----------
        start : int, (optional)
            first sample to include (first is 0). If omitted, defaults to the
            first sample in data.
        stop : int, (optional)
            First sample to not include.
            If omitted, data is included to the end.
        sel : array, optional
            Indices of channels to select.
        projector : array
            SSP operator to apply to the data.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).

        Returns
        -------
        data : array, [channels x samples]
           the data matrix (channels x samples).
        times : array, [samples]
            returns the time values corresponding to the samples.
        """
        if sel is None:
            sel = range(self.info['nchan'])
        elif len(sel) == 1 and sel[0] == 0 and start == 0 and stop == 1:
            return (666, 666)
        if projector is not None:
            raise NotImplementedError('Currently does not handle projections.')
        if stop is None:
            stop = self.last_samp + 1
        elif stop > self.last_samp + 1:
            stop = self.last_samp + 1

        #  Initial checks
        start = int(start)
        stop = int(stop)

        if start >= stop:
            raise ValueError('No data in this range')

        logger.info('Reading %d ... %d  =  %9.3f ... %9.3f secs...' %
                    (start, stop - 1, start / float(self.info['sfreq']),
                               (stop - 1) / float(self.info['sfreq'])))

        with open(self._sqd_params['fname'], 'r') as fid:
            # extract data
            fid.seek(KIT.DATA_OFFSET)
            # data offset info
            data_offset = unpack('i', fid.read(KIT.INT))[0]
            nchan = self._sqd_params['nchan']
            buffer_size = stop - start
            count = buffer_size * nchan
            pointer = start * nchan * KIT.SHORT
            fid.seek(data_offset + pointer)
            data = np.fromfile(fid, dtype='h', count=count)
            data = data.reshape((buffer_size, nchan))
        # amplifier applies only to the sensor channels
        n_sens = self._sqd_params['n_sens']
        sensor_gain = np.copy(self._sqd_params['sensor_gain'])
        sensor_gain[:n_sens] = (sensor_gain[:n_sens] /
                                self._sqd_params['amp_gain'])
        conv_factor = np.array((KIT.VOLTAGE_RANGE /
                                self._sqd_params['DYNAMIC_RANGE'])
                               * sensor_gain, ndmin=2)
        data = conv_factor * data
        data = data.T
        # Create a synthetic channel
        trig_chs = data[self._sqd_params['stim'], :]
        if self._sqd_params['slope'] == '+':
            trig_chs = trig_chs > self._sqd_params['stimthresh']
        elif self._sqd_params['slope'] == '-':
            trig_chs = trig_chs < self._sqd_params['stimthresh']
        else:
            raise ValueError("slope needs to be '+' or '-'")
        trig_vals = np.array(2 ** np.arange(len(self._sqd_params['stim'])),
                             ndmin=2).T
        trig_chs = trig_chs * trig_vals
        stim_ch = np.array(trig_chs.sum(axis=0), ndmin=2)
        data = np.vstack((data, stim_ch))
        data = data[sel]

        logger.info('[done]')
        times = np.arange(start, stop) / self.info['sfreq']

        return data, times

    def set_dig(self, mrk, elp, hsp):
        """
        Fill in the digitizer data using points in Polhemus space

        Parameters
        ----------
        mrk : None | array, shape = (5, 3)
            Marker points used to estimate the device head transform. If trans
            is provided, mrk is not used and can be None).
        max_hsp_n : None | int
            Maximum number of head shape points to keep.
        """
        if isinstance(hsp, basestring):
            hsp = read_hsp(hsp)
            n_pts = len(hsp)
            if n_pts > KIT.DIG_POINTS:
                msg = ("The selected head shape contains %i points, which is "
                       "more than recommended (%i)" % (n_pts, KIT.DIG_POINTS))
                warnings.warn(msg)

        if isinstance(elp, basestring):
            elp = read_elp(elp)

        if isinstance(mrk, basestring):
            mrk = read_mrk(mrk)

        hsp = transform_ALS_to_RAS(hsp)
        elp = transform_ALS_to_RAS(elp)
        mrk = transform_ALS_to_RAS(mrk, unit='m')

        nasion, lpa, rpa = elp[:3]
        nmtrans = get_neuromag_transform(nasion, lpa, rpa).T
        elp = np.dot(elp, nmtrans)
        hsp = np.dot(hsp, nmtrans)

        # device head transform
        trans = fit_matched_pts(tgt_pts=elp[3:], src_pts=mrk, out='trans')

        self.set_transformed_dig(elp[:3], elp[3:], hsp, trans)

    def set_transformed_dig(self, fid, elp, hsp, trans):
        """
        Fill in the digitizer data using points that are already transformed to
        neuromag space

        Parameters
        ----------
        fid : array, shape = (3, 3)
            Digitizer fiducials.
        elp : array, shape = (5, 3)
            Digitizer ELP points.
        trans : None | array, shape = (4, 4)
            Device head transformation.
        """
        self.info['dig'] = get_dig_points(fid, elp, hsp)

        trans = np.asarray(trans)
        if not trans.shape == (4, 4):
            raise ValueError("trans needs to be 4 by 4 array")
        self.info['dev_head_t']['trans'] = trans



def get_sqd_params(rawfile):
    """Extracts all the information from the sqd file.

    Parameters
    ----------
    rawfile : str
        Raw sqd file to be read.

    Returns
    -------
    sqd : dict
        A dict containing all the sqd parameter settings.
    """
    sqd = dict()
    sqd['rawfile'] = rawfile
    with open(rawfile, 'r') as fid:
        fid.seek(KIT.BASIC_INFO)
        basic_offset = unpack('i', fid.read(KIT.INT))[0]
        fid.seek(basic_offset)
        # skips version, revision, sysid
        fid.seek(KIT.INT * 3, SEEK_CUR)
        # basic info
        sysname = unpack('128s', fid.read(KIT.STRING))
        sysname = sysname[0].split('\n')[0]
        fid.seek(KIT.STRING, SEEK_CUR)  # skips modelname
        sqd['nchan'] = unpack('i', fid.read(KIT.INT))[0]

        if sysname == 'New York University Abu Dhabi':
            KIT_SYS = KIT_AD
        elif sysname == 'NYU 160ch System since Jan24 2009':
            KIT_SYS = KIT_NY
        else:
            raise NotImplementedError

        # amplifier gain
        fid.seek(KIT_SYS.AMPLIFIER_INFO)
        amp_offset = unpack('i', fid.read(KIT_SYS.INT))[0]
        fid.seek(amp_offset)
        amp_data = unpack('i', fid.read(KIT_SYS.INT))[0]

        gain1 = KIT_SYS.GAINS[(KIT_SYS.GAIN1_MASK & amp_data)
                              >> KIT_SYS.GAIN1_BIT]
        gain2 = KIT_SYS.GAINS[(KIT_SYS.GAIN2_MASK & amp_data)
                              >> KIT_SYS.GAIN2_BIT]
        if KIT_SYS.GAIN3_BIT:
            gain3 = KIT_SYS.GAINS[(KIT_SYS.GAIN3_MASK & amp_data)
                                     >> KIT_SYS.GAIN3_BIT]
            sqd['amp_gain'] = gain1 * gain2 * gain3
        else:
            sqd['amp_gain'] = gain1 * gain2

        # filter settings
        sqd['lowpass'] = KIT_SYS.LPFS[(KIT_SYS.LPF_MASK & amp_data)
                                      >> KIT_SYS.LPF_BIT]
        sqd['highpass'] = KIT_SYS.HPFS[(KIT_SYS.HPF_MASK & amp_data)
                                       >> KIT_SYS.HPF_BIT]
        sqd['notch'] = KIT_SYS.BEFS[(KIT_SYS.BEF_MASK & amp_data)
                                    >> KIT_SYS.BEF_BIT]

        # only sensor channels requires gain. the additional misc channels
        # (trigger channels, audio and voice channels) are passed
        # through unaffected

        fid.seek(KIT_SYS.CHAN_SENS)
        sens_offset = unpack('i', fid.read(KIT_SYS.INT))[0]
        fid.seek(sens_offset)
        sens = np.fromfile(fid, dtype='d', count=sqd['nchan'] * 2)
        sensitivities = (np.reshape(sens, (sqd['nchan'], 2))
                         [:KIT_SYS.n_sens, 1])
        sqd['sensor_gain'] = np.ones(KIT_SYS.nchan)
        sqd['sensor_gain'][:KIT_SYS.n_sens] = sensitivities

        fid.seek(KIT_SYS.SAMPLE_INFO)
        acqcond_offset = unpack('i', fid.read(KIT_SYS.INT))[0]
        fid.seek(acqcond_offset)
        acq_type = unpack('i', fid.read(KIT_SYS.INT))[0]
        if acq_type == 1:
            sqd['sfreq'] = unpack('d', fid.read(KIT_SYS.DOUBLE))[0]
            _ = fid.read(KIT_SYS.INT)  # initialized estimate of samples
            sqd['nsamples'] = unpack('i', fid.read(KIT_SYS.INT))[0]
        else:
            raise NotImplementedError
        sqd['n_sens'] = KIT_SYS.n_sens
        sqd['nmegchan'] = KIT_SYS.nmegchan
        sqd['nmiscchan'] = KIT_SYS.nmiscchan
        sqd['DYNAMIC_RANGE'] = KIT_SYS.DYNAMIC_RANGE
    return sqd


def read_raw_kit(input_fname, mrk, elp, hsp,
                 stim='>', slope='-', stimthresh=1,
                 verbose=None, preload=False):
    """Reader function for KIT conversion to FIF

    Parameters
    ----------
    input_fname : str
        Absolute path to the sqd file.
    mrk : str | array, shape = (5, 3)
        Absolute path to marker coils file, or array of points.
    elp : str | array, shape = (8, 3)
        Absolute path to elp digitizer laser points file, or array with points.
    hsp : str | array, shape = (n_pts, 3)
        Absolute path to elp digitizer head shape points file, or array with
        points.
    sns_fname : str
        Absolute path to sensor information file.
    stim : list of int | '<' | '>'
        Can be submitted as list of trigger channels.
        If a list is not specified, the default triggers extracted from
        misc channels, will be used with specified directionality.
        '<' means that largest values assigned to the first channel
        in sequence.
        '>' means the largest trigger assigned to the last channel
        in sequence.
    slope : '+' | '-'
        '+' means a positive slope (low-to-high) on the event channel(s)
        is used to trigger an event.
        '-' means a negative slope (high-to-low) on the event channel(s)
        is used to trigger an event.
    stimthresh : float
        The threshold level for accepting voltage change as a trigger event.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.
    """
    return RawKIT(input_fname=input_fname, mrk=mrk, elp=elp, hsp=hsp,
                  sns_fname=sns_fname, stim=stim, slope=slope,
                  stimthresh=stimthresh, verbose=verbose, preload=preload)
