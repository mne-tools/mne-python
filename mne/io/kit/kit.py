"""Conversion tool from SQD to FIF

RawKIT class is adapted from Denis Engemann et al.'s mne_bti2fiff.py

"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import os
from os import SEEK_CUR
from struct import unpack
import time

import numpy as np
from scipy import linalg

from ..pick import pick_types
from ...coreg import (read_elp, fit_matched_points, _decimate_points,
                      get_ras_to_neuromag_trans)
from ...utils import verbose, logger
from ...transforms import apply_trans, als_ras_trans, als_ras_trans_mm
from ..base import _BaseRaw
from ..constants import FIFF
from ..meas_info import Info
from ..tag import _loc_to_trans
from .constants import KIT, KIT_NY, KIT_AD
from .coreg import read_hsp, read_mrk
from ...externals.six import string_types


class RawKIT(_BaseRaw):
    """Raw object from KIT SQD file adapted from bti/raw.py

    Parameters
    ----------
    input_fname : str
        Path to the sqd file.
    mrk : None | str | array_like, shape = (5, 3) | list of str or array_like
        Marker points representing the location of the marker coils with
        respect to the MEG Sensors, or path to a marker file.
        If list, all of the markers will be averaged together.
    elp : None | str | array_like, shape = (8, 3)
        Digitizer points representing the location of the fiducials and the
        marker coils with respect to the digitized head shape, or path to a
        file containing these points.
    hsp : None | str | array, shape = (n_points, 3)
        Digitizer head shape points, or path to head shape file. If more than
        10`000 points are in the head shape, they are automatically decimated.
    stim : list of int | '<' | '>'
        Channel-value correspondence when converting KIT trigger channels to a
        Neuromag-style stim channel. For '<', the largest values are assigned
        to the first channel (default). For '>', the largest values are
        assigned to the last channel. Can also be specified as a list of
        trigger channel indexes.
    slope : '+' | '-'
        How to interpret values on KIT trigger channels when synthesizing a
        Neuromag-style stim channel. With '+', a positive slope (low-to-high)
        is interpreted as an event. With '-', a negative slope (high-to-low)
        is interpreted as an event.
    stimthresh : float
        The threshold level for accepting voltage changes in KIT trigger
        channels as a trigger event.
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
    def __init__(self, input_fname, mrk=None, elp=None, hsp=None, stim='>',
                 slope='-', stimthresh=1, preload=False, verbose=None):
        logger.info('Extracting SQD Parameters from %s...' % input_fname)
        input_fname = os.path.abspath(input_fname)
        self._sqd_params = get_sqd_params(input_fname)
        self._sqd_params['stimthresh'] = stimthresh
        self._sqd_params['fname'] = input_fname
        logger.info('Creating Raw.info structure...')

        # Raw attributes
        self.verbose = verbose
        self.preload = False
        self._projector = None
        self.first_samp = 0
        self.last_samp = self._sqd_params['nsamples'] - 1
        self.comp = None  # no compensation for KIT
        self.proj = False

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
        self._filenames = []
        self.info['dig'] = None
        self.info['dev_head_t'] = None

        if isinstance(mrk, list):
            mrk = [read_mrk(marker) if isinstance(marker, string_types)
                   else marker for marker in mrk]
            mrk = np.mean(mrk, axis=0)

        if (mrk is not None and elp is not None and hsp is not None):
            self._set_dig_kit(mrk, elp, hsp)
        elif (mrk is not None or elp is not None or hsp is not None):
            err = ("mrk, elp and hsp need to be provided as a group (all or "
                   "none)")
            raise ValueError(err)

        # Creates a list of dicts of meg channels for raw.info
        logger.info('Setting channel info structure...')
        ch_names = {}
        ch_names['MEG'] = ['MEG %03d' % ch for ch
                           in range(1, self._sqd_params['n_sens'] + 1)]
        ch_names['MISC'] = ['MISC %03d' % ch for ch
                            in range(1, self._sqd_params['nmiscchan'] + 1)]
        ch_names['STIM'] = ['STI 014']
        locs = self._sqd_params['sensor_locs']
        chan_locs = apply_trans(als_ras_trans, locs[:, :3])
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
                chan_info['coil_type'] = FIFF.FIFFV_COIL_KIT_REF_MAG
                chan_info['kind'] = FIFF.FIFFV_REF_MEG_CH
            chan_info['eeg_loc'] = None

            # create three orthogonal vector
            # ch_angles[0]: theta, ch_angles[1]: phi
            ch_angles = np.radians(ch_angles)
            x = np.sin(ch_angles[0]) * np.cos(ch_angles[1])
            y = np.sin(ch_angles[0]) * np.sin(ch_angles[1])
            z = np.cos(ch_angles[0])
            vec_z = np.array([x, y, z])
            length = linalg.norm(vec_z)
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
            length = linalg.norm(vec_x)
            vec_x /= length
            vec_y = np.cross(vec_z, vec_x)
            # transform to Neuromag like coordinate space
            vecs = np.vstack((vec_x, vec_y, vec_z))
            vecs = apply_trans(als_ras_trans, vecs)
            chan_info['loc'] = np.vstack((ch_loc, vecs)).ravel()
            chan_info['coil_trans'] = _loc_to_trans(chan_info['loc'])
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

        self._set_stimchannels(stim, slope)
        if preload:
            self.preload = preload
            logger.info('Reading raw data from %s...' % input_fname)
            self._data, _ = self._read_segment()
            assert len(self._data) == self.info['nchan']

            # Create a synthetic channel
            stim = self._sqd_params['stim']
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
                                                 self.last_samp -
                                                 self.first_samp + 1))
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

        pick = pick_types(self.info, meg=False, ref_meg=False,
                          stim=True, exclude=[])
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
            sel = list(range(self.info['nchan']))
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

        with open(self._sqd_params['fname'], 'rb', buffering=0) as fid:
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

    def _set_dig_kit(self, mrk, elp, hsp, auto_decimate=True):
        """Add landmark points and head shape data to the RawKIT instance

        Digitizer data (elp and hsp) are represented in [mm] in the Polhemus
        ALS coordinate system.

        Parameters
        ----------
        mrk : None | str | array_like, shape = (5, 3)
            Marker points representing the location of the marker coils with
            respect to the MEG Sensors, or path to a marker file.
        elp : None | str | array_like, shape = (8, 3)
            Digitizer points representing the location of the fiducials and the
            marker coils with respect to the digitized head shape, or path to a
            file containing these points.
        hsp : None | str | array, shape = (n_points, 3)
            Digitizer head shape points, or path to head shape file. If more
            than 10`000 points are in the head shape, they are automatically
            decimated.
        auto_decimate : bool
            Decimate hsp points for head shape files with more than 10'000
            points.
        """
        if isinstance(hsp, string_types):
            hsp = read_hsp(hsp)

        n_pts = len(hsp)
        if n_pts > KIT.DIG_POINTS:
            hsp = _decimate_points(hsp, 5)
            n_new = len(hsp)
            msg = ("The selected head shape contained {n_in} points, which is "
                   "more than recommended ({n_rec}), and was automatically "
                   "downsampled to {n_new} points. The preferred way to "
                   "downsample is using FastScan.")
            msg = msg.format(n_in=n_pts, n_rec=KIT.DIG_POINTS, n_new=n_new)
            logger.warning(msg)

        if isinstance(elp, string_types):
            elp_points = read_elp(elp)[:8]
            if len(elp) < 8:
                err = ("File %r contains fewer than 8 points; got shape "
                       "%s." % (elp, elp_points.shape))
                raise ValueError(err)
            elp = elp_points

        if isinstance(mrk, string_types):
            mrk = read_mrk(mrk)

        hsp = apply_trans(als_ras_trans_mm, hsp)
        elp = apply_trans(als_ras_trans_mm, elp)
        mrk = apply_trans(als_ras_trans, mrk)

        nasion, lpa, rpa = elp[:3]
        nmtrans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
        elp = apply_trans(nmtrans, elp)
        hsp = apply_trans(nmtrans, hsp)

        # device head transform
        trans = fit_matched_points(tgt_pts=elp[3:], src_pts=mrk, out='trans')

        self._set_dig_neuromag(elp[:3], elp[3:], hsp, trans)

    def _set_dig_neuromag(self, fid, elp, hsp, trans):
        """Fill in the digitizer data using points in neuromag space

        Parameters
        ----------
        fid : array, shape = (3, 3)
            Digitizer fiducials.
        elp : array, shape = (5, 3)
            Digitizer ELP points.
        hsp : array, shape = (n_points, 3)
            Head shape points.
        trans : None | array, shape = (4, 4)
            Device head transformation.
        """
        trans = np.asarray(trans)
        if fid.shape != (3, 3):
            raise ValueError("fid needs to be a 3 by 3 array")
        if elp.shape != (5, 3):
            raise ValueError("elp needs to be a 5 by 3 array")
        if trans.shape != (4, 4):
            raise ValueError("trans needs to be 4 by 4 array")

        nasion, lpa, rpa = fid
        dig = [{'r': nasion, 'ident': FIFF.FIFFV_POINT_NASION,
                'kind': FIFF.FIFFV_POINT_CARDINAL,
                'coord_frame':  FIFF.FIFFV_COORD_HEAD},
               {'r': lpa, 'ident': FIFF.FIFFV_POINT_LPA,
                'kind': FIFF.FIFFV_POINT_CARDINAL,
                'coord_frame': FIFF.FIFFV_COORD_HEAD},
               {'r': rpa, 'ident': FIFF.FIFFV_POINT_RPA,
                'kind': FIFF.FIFFV_POINT_CARDINAL,
                'coord_frame': FIFF.FIFFV_COORD_HEAD}]

        for idx, point in enumerate(elp):
            dig.append({'r': point, 'ident': idx, 'kind': FIFF.FIFFV_POINT_HPI,
                        'coord_frame': FIFF.FIFFV_COORD_HEAD})

        for idx, point in enumerate(hsp):
            dig.append({'r': point, 'ident': idx,
                        'kind': FIFF.FIFFV_POINT_EXTRA,
                        'coord_frame': FIFF.FIFFV_COORD_HEAD})

        dev_head_t = {'from': FIFF.FIFFV_COORD_DEVICE,
                      'to': FIFF.FIFFV_COORD_HEAD, 'trans': trans}

        self.info['dig'] = dig
        self.info['dev_head_t'] = dev_head_t

    def _set_stimchannels(self, stim='<', slope='-'):
        """Specify how the trigger channel is synthesized form analog channels.

        Has to be done before loading data. For a RawKIT instance that has been
        created with preload=True, this method will raise a
        NotImplementedError.

        Parameters
        ----------
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
        """
        if self.preload:
            err = "Can't change stim channel after preloading data"
            raise NotImplementedError(err)

        self._sqd_params['slope'] = slope

        if isinstance(stim, str):
            picks = pick_types(self.info, meg=False, ref_meg=False,
                               misc=True, exclude=[])[:8]
            if stim == '<':
                stim = picks[::-1]
            elif stim == '>':
                stim = picks
            else:
                raise ValueError("stim needs to be list of int, '>' or "
                                 "'<', not %r" % str(stim))
        elif np.max(stim) >= self._sqd_params['nchan']:
            msg = ("Tried to set stim channel %i, but squid file only has %i"
                   " channels" % (np.max(stim), self._sqd_params['nchan']))
            raise ValueError(msg)

        self._sqd_params['stim'] = stim


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
    with open(rawfile, 'rb', buffering=0) as fid:  # buffering=0 for np bug
        fid.seek(KIT.BASIC_INFO)
        basic_offset = unpack('i', fid.read(KIT.INT))[0]
        fid.seek(basic_offset)
        # skips version, revision, sysid
        fid.seek(KIT.INT * 3, SEEK_CUR)
        # basic info
        sysname = unpack('128s', fid.read(KIT.STRING))
        sysname = sysname[0].decode().split('\n')[0]
        fid.seek(KIT.STRING, SEEK_CUR)  # skips modelname
        sqd['nchan'] = unpack('i', fid.read(KIT.INT))[0]

        if sysname == 'New York University Abu Dhabi':
            KIT_SYS = KIT_AD
        elif sysname == 'NYU 160ch System since Jan24 2009':
            KIT_SYS = KIT_NY
        else:
            raise NotImplementedError

        # channel locations
        fid.seek(KIT_SYS.CHAN_LOC_OFFSET)
        chan_offset = unpack('i', fid.read(KIT.INT))[0]
        chan_size = unpack('i', fid.read(KIT.INT))[0]

        fid.seek(chan_offset)
        sensors = []
        for i in range(KIT_SYS.N_SENS):
            fid.seek(chan_offset + chan_size * i)
            sens_type = unpack('i', fid.read(KIT.INT))[0]
            if sens_type == 1:
                # magnetometer
                # x,y,z,theta,phi,coilsize
                sensors.append(np.fromfile(fid, dtype='d', count=6))
            elif sens_type == 2:
                # axialgradiometer
                # x,y,z,theta,phi,baseline,coilsize
                sensors.append(np.fromfile(fid, dtype='d', count=7))
            elif sens_type == 3:
                # planargradiometer
                # x,y,z,theta,phi,btheta,bphi,baseline,coilsize
                sensors.append(np.fromfile(fid, dtype='d', count=9))
            elif sens_type == 257:
                # reference channels
                sensors.append(np.zeros(7))
                sqd['i'] = sens_type
        sqd['sensor_locs'] = np.array(sensors)

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
                         [:KIT_SYS.N_SENS, 1])
        sqd['sensor_gain'] = np.ones(KIT_SYS.NCHAN)
        sqd['sensor_gain'][:KIT_SYS.N_SENS] = sensitivities

        fid.seek(KIT_SYS.SAMPLE_INFO)
        acqcond_offset = unpack('i', fid.read(KIT_SYS.INT))[0]
        fid.seek(acqcond_offset)
        acq_type = unpack('i', fid.read(KIT_SYS.INT))[0]
        if acq_type == 1:
            sqd['sfreq'] = unpack('d', fid.read(KIT_SYS.DOUBLE))[0]
            _ = fid.read(KIT_SYS.INT)  # initialized estimate of samples
            sqd['nsamples'] = unpack('i', fid.read(KIT_SYS.INT))[0]
        else:
            err = ("You are probably trying to load a file that is not a "
                   "continuous recording sqd file.")
            raise ValueError(err)
        sqd['n_sens'] = KIT_SYS.N_SENS
        sqd['nmegchan'] = KIT_SYS.NMEGCHAN
        sqd['nmiscchan'] = KIT_SYS.NMISCCHAN
        sqd['DYNAMIC_RANGE'] = KIT_SYS.DYNAMIC_RANGE
    return sqd


def read_raw_kit(input_fname, mrk=None, elp=None, hsp=None, stim='>',
                 slope='-', stimthresh=1, preload=False, verbose=None):
    """Reader function for KIT conversion to FIF

    Parameters
    ----------
    input_fname : str
        Path to the sqd file.
    mrk : None | str | array_like, shape = (5, 3) | list of str or array_like
        Marker points representing the location of the marker coils with
        respect to the MEG Sensors, or path to a marker file.
        If list, all of the markers will be averaged together.
    elp : None | str | array_like, shape = (8, 3)
        Digitizer points representing the location of the fiducials and the
        marker coils with respect to the digitized head shape, or path to a
        file containing these points.
    hsp : None | str | array, shape = (n_points, 3)
        Digitizer head shape points, or path to head shape file. If more than
        10`000 points are in the head shape, they are automatically decimated.
    stim : list of int | '<' | '>'
        Channel-value correspondence when converting KIT trigger channels to a
        Neuromag-style stim channel. For '<', the largest values are assigned
        to the first channel (default). For '>', the largest values are
        assigned to the last channel. Can also be specified as a list of
        trigger channel indexes.
    slope : '+' | '-'
        How to interpret values on KIT trigger channels when synthesizing a
        Neuromag-style stim channel. With '+', a positive slope (low-to-high)
        is interpreted as an event. With '-', a negative slope (high-to-low)
        is interpreted as an event.
    stimthresh : float
        The threshold level for accepting voltage changes in KIT trigger
        channels as a trigger event.
    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    return RawKIT(input_fname=input_fname, mrk=mrk, elp=elp, hsp=hsp,
                  stim=stim, slope=slope, stimthresh=stimthresh,
                  preload=preload, verbose=verbose)
