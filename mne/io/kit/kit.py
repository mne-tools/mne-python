"""Conversion tool from SQD to FIF

RawKIT class is adapted from Denis Engemann et al.'s mne_bti2fiff.py

"""

# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

from os import SEEK_CUR, path as op
from struct import unpack
import time

import numpy as np
from scipy import linalg

from ..pick import pick_types
from ...coreg import fit_matched_points, _decimate_points
from ...utils import verbose, logger
from ...transforms import (apply_trans, als_ras_trans, als_ras_trans_mm,
                           get_ras_to_neuromag_trans)
from ..base import _BaseRaw
from ...epochs import EpochsArray
from ..constants import FIFF
from ..meas_info import _empty_info, _read_dig_points, _make_dig_points
from ..tag import _loc_to_trans
from .constants import KIT, KIT_NY, KIT_AD
from .coreg import read_mrk
from ...externals.six import string_types
from ...event import read_events


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
        channels as a trigger event. If None, stim must also be set to None.
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
        input_fname = op.abspath(input_fname)
        self.preload = False
        logger.info('Creating Raw.info structure...')
        info, self._kit_info = get_kit_info(input_fname)
        self._kit_info['slope'] = slope
        self._kit_info['stimthresh'] = stimthresh
        self._kit_info['fname'] = input_fname
        if self._kit_info['acq_type'] != 1:
            err = 'SQD file contains epochs, not raw data. Wrong reader.'
            raise TypeError(err)
        logger.info('Creating Info structure...')

        last_samps = [self._kit_info['n_samples'] - 1]
        super(RawKIT, self).__init__(
            info, last_samps=last_samps, verbose=verbose)
        self._set_stimchannels(stim)

        if isinstance(mrk, list):
            mrk = [read_mrk(marker) if isinstance(marker, string_types)
                   else marker for marker in mrk]
            mrk = np.mean(mrk, axis=0)
        if (mrk is not None and elp is not None and hsp is not None):
            dig_points, dev_head_t = _set_dig_kit(mrk, elp, hsp)
            self.info['dig'] = dig_points
            self.info['dev_head_t'] = dev_head_t
        elif (mrk is not None or elp is not None or hsp is not None):
            err = ("mrk, elp and hsp need to be provided as a group (all or "
                   "none)")
            raise ValueError(err)

        if preload:
            self.preload = preload
            logger.info('Reading raw data from %s...' % input_fname)
            self._data, _ = self._read_segment()
            assert self._data.shape == (self.info['nchan'], self.last_samp + 1)

            logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs'
                        % (self.first_samp, self.last_samp,
                           self.times[0], self.times[-1]))
        logger.info('Ready.')

    def __repr__(self):
        s = ('%r' % op.basename(self._kit_info['fname']),
             "n_channels x n_times : %s x %s" % (len(self.info['ch_names']),
                                                 self._kit_info['n_samples']))
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

    def _set_stimchannels(self, stim='<'):
        """Specify how the trigger channel is synthesized from analog channels.

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
        """
        if stim is not None:
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
            elif np.max(stim) >= self._kit_info['nchan']:
                msg = ("Tried to set stim channel %i, but sqd file only has %i"
                       " channels" % (np.max(stim), self._kit_info['nchan']))
                raise ValueError(msg)
            # modify info
            self.info['nchan'] = self._kit_info['nchan'] + 1
            ch_name = 'STI 014'
            chan_info = {}
            chan_info['cal'] = KIT.CALIB_FACTOR
            chan_info['logno'] = self.info['nchan']
            chan_info['scanno'] = self.info['nchan']
            chan_info['range'] = 1.0
            chan_info['unit'] = FIFF.FIFF_UNIT_NONE
            chan_info['unit_mul'] = 0
            chan_info['ch_name'] = ch_name
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['loc'] = np.zeros(12)
            chan_info['kind'] = FIFF.FIFFV_STIM_CH
            self.info['chs'].append(chan_info)
            self.info['ch_names'].append(ch_name)
        if self.preload:
            err = "Can't change stim channel after preloading data"
            raise NotImplementedError(err)

        self._kit_info['stim'] = stim

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

        with open(self._kit_info['fname'], 'rb', buffering=0) as fid:
            # extract data
            data_offset = KIT.RAW_OFFSET
            fid.seek(data_offset)
            # data offset info
            data_offset = unpack('i', fid.read(KIT.INT))[0]
            nchan = self._kit_info['nchan']
            buffer_size = stop - start
            count = buffer_size * nchan
            pointer = start * nchan * KIT.SHORT
            fid.seek(data_offset + pointer)
            data = np.fromfile(fid, dtype='h', count=count)
            data = data.reshape((buffer_size, nchan))
        # amplifier applies only to the sensor channels
        n_sens = self._kit_info['n_sens']
        sensor_gain = np.copy(self._kit_info['sensor_gain'])
        sensor_gain[:n_sens] = (sensor_gain[:n_sens] /
                                self._kit_info['amp_gain'])
        conv_factor = np.array((KIT.VOLTAGE_RANGE /
                                self._kit_info['DYNAMIC_RANGE']) *
                               sensor_gain, ndmin=2)
        data = conv_factor * data
        data = data.T

        # Create a synthetic channel
        if self._kit_info['stim'] is not None:
            trig_chs = data[self._kit_info['stim'], :]
            if self._kit_info['slope'] == '+':
                trig_chs = trig_chs > self._kit_info['stimthresh']
            elif self._kit_info['slope'] == '-':
                trig_chs = trig_chs < self._kit_info['stimthresh']
            else:
                raise ValueError("slope needs to be '+' or '-'")
            trig_vals = np.array(2 ** np.arange(len(self._kit_info['stim'])),
                                 ndmin=2).T
            trig_chs = trig_chs * trig_vals
            stim_ch = np.array(trig_chs.sum(axis=0), ndmin=2)
            data = np.vstack((data, stim_ch))
        data = data[sel]

        logger.info('[done]')
        times = np.arange(start, stop, dtype=float) / self.info['sfreq']

        return data, times


class EpochsKIT(EpochsArray):
    """Epochs Array object from KIT SQD file

    Parameters
    ----------
    input_fname : str
        Path to the sqd file.
    events : str | array, shape (n_events, 3)
        Path to events file. If array, it is the events typically returned
        by the read_events function. If some events don't match the events
        of interest as specified by event_id,they will be marked as 'IGNORED'
        in the drop log.
    event_id : int | list of int | dict | None
        The id of the event to consider. If dict,
        the keys can later be used to acces associated events. Example:
        dict(auditory=1, visual=3). If int, a dict will be created with
        the id as string. If a list, all events with the IDs specified
        in the list are used. If None, all events will be used with
        and a dict is created with string integer names corresponding
        to the event id integers.
    tmin : float
        Start time before event.
    baseline : None or tuple of length 2 (default (None, 0))
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used.
        The baseline (a, b) includes both endpoints, i.e. all
        timepoints t such that a <= t <= b.
    reject : dict | None
        Rejection parameters based on peak-to-peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # uV (EEG channels)
                          eog=250e-6 # uV (EOG channels)
                          )
    flat : dict | None
        Rejection parameters based on flatness of signal.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg', and values
        are floats that set the minimum acceptable peak-to-peak amplitude.
        If flat is None then no rejection is done.
    reject_tmin : scalar | None
        Start of the time window used to reject epochs (with the default None,
        the window will start with tmin).
    reject_tmax : scalar | None
        End of the time window used to reject epochs (with the default None,
        the window will end with tmax).
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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.Epochs : Documentation of attribute and methods.
    """
    @verbose
    def __init__(self, input_fname, events, event_id=None, tmin=0,
                 baseline=None,  reject=None, flat=None, reject_tmin=None,
                 reject_tmax=None, mrk=None, elp=None, hsp=None, verbose=None):

        if isinstance(events, string_types):
            events = read_events(events)
        if isinstance(mrk, list):
            mrk = [read_mrk(marker) if isinstance(marker, string_types)
                   else marker for marker in mrk]
            mrk = np.mean(mrk, axis=0)

        if (mrk is not None and elp is not None and hsp is not None):
            dig_points, dev_head_t = _set_dig_kit(mrk, elp, hsp)
            self.info['dig'] = dig_points
            self.info['dev_head_t'] = dev_head_t
        elif (mrk is not None or elp is not None or hsp is not None):
            err = ("mrk, elp and hsp need to be provided as a group (all or "
                   "none)")
            raise ValueError(err)

        logger.info('Extracting KIT Parameters from %s...' % input_fname)
        input_fname = op.abspath(input_fname)
        self.info, self._kit_info = get_kit_info(input_fname)
        if len(events) != self._kit_info['n_epochs']:
            raise ValueError('Event list does not match number of epochs.')

        self._kit_info['fname'] = input_fname
        if self._kit_info['acq_type'] == 3:
            self._kit_info['data_offset'] = KIT.RAW_OFFSET
            self._kit_info['data_length'] = KIT.INT
            self._kit_info['dtype'] = 'h'
        else:
            err = ('SQD file contains raw data, not epochs or average. '
                   'Wrong reader.')
            raise TypeError(err)

        if event_id is None:  # convert to int to make typing-checks happy
            event_id = dict((str(e), int(e)) for e in np.unique(events[:, 2]))

        for key, val in event_id.items():
            if val not in events[:, 2]:
                msg = ('No matching events found for %s '
                       '(event id %i)' % (key, val))
                raise ValueError(msg)

        data = self._read_data()
        assert data.shape == (self._kit_info['n_epochs'], self.info['nchan'],
                              self._kit_info['frame_length'])

        super(EpochsKIT, self).__init__(data=data, info=self.info,
                                        events=events, event_id=event_id,
                                        baseline=baseline, tmin=tmin,
                                        reject=reject, flat=flat,
                                        reject_tmin=reject_tmin,
                                        reject_tmax=reject_tmax,
                                        verbose=verbose)
        logger.info('Ready.')

    def _read_data(self):
        """Read epochs data

        Returns
        -------
        data : array, [channels x samples]
           the data matrix (channels x samples).
        times : array, [samples]
            returns the time values corresponding to the samples.
        """
        #  Initial checks
        epoch_length = self._kit_info['frame_length']
        n_epochs = self._kit_info['n_epochs']
        n_samples = self._kit_info['n_samples']

        with open(self._kit_info['fname'], 'rb', buffering=0) as fid:
            # extract data
            data_offset = self._kit_info['data_offset']
            dtype = self._kit_info['dtype']
            fid.seek(data_offset)
            # data offset info
            data_offset = unpack('i', fid.read(KIT.INT))[0]
            nchan = self._kit_info['nchan']
            count = n_samples * nchan
            fid.seek(data_offset)
            data = np.fromfile(fid, dtype=dtype, count=count)
            data = data.reshape((n_samples, nchan))
        # amplifier applies only to the sensor channels
        n_sens = self._kit_info['n_sens']
        sensor_gain = np.copy(self._kit_info['sensor_gain'])
        sensor_gain[:n_sens] = (sensor_gain[:n_sens] /
                                self._kit_info['amp_gain'])
        conv_factor = np.array((KIT.VOLTAGE_RANGE /
                                self._kit_info['DYNAMIC_RANGE']) *
                               sensor_gain, ndmin=2)
        data = conv_factor * data
        # reshape
        data = data.T
        data = data.reshape((nchan, n_epochs, epoch_length))
        data = data.transpose((1, 0, 2))

        return data

    def __repr__(self):
        s = ('%r ' % op.basename(self._kit_info['fname']),
             "n_epochs x n_channels x n_times : %s x %s x %s"
             % (self._kit_info['n_epochs'], self.info['nchan'],
                self._kit_info['frame_length']))
        return "<EpochsKIT  |  %s>" % ', '.join(s)


def _set_dig_kit(mrk, elp, hsp, auto_decimate=True):
    """Add landmark points and head shape data to the KIT instance

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

    Returns
    -------
    dig_points : list
        List of digitizer points for info['dig'].
    dev_head_t : dict
        A dictionary describe the device-head transformation.
    """
    if isinstance(hsp, string_types):
        hsp = _read_dig_points(hsp)
    n_pts = len(hsp)
    if n_pts > KIT.DIG_POINTS:
        hsp = _decimate_points(hsp, decim=5)
        n_new = len(hsp)
        msg = ("The selected head shape contained {n_in} points, which is "
               "more than recommended ({n_rec}), and was automatically "
               "downsampled to {n_new} points. The preferred way to "
               "downsample is using FastScan.")
        msg = msg.format(n_in=n_pts, n_rec=KIT.DIG_POINTS, n_new=n_new)
        logger.warning(msg)

    if isinstance(elp, string_types):
        elp_points = _read_dig_points(elp)
        if len(elp_points) != 8:
            err = ("File %r should contain 8 points; got shape "
                   "%s." % (elp, elp_points.shape))
            raise ValueError(err)
        elp = elp_points

    elif len(elp) != 8:
        err = ("ELP should contain 8 points; got shape "
               "%s." % (elp.shape,))
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

    nasion, lpa, rpa = elp[:3]
    elp = elp[3:]

    dig_points = _make_dig_points(nasion, lpa, rpa, elp, hsp)
    dev_head_t = {'from': FIFF.FIFFV_COORD_DEVICE, 'to': FIFF.FIFFV_COORD_HEAD,
                  'trans': trans}

    return dig_points, dev_head_t


def get_kit_info(rawfile):
    """Extracts all the information from the sqd file.

    Parameters
    ----------
    rawfile : str
        KIT file to be read.

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

        gain1 = KIT_SYS.GAINS[(KIT_SYS.GAIN1_MASK & amp_data) >>
                              KIT_SYS.GAIN1_BIT]
        gain2 = KIT_SYS.GAINS[(KIT_SYS.GAIN2_MASK & amp_data) >>
                              KIT_SYS.GAIN2_BIT]
        if KIT_SYS.GAIN3_BIT:
            gain3 = KIT_SYS.GAINS[(KIT_SYS.GAIN3_MASK & amp_data) >>
                                  KIT_SYS.GAIN3_BIT]
            sqd['amp_gain'] = gain1 * gain2 * gain3
        else:
            sqd['amp_gain'] = gain1 * gain2

        # filter settings
        sqd['lowpass'] = KIT_SYS.LPFS[(KIT_SYS.LPF_MASK & amp_data) >>
                                      KIT_SYS.LPF_BIT]
        sqd['highpass'] = KIT_SYS.HPFS[(KIT_SYS.HPF_MASK & amp_data) >>
                                       KIT_SYS.HPF_BIT]
        sqd['notch'] = KIT_SYS.BEFS[(KIT_SYS.BEF_MASK & amp_data) >>
                                    KIT_SYS.BEF_BIT]

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
        sqd['sfreq'] = unpack('d', fid.read(KIT_SYS.DOUBLE))[0]
        if acq_type == 1:
            fid.read(KIT_SYS.INT)  # initialized estimate of samples
            sqd['n_samples'] = unpack('i', fid.read(KIT_SYS.INT))[0]
        elif acq_type == 2 or acq_type == 3:
            sqd['frame_length'] = unpack('i', fid.read(KIT_SYS.INT))[0]
            sqd['pretrigger_length'] = unpack('i', fid.read(KIT_SYS.INT))[0]
            sqd['average_count'] = unpack('i', fid.read(KIT_SYS.INT))[0]
            sqd['n_epochs'] = unpack('i', fid.read(KIT_SYS.INT))[0]
            sqd['n_samples'] = sqd['frame_length'] * sqd['n_epochs']
        else:
            err = ("Your file is neither continuous nor epoched data. "
                   "What type of file is it?!")
            raise TypeError(err)
        sqd['n_sens'] = KIT_SYS.N_SENS
        sqd['nmegchan'] = KIT_SYS.NMEGCHAN
        sqd['nmiscchan'] = KIT_SYS.NMISCCHAN
        sqd['DYNAMIC_RANGE'] = KIT_SYS.DYNAMIC_RANGE
        sqd['acq_type'] = acq_type

        # Create raw.info dict for raw fif object with SQD data
        info = _empty_info()
        info['events'] = []
        info['meas_id'] = None
        info['file_id'] = None
        info['meas_date'] = int(time.time())
        info['projs'] = []
        info['comps'] = []
        info['lowpass'] = sqd['lowpass']
        info['highpass'] = sqd['highpass']
        info['sfreq'] = float(sqd['sfreq'])
        info['bads'] = []
        info['acq_pars'], info['acq_stim'] = None, None
        info['filename'] = None
        info['ctf_head_t'] = None
        info['dev_ctf_t'] = []
        info['nchan'] = sqd['nchan']
        info['dig'] = None
        info['dev_head_t'] = None

        # Creates a list of dicts of meg channels for raw.info
        logger.info('Setting channel info structure...')
        ch_names = {}
        ch_names['MEG'] = ['MEG %03d' % ch for ch
                           in range(1, sqd['n_sens'] + 1)]
        ch_names['MISC'] = ['MISC %03d' % ch for ch
                            in range(1, sqd['nmiscchan'] + 1)]
        locs = sqd['sensor_locs']
        chan_locs = apply_trans(als_ras_trans, locs[:, :3])
        chan_angles = locs[:, 3:]
        info['chs'] = []
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
            if idx <= sqd['nmegchan']:
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
            info['chs'].append(chan_info)

        # label trigger and misc channels
        for idy, ch_name in enumerate(ch_names['MISC'],
                                      sqd['n_sens'] + 1):
            chan_info = {}
            chan_info['cal'] = KIT.CALIB_FACTOR
            chan_info['logno'] = idy
            chan_info['scanno'] = idy
            chan_info['range'] = 1.0
            chan_info['unit'] = FIFF.FIFF_UNIT_V
            chan_info['unit_mul'] = 0
            chan_info['ch_name'] = ch_name
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['loc'] = np.zeros(12)
            chan_info['kind'] = FIFF.FIFFV_MISC_CH
            info['chs'].append(chan_info)

        info['ch_names'] = ch_names['MEG'] + ch_names['MISC']

    return info, sqd


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

    Returns
    -------
    raw : Instance of RawKIT
        A Raw object containing KIT data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawKIT(input_fname=input_fname, mrk=mrk, elp=elp, hsp=hsp,
                  stim=stim, slope=slope, stimthresh=stimthresh,
                  preload=preload, verbose=verbose)


def read_epochs_kit(input_fname, events, event_id=None,
                    mrk=None, elp=None, hsp=None, verbose=None):
    """Reader function for KIT epochs files

    Parameters
    ----------
    input_fname : str
        Path to the sqd file.
    events : array, shape (n_events, 3)
        The events typically returned by the read_events function.
        If some events don't match the events of interest as specified
        by event_id, they will be marked as 'IGNORED' in the drop log.
    event_id : int | list of int | dict | None
        The id of the event to consider. If dict,
        the keys can later be used to acces associated events. Example:
        dict(auditory=1, visual=3). If int, a dict will be created with
        the id as string. If a list, all events with the IDs specified
        in the list are used. If None, all events will be used with
        and a dict is created with string integer names corresponding
        to the event id integers.
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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    epochs : instance of Epochs
        The epochs.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    epochs = EpochsKIT(input_fname=input_fname, events=events,
                       event_id=event_id, mrk=mrk, elp=elp, hsp=hsp,
                       verbose=verbose)
    return epochs
