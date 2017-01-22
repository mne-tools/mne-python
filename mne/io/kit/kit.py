"""Conversion tool from SQD to FIF.

RawKIT class is adapted from Denis Engemann et al.'s mne_bti2fiff.py.
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
from ...utils import verbose, logger, warn
from ...transforms import (apply_trans, als_ras_trans,
                           get_ras_to_neuromag_trans, Transform)
from ..base import BaseRaw
from ..utils import _mult_cal_one
from ...epochs import BaseEpochs
from ..constants import FIFF
from ..meas_info import _empty_info, _read_dig_points, _make_dig_points
from .constants import KIT, KIT_CONSTANTS, SYSNAMES
from .coreg import read_mrk
from ...externals.six import string_types
from ...event import read_events


class RawKIT(BaseRaw):
    """Raw object from KIT SQD file.

    Parameters
    ----------
    input_fname : str
        Path to the sqd file.
    mrk : None | str | array_like, shape (5, 3) | list of str or array_like
        Marker points representing the location of the marker coils with
        respect to the MEG Sensors, or path to a marker file.
        If list, all of the markers will be averaged together.
    elp : None | str | array_like, shape (8, 3)
        Digitizer points representing the location of the fiducials and the
        marker coils with respect to the digitized head shape, or path to a
        file containing these points.
    hsp : None | str | array, shape (n_points, 3)
        Digitizer head shape points, or path to head shape file. If more than
        10,000 points are in the head shape, they are automatically decimated.
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
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    stim_code : 'binary' | 'channel'
        How to decode trigger values from stim channels. 'binary' read stim
        channel events as binary code, 'channel' encodes channel number.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Notes
    -----
    ``elp`` and ``hsp`` are usually the exported text files (*.txt) from the
    Polhemus FastScan system. hsp refers to the headshape surface points. elp
    refers to the points in head-space that corresponds to the HPI points.
    Currently, '*.elp' and '*.hsp' files are NOT supported.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, input_fname, mrk=None, elp=None, hsp=None, stim='>',
                 slope='-', stimthresh=1, preload=False, stim_code='binary',
                 verbose=None):  # noqa: D102
        logger.info('Extracting SQD Parameters from %s...' % input_fname)
        input_fname = op.abspath(input_fname)
        self.preload = False
        logger.info('Creating Raw.info structure...')
        info, kit_info = get_kit_info(input_fname)
        kit_info['slope'] = slope
        kit_info['stimthresh'] = stimthresh
        if kit_info['acq_type'] != 1:
            err = 'SQD file contains epochs, not raw data. Wrong reader.'
            raise TypeError(err)
        logger.info('Creating Info structure...')

        last_samps = [kit_info['n_samples'] - 1]
        self._raw_extras = [kit_info]
        self._set_stimchannels(info, stim, stim_code)
        super(RawKIT, self).__init__(
            info, preload, last_samps=last_samps, filenames=[input_fname],
            raw_extras=self._raw_extras, verbose=verbose)

        if isinstance(mrk, list):
            mrk = [read_mrk(marker) if isinstance(marker, string_types)
                   else marker for marker in mrk]
            mrk = np.mean(mrk, axis=0)
        if mrk is not None and elp is not None and hsp is not None:
            dig_points, dev_head_t = _set_dig_kit(mrk, elp, hsp)
            self.info['dig'] = dig_points
            self.info['dev_head_t'] = dev_head_t
        elif mrk is not None or elp is not None or hsp is not None:
            raise ValueError('mrk, elp and hsp need to be provided as a group '
                             '(all or none)')

        logger.info('Ready.')

    def read_stim_ch(self, buffer_size=1e5):
        """Read events from data.

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
            x = self[pick, b_start:b_stop][0]
            stim_ch[:, b_start:b_start + x.shape[1]] = x

        return stim_ch

    def _set_stimchannels(self, info, stim, stim_code):
        """Specify how the trigger channel is synthesized from analog channels.

        Has to be done before loading data. For a RawKIT instance that has been
        created with preload=True, this method will raise a
        NotImplementedError.

        Parameters
        ----------
        info : instance of MeasInfo
            The measurement info.
        stim : list of int | '<' | '>'
            Can be submitted as list of trigger channels.
            If a list is not specified, the default triggers extracted from
            misc channels will be used with specified directionality.
            '<' means that largest values assigned to the first channel
            in sequence.
            '>' means the largest trigger assigned to the last channel
            in sequence.
        stim_code : 'binary' | 'channel'
            How to decode trigger values from stim channels. 'binary' read stim
            channel events as binary code, 'channel' encodes channel number.
        """
        if stim_code not in ('binary', 'channel'):
            raise ValueError("stim_code=%r, needs to be 'binary' or 'channel'"
                             % stim_code)

        if stim is not None:
            if isinstance(stim, str):
                picks = _default_stim_chs(info)
                if stim == '<':
                    stim = picks[::-1]
                elif stim == '>':
                    stim = picks
                else:
                    raise ValueError("stim needs to be list of int, '>' or "
                                     "'<', not %r" % str(stim))
            else:
                stim = np.asarray(stim, int)
                if stim.max() >= self._raw_extras[0]['nchan']:
                    raise ValueError(
                        'Got stim=%s, but sqd file only has %i channels' %
                        (stim, self._raw_extras[0]['nchan']))

            # modify info
            nchan = self._raw_extras[0]['nchan'] + 1
            ch_name = 'STI 014'
            chan_info = {}
            chan_info['cal'] = KIT.CALIB_FACTOR
            chan_info['logno'] = nchan
            chan_info['scanno'] = nchan
            chan_info['range'] = 1.0
            chan_info['unit'] = FIFF.FIFF_UNIT_NONE
            chan_info['unit_mul'] = 0
            chan_info['ch_name'] = ch_name
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['loc'] = np.zeros(12)
            chan_info['kind'] = FIFF.FIFFV_STIM_CH
            info['chs'].append(chan_info)
            info._update_redundant()
        if self.preload:
            err = "Can't change stim channel after preloading data"
            raise NotImplementedError(err)

        self._raw_extras[0]['stim'] = stim
        self._raw_extras[0]['stim_code'] = stim_code

    @verbose
    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        nchan = self._raw_extras[fi]['nchan']
        data_left = (stop - start) * nchan
        # amplifier applies only to the sensor channels
        n_sens = self._raw_extras[fi]['n_sens']
        sensor_gain = self._raw_extras[fi]['sensor_gain'].copy()
        sensor_gain[:n_sens] = (sensor_gain[:n_sens] /
                                self._raw_extras[fi]['amp_gain'])
        conv_factor = np.array((KIT.VOLTAGE_RANGE /
                                self._raw_extras[fi]['DYNAMIC_RANGE']) *
                               sensor_gain)
        n_bytes = 2
        # Read up to 100 MB of data at a time.
        blk_size = min(data_left, (100000000 // n_bytes // nchan) * nchan)
        with open(self._filenames[fi], 'rb', buffering=0) as fid:
            # extract data
            data_offset = KIT.RAW_OFFSET
            fid.seek(data_offset)
            # data offset info
            data_offset = unpack('i', fid.read(KIT.INT))[0]
            pointer = start * nchan * KIT.SHORT
            fid.seek(data_offset + pointer)
            stim = self._raw_extras[fi]['stim']
            for blk_start in np.arange(0, data_left, blk_size) // nchan:
                blk_size = min(blk_size, data_left - blk_start * nchan)
                block = np.fromfile(fid, dtype='h', count=blk_size)
                block = block.reshape(nchan, -1, order='F').astype(float)
                blk_stop = blk_start + block.shape[1]
                data_view = data[:, blk_start:blk_stop]
                block *= conv_factor[:, np.newaxis]

                # Create a synthetic stim channel
                if stim is not None:
                    params = self._raw_extras[fi]
                    stim_ch = _make_stim_channel(block[stim, :],
                                                 params['slope'],
                                                 params['stimthresh'],
                                                 params['stim_code'], stim)
                    block = np.vstack((block, stim_ch))

                _mult_cal_one(data_view, block, idx, None, mult)
        # cals are all unity, so can be ignored


def _default_stim_chs(info):
    """Default stim channels for SQD files."""
    return pick_types(info, meg=False, ref_meg=False, misc=True,
                      exclude=[])[:8]


def _make_stim_channel(trigger_chs, slope, threshold, stim_code,
                       trigger_values):
    """Create synthetic stim channel from multiple trigger channels."""
    if slope == '+':
        trig_chs_bin = trigger_chs > threshold
    elif slope == '-':
        trig_chs_bin = trigger_chs < threshold
    else:
        raise ValueError("slope needs to be '+' or '-'")
    # trigger value
    if stim_code == 'binary':
        trigger_values = 2 ** np.arange(len(trigger_chs))
    elif stim_code != 'channel':
        raise ValueError("stim_code must be 'binary' or 'channel', got %s" %
                         repr(stim_code))
    trig_chs = trig_chs_bin * trigger_values[:, np.newaxis]
    return np.array(trig_chs.sum(axis=0), ndmin=2)


class EpochsKIT(BaseEpochs):
    """Epochs Array object from KIT SQD file.

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
        the keys can later be used to access associated events. Example:
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
                          eeg=40e-6, # V (EEG channels)
                          eog=250e-6 # V (EOG channels)
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
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Notes
    -----
    ``elp`` and ``hsp`` are usually the exported text files (*.txt) from the
    Polhemus FastScan system. hsp refers to the headshape surface points. elp
    refers to the points in head-space that corresponds to the HPI points.
    Currently, '*.elp' and '*.hsp' files are NOT supported.

    See Also
    --------
    mne.Epochs : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, input_fname, events, event_id=None, tmin=0,
                 baseline=None,  reject=None, flat=None, reject_tmin=None,
                 reject_tmax=None, mrk=None, elp=None, hsp=None,
                 verbose=None):  # noqa: D102

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
        self.info, kit_info = get_kit_info(input_fname)
        kit_info.update(filename=input_fname)
        self._raw_extras = [kit_info]
        self._filenames = []
        if len(events) != self._raw_extras[0]['n_epochs']:
            raise ValueError('Event list does not match number of epochs.')

        if self._raw_extras[0]['acq_type'] == 3:
            self._raw_extras[0]['data_offset'] = KIT.RAW_OFFSET
            self._raw_extras[0]['data_length'] = KIT.INT
            self._raw_extras[0]['dtype'] = 'h'
        else:
            err = ('SQD file contains raw data, not epochs or average. '
                   'Wrong reader.')
            raise TypeError(err)

        if event_id is None:  # convert to int to make typing-checks happy
            event_id = dict((str(e), int(e)) for e in np.unique(events[:, 2]))

        for key, val in event_id.items():
            if val not in events[:, 2]:
                raise ValueError('No matching events found for %s '
                                 '(event id %i)' % (key, val))

        data = self._read_kit_data()
        assert data.shape == (self._raw_extras[0]['n_epochs'],
                              self.info['nchan'],
                              self._raw_extras[0]['frame_length'])
        tmax = ((data.shape[2] - 1) / self.info['sfreq']) + tmin
        super(EpochsKIT, self).__init__(
            self.info, data, events, event_id, tmin, tmax, baseline,
            reject=reject, flat=flat, reject_tmin=reject_tmin,
            reject_tmax=reject_tmax, filename=input_fname, verbose=verbose)
        logger.info('Ready.')

    def _read_kit_data(self):
        """Read epochs data.

        Returns
        -------
        data : array, [channels x samples]
           the data matrix (channels x samples).
        times : array, [samples]
            returns the time values corresponding to the samples.
        """
        #  Initial checks
        epoch_length = self._raw_extras[0]['frame_length']
        n_epochs = self._raw_extras[0]['n_epochs']
        n_samples = self._raw_extras[0]['n_samples']
        filename = self._raw_extras[0]['filename']

        with open(filename, 'rb', buffering=0) as fid:
            # extract data
            data_offset = self._raw_extras[0]['data_offset']
            dtype = self._raw_extras[0]['dtype']
            fid.seek(data_offset)
            # data offset info
            data_offset = unpack('i', fid.read(KIT.INT))[0]
            nchan = self._raw_extras[0]['nchan']
            count = n_samples * nchan
            fid.seek(data_offset)
            data = np.fromfile(fid, dtype=dtype, count=count)
            data = data.reshape((n_samples, nchan))
        # amplifier applies only to the sensor channels
        n_sens = self._raw_extras[0]['n_sens']
        sensor_gain = np.copy(self._raw_extras[0]['sensor_gain'])
        sensor_gain[:n_sens] = (sensor_gain[:n_sens] /
                                self._raw_extras[0]['amp_gain'])
        conv_factor = np.array((KIT.VOLTAGE_RANGE /
                                self._raw_extras[0]['DYNAMIC_RANGE']) *
                               sensor_gain, ndmin=2)
        data = conv_factor * data
        # reshape
        data = data.T
        data = data.reshape((nchan, n_epochs, epoch_length))
        data = data.transpose((1, 0, 2))

        return data


def _set_dig_kit(mrk, elp, hsp):
    """Add landmark points and head shape data to the KIT instance.

    Digitizer data (elp and hsp) are represented in [mm] in the Polhemus
    ALS coordinate system. This is converted to [m].

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
        hsp = _decimate_points(hsp, res=0.005)
        n_new = len(hsp)
        warn("The selected head shape contained {n_in} points, which is "
             "more than recommended ({n_rec}), and was automatically "
             "downsampled to {n_new} points. The preferred way to "
             "downsample is using FastScan.".format(
                 n_in=n_pts, n_rec=KIT.DIG_POINTS, n_new=n_new))

    if isinstance(elp, string_types):
        elp_points = _read_dig_points(elp)
        if len(elp_points) != 8:
            raise ValueError("File %r should contain 8 points; got shape "
                             "%s." % (elp, elp_points.shape))
        elp = elp_points
    elif len(elp) != 8:
        raise ValueError("ELP should contain 8 points; got shape "
                         "%s." % (elp.shape,))
    if isinstance(mrk, string_types):
        mrk = read_mrk(mrk)

    hsp = apply_trans(als_ras_trans, hsp)
    elp = apply_trans(als_ras_trans, elp)
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
    dev_head_t = Transform('meg', 'head', trans)

    return dig_points, dev_head_t


def get_kit_info(rawfile):
    """Extract all the information from the sqd file.

    Parameters
    ----------
    rawfile : str
        KIT file to be read.

    Returns
    -------
    info : instance of Info
        An Info for the instance.
    sqd : dict
        A dict containing all the sqd parameter settings.
    """
    sqd = dict()
    sqd['rawfile'] = rawfile
    with open(rawfile, 'rb', buffering=0) as fid:  # buffering=0 for np bug
        fid.seek(KIT.BASIC_INFO)
        basic_offset = unpack('i', fid.read(KIT.INT))[0]
        fid.seek(basic_offset)
        # skips version, revision
        fid.seek(KIT.INT * 2, SEEK_CUR)
        sysid = unpack('i', fid.read(KIT.INT))[0]
        # basic info
        sysname = unpack('128s', fid.read(KIT.STRING))
        sysname = sysname[0].decode().split('\n')[0]
        if sysid not in KIT_CONSTANTS:
            raise NotImplementedError("Data from the KIT system %s (ID %s) "
                                      "can not currently be read, please "
                                      "contact the MNE-Python developers."
                                      % (sysname, sysid))
        KIT_SYS = KIT_CONSTANTS[sysid]
        logger.info("KIT-System ID %i: %s" % (sysid, sysname))
        if sysid in SYSNAMES:
            if sysname != SYSNAMES[sysid]:
                warn("KIT file %s has system-name %r, expected %r"
                     % (rawfile, sysname, SYSNAMES[sysid]))

        # channels
        fid.seek(KIT.STRING, SEEK_CUR)  # skips modelname
        sqd['nchan'] = unpack('i', fid.read(KIT.INT))[0]
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
            elif sens_type in (257, 0):
                # reference channels
                sensors.append(np.zeros(7))
                sqd['i'] = sens_type
            else:
                raise IOError("Unknown KIT channel type: %i" % sens_type)
        sqd['sensor_locs'] = np.array(sensors)
        if len(sqd['sensor_locs']) != KIT_SYS.N_SENS:
            raise IOError("An error occurred while reading %s" % rawfile)

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
        sens.shape = (sqd['nchan'], 2)
        sqd['sensor_gain'] = np.ones(KIT_SYS.NCHAN)
        sqd['sensor_gain'][:KIT_SYS.N_SENS] = sens[:KIT_SYS.N_SENS, 1]

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
        info = _empty_info(float(sqd['sfreq']))
        info.update(meas_date=int(time.time()), lowpass=sqd['lowpass'],
                    highpass=sqd['highpass'], buffer_size_sec=1.,
                    kit_system_id=sysid)

        # Creates a list of dicts of meg channels for raw.info
        logger.info('Setting channel info structure...')
        locs = sqd['sensor_locs']
        chan_locs = apply_trans(als_ras_trans, locs[:, :3])
        chan_angles = locs[:, 3:]
        for idx, (ch_loc, ch_angles) in enumerate(zip(chan_locs, chan_angles),
                                                  1):
            chan_info = {'cal': KIT.CALIB_FACTOR,
                         'logno': idx,
                         'scanno': idx,
                         'range': KIT.RANGE,
                         'unit_mul': KIT.UNIT_MUL,
                         'ch_name': 'MEG %03d' % idx,
                         'unit': FIFF.FIFF_UNIT_T,
                         'coord_frame': FIFF.FIFFV_COORD_DEVICE}
            if idx <= sqd['nmegchan']:
                chan_info['coil_type'] = FIFF.FIFFV_COIL_KIT_GRAD
                chan_info['kind'] = FIFF.FIFFV_MEG_CH
            else:
                chan_info['coil_type'] = FIFF.FIFFV_COIL_KIT_REF_MAG
                chan_info['kind'] = FIFF.FIFFV_REF_MEG_CH

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
            info['chs'].append(chan_info)

        # label trigger and misc channels
        for idx in range(1, sqd['nmiscchan'] + 1):
            ch_idx = idx + KIT_SYS.N_SENS
            chan_info = {'cal': KIT.CALIB_FACTOR,
                         'logno': ch_idx,
                         'scanno': ch_idx,
                         'range': 1.0,
                         'unit': FIFF.FIFF_UNIT_V,
                         'unit_mul': 0,
                         'ch_name': 'MISC %03d' % idx,
                         'coil_type': FIFF.FIFFV_COIL_NONE,
                         'loc': np.zeros(12),
                         'kind': FIFF.FIFFV_MISC_CH}
            info['chs'].append(chan_info)
    info._update_redundant()
    return info, sqd


def read_raw_kit(input_fname, mrk=None, elp=None, hsp=None, stim='>',
                 slope='-', stimthresh=1, preload=False, stim_code='binary',
                 verbose=None):
    """Reader function for KIT conversion to FIF.

    Parameters
    ----------
    input_fname : str
        Path to the sqd file.
    mrk : None | str | array_like, shape (5, 3) | list of str or array_like
        Marker points representing the location of the marker coils with
        respect to the MEG Sensors, or path to a marker file.
        If list, all of the markers will be averaged together.
    elp : None | str | array_like, shape (8, 3)
        Digitizer points representing the location of the fiducials and the
        marker coils with respect to the digitized head shape, or path to a
        file containing these points.
    hsp : None | str | array, shape (n_points, 3)
        Digitizer head shape points, or path to head shape file. If more than
        10,000 points are in the head shape, they are automatically decimated.
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
    stim_code : 'binary' | 'channel'
        How to decode trigger values from stim channels. 'binary' read stim
        channel events as binary code, 'channel' encodes channel number.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raw : Instance of RawKIT
        A Raw object containing KIT data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    If mrk, hsp or elp are array_like inputs, then the numbers in xyz
    coordinates should be in units of meters.
    """
    return RawKIT(input_fname=input_fname, mrk=mrk, elp=elp, hsp=hsp,
                  stim=stim, slope=slope, stimthresh=stimthresh,
                  preload=preload, stim_code=stim_code, verbose=verbose)


def read_epochs_kit(input_fname, events, event_id=None,
                    mrk=None, elp=None, hsp=None, verbose=None):
    """Reader function for KIT epochs files.

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
        the keys can later be used to access associated events. Example:
        dict(auditory=1, visual=3). If int, a dict will be created with
        the id as string. If a list, all events with the IDs specified
        in the list are used. If None, all events will be used with
        and a dict is created with string integer names corresponding
        to the event id integers.
    mrk : None | str | array_like, shape (5, 3) | list of str or array_like
        Marker points representing the location of the marker coils with
        respect to the MEG Sensors, or path to a marker file.
        If list, all of the markers will be averaged together.
    elp : None | str | array_like, shape (8, 3)
        Digitizer points representing the location of the fiducials and the
        marker coils with respect to the digitized head shape, or path to a
        file containing these points.
    hsp : None | str | array, shape (n_points, 3)
        Digitizer head shape points, or path to head shape file. If more than
        10,000 points are in the head shape, they are automatically decimated.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

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
