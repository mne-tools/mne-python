"""Conversion tool from SQD to FIF.

RawKIT class is adapted from Denis Engemann et al.'s mne_bti2fiff.py.
"""

# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

from collections import defaultdict, OrderedDict
from math import sin, cos
from os import SEEK_CUR, path as op

import numpy as np

from ..pick import pick_types
from ...utils import (verbose, logger, warn, fill_doc, _check_option,
                      _stamp_to_dt, _check_fname)
from ...transforms import apply_trans, als_ras_trans
from ..base import BaseRaw
from ..utils import _mult_cal_one
from ...epochs import BaseEpochs
from ..constants import FIFF
from ..meas_info import _empty_info
from .constants import KIT, LEGACY_AMP_PARAMS
from .coreg import read_mrk, _set_dig_kit
from ...event import read_events


FLOAT64 = '<f8'
UINT32 = '<u4'
INT32 = '<i4'


def _call_digitization(info, mrk, elp, hsp, kit_info):
    # Use values from kit_info only if all others are None
    if mrk is None and elp is None and hsp is None:
        mrk = kit_info.get('mrk', None)
        elp = kit_info.get('elp', None)
        hsp = kit_info.get('hsp', None)

    # prepare mrk
    if isinstance(mrk, list):
        mrk = [read_mrk(marker) if isinstance(marker, str)
               else marker for marker in mrk]
        mrk = np.mean(mrk, axis=0)

    # setup digitization
    if mrk is not None and elp is not None and hsp is not None:
        with info._unlock():
            info['dig'], info['dev_head_t'], info['hpi_results'] = \
                _set_dig_kit(mrk, elp, hsp, kit_info['eeg_dig'])
    elif mrk is not None or elp is not None or hsp is not None:
        raise ValueError("mrk, elp and hsp need to be provided as a group "
                         "(all or none)")

    return info


class UnsupportedKITFormat(ValueError):
    """Our reader is not guaranteed to work with old files."""

    def __init__(self, sqd_version, *args, **kwargs):  # noqa: D102
        self.sqd_version = sqd_version
        ValueError.__init__(self, *args, **kwargs)


@fill_doc
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
    stim : list of int | '<' | '>' | None
        Channel-value correspondence when converting KIT trigger channels to a
        Neuromag-style stim channel. For '<', the largest values are assigned
        to the first channel (default). For '>', the largest values are
        assigned to the last channel. Can also be specified as a list of
        trigger channel indexes. If None, no synthesized channel is generated.
    slope : '+' | '-'
        How to interpret values on KIT trigger channels when synthesizing a
        Neuromag-style stim channel. With '+', a positive slope (low-to-high)
        is interpreted as an event. With '-', a negative slope (high-to-low)
        is interpreted as an event.
    stimthresh : float
        The threshold level for accepting voltage changes in KIT trigger
        channels as a trigger event. If None, stim must also be set to None.
    %(preload)s
    stim_code : 'binary' | 'channel'
        How to decode trigger values from stim channels. 'binary' read stim
        channel events as binary code, 'channel' encodes channel number.
    allow_unknown_format : bool
        Force reading old data that is not officially supported. Alternatively,
        read and re-save the data with the KIT MEG Laboratory application.
    %(standardize_names)s
    %(verbose)s

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
                 allow_unknown_format=False, standardize_names=None,
                 verbose=None):  # noqa: D102
        logger.info('Extracting SQD Parameters from %s...' % input_fname)
        input_fname = op.abspath(input_fname)
        self.preload = False
        logger.info('Creating Raw.info structure...')
        info, kit_info = get_kit_info(
            input_fname, allow_unknown_format, standardize_names)
        kit_info['slope'] = slope
        kit_info['stimthresh'] = stimthresh
        if kit_info['acq_type'] != KIT.CONTINUOUS:
            raise TypeError('SQD file contains epochs, not raw data. Wrong '
                            'reader.')
        logger.info('Creating Info structure...')

        last_samps = [kit_info['n_samples'] - 1]
        self._raw_extras = [kit_info]
        self._set_stimchannels(info, stim, stim_code)
        super(RawKIT, self).__init__(
            info, preload, last_samps=last_samps, filenames=[input_fname],
            raw_extras=self._raw_extras, verbose=verbose)
        self.info = _call_digitization(
            info=self.info, mrk=mrk, elp=elp, hsp=hsp, kit_info=kit_info)
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
        stim_ch = np.empty((1, stop), dtype=np.int64)
        for b_start in range(start, stop, buffer_size):
            b_stop = b_start + buffer_size
            x = self[pick, b_start:b_stop][0]
            stim_ch[:, b_start:b_start + x.shape[1]] = x

        return stim_ch

    @fill_doc
    def _set_stimchannels(self, info, stim, stim_code):
        """Specify how the trigger channel is synthesized from analog channels.

        Has to be done before loading data. For a RawKIT instance that has been
        created with preload=True, this method will raise a
        NotImplementedError.

        Parameters
        ----------
        %(info_not_none)s
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
        if self.preload:
            raise NotImplementedError("Can't change stim channel after "
                                      "loading data")
        _check_option('stim_code', stim_code, ['binary', 'channel'])

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
            info['chs'].append(dict(
                cal=KIT.CALIB_FACTOR, logno=nchan, scanno=nchan, range=1.0,
                unit=FIFF.FIFF_UNIT_NONE, unit_mul=FIFF.FIFF_UNITM_NONE,
                ch_name='STI 014',
                coil_type=FIFF.FIFFV_COIL_NONE, loc=np.full(12, np.nan),
                kind=FIFF.FIFFV_STIM_CH, coord_frame=FIFF.FIFFV_COORD_UNKNOWN))
            info._update_redundant()

        self._raw_extras[0]['stim'] = stim
        self._raw_extras[0]['stim_code'] = stim_code

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        sqd = self._raw_extras[fi]
        nchan = sqd['nchan']
        data_left = (stop - start) * nchan
        conv_factor = sqd['conv_factor']

        n_bytes = sqd['dtype'].itemsize
        assert n_bytes in (2, 4)
        # Read up to 100 MB of data at a time.
        blk_size = min(data_left, (100000000 // n_bytes // nchan) * nchan)
        with open(self._filenames[fi], 'rb', buffering=0) as fid:
            # extract data
            pointer = start * nchan * n_bytes
            fid.seek(sqd['dirs'][KIT.DIR_INDEX_RAW_DATA]['offset'] + pointer)
            stim = sqd['stim']
            for blk_start in np.arange(0, data_left, blk_size) // nchan:
                blk_size = min(blk_size, data_left - blk_start * nchan)
                block = np.fromfile(fid, dtype=sqd['dtype'], count=blk_size)
                block = block.reshape(nchan, -1, order='F').astype(float)
                blk_stop = blk_start + block.shape[1]
                data_view = data[:, blk_start:blk_stop]
                block *= conv_factor

                # Create a synthetic stim channel
                if stim is not None:
                    stim_ch = _make_stim_channel(
                        block[stim, :], sqd['slope'], sqd['stimthresh'],
                        sqd['stim_code'], stim)
                    block = np.vstack((block, stim_ch))

                _mult_cal_one(data_view, block, idx, cals, mult)
        # cals are all unity, so can be ignored


def _default_stim_chs(info):
    """Return default stim channels for SQD files."""
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
    allow_unknown_format : bool
        Force reading old data that is not officially supported. Alternatively,
        read and re-save the data with the KIT MEG Laboratory application.
    %(standardize_names)s
    %(verbose)s

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
                 baseline=None, reject=None, flat=None, reject_tmin=None,
                 reject_tmax=None, mrk=None, elp=None, hsp=None,
                 allow_unknown_format=False, standardize_names=None,
                 verbose=None):  # noqa: D102

        if isinstance(events, str):
            events = read_events(events)

        input_fname = _check_fname(fname=input_fname, must_exist=True,
                                   overwrite='read')
        logger.info('Extracting KIT Parameters from %s...' % input_fname)
        self.info, kit_info = get_kit_info(
            input_fname, allow_unknown_format, standardize_names)
        kit_info.update(filename=input_fname)
        self._raw_extras = [kit_info]
        self._filenames = []
        if len(events) != self._raw_extras[0]['n_epochs']:
            raise ValueError('Event list does not match number of epochs.')

        if self._raw_extras[0]['acq_type'] == KIT.EPOCHS:
            self._raw_extras[0]['data_length'] = KIT.INT
        else:
            raise TypeError('SQD file contains raw data, not epochs or '
                            'average. Wrong reader.')

        if event_id is None:  # convert to int to make typing-checks happy
            event_id = {str(e): int(e) for e in np.unique(events[:, 2])}

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
        self.info = _call_digitization(
            info=self.info, mrk=mrk, elp=elp, hsp=hsp, kit_info=kit_info)
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
        info = self._raw_extras[0]
        epoch_length = info['frame_length']
        n_epochs = info['n_epochs']
        n_samples = info['n_samples']
        filename = info['filename']
        dtype = info['dtype']
        nchan = info['nchan']

        with open(filename, 'rb', buffering=0) as fid:
            fid.seek(info['dirs'][KIT.DIR_INDEX_RAW_DATA]['offset'])
            count = n_samples * nchan
            data = np.fromfile(fid, dtype=dtype, count=count)
        data = data.reshape((n_samples, nchan)).T
        data = data * info['conv_factor']
        data = data.reshape((nchan, n_epochs, epoch_length))
        data = data.transpose((1, 0, 2))

        return data


def _read_dir(fid):
    return dict(offset=np.fromfile(fid, UINT32, 1)[0],
                size=np.fromfile(fid, INT32, 1)[0],
                max_count=np.fromfile(fid, INT32, 1)[0],
                count=np.fromfile(fid, INT32, 1)[0])


@verbose
def _read_dirs(fid, verbose=None):
    dirs = list()
    dirs.append(_read_dir(fid))
    for ii in range(dirs[0]['count'] - 1):
        logger.debug(f'    KIT dir entry {ii} @ {fid.tell()}')
        dirs.append(_read_dir(fid))
    assert len(dirs) == dirs[KIT.DIR_INDEX_DIR]['count']
    return dirs


@verbose
def get_kit_info(rawfile, allow_unknown_format, standardize_names=None,
                 verbose=None):
    """Extract all the information from the sqd/con file.

    Parameters
    ----------
    rawfile : str
        KIT file to be read.
    allow_unknown_format : bool
        Force reading old data that is not officially supported. Alternatively,
        read and re-save the data with the KIT MEG Laboratory application.
    %(standardize_names)s
    %(verbose)s

    Returns
    -------
    %(info_not_none)s
    sqd : dict
        A dict containing all the sqd parameter settings.
    """
    sqd = dict()
    sqd['rawfile'] = rawfile
    unsupported_format = False
    with open(rawfile, 'rb', buffering=0) as fid:  # buffering=0 for np bug
        #
        # directories (0)
        #
        sqd['dirs'] = dirs = _read_dirs(fid)

        #
        # system (1)
        #
        fid.seek(dirs[KIT.DIR_INDEX_SYSTEM]['offset'])
        # check file format version
        version, revision = np.fromfile(fid, INT32, 2)
        if version < 2 or (version == 2 and revision < 3):
            version_string = "V%iR%03i" % (version, revision)
            if allow_unknown_format:
                unsupported_format = True
                warn("Force loading KIT format %s" % version_string)
            else:
                raise UnsupportedKITFormat(
                    version_string,
                    "SQD file format %s is not officially supported. "
                    "Set allow_unknown_format=True to load it anyways." %
                    (version_string,))

        sysid = np.fromfile(fid, INT32, 1)[0]
        # basic info
        system_name = _read_name(fid, n=128)
        # model name
        model_name = _read_name(fid, n=128)
        # channels
        sqd['nchan'] = channel_count = int(np.fromfile(fid, INT32, 1)[0])
        comment = _read_name(fid, n=256)
        create_time, last_modified_time = np.fromfile(fid, INT32, 2)
        fid.seek(KIT.INT * 3, SEEK_CUR)  # reserved
        dewar_style = np.fromfile(fid, INT32, 1)[0]
        fid.seek(KIT.INT * 3, SEEK_CUR)  # spare
        fll_type = np.fromfile(fid, INT32, 1)[0]
        fid.seek(KIT.INT * 3, SEEK_CUR)  # spare
        trigger_type = np.fromfile(fid, INT32, 1)[0]
        fid.seek(KIT.INT * 3, SEEK_CUR)  # spare
        adboard_type = np.fromfile(fid, INT32, 1)[0]
        fid.seek(KIT.INT * 29, SEEK_CUR)  # reserved

        if version < 2 or (version == 2 and revision <= 3):
            adc_range = float(np.fromfile(fid, INT32, 1)[0])
        else:
            adc_range = np.fromfile(fid, FLOAT64, 1)[0]
        adc_polarity, adc_allocated, adc_stored = np.fromfile(fid, INT32, 3)
        system_name = system_name.replace('\x00', '')
        system_name = system_name.strip().replace('\n', '/')
        model_name = model_name.replace('\x00', '')
        model_name = model_name.strip().replace('\n', '/')

        full_version = f'V{version:d}R{revision:03d}'
        logger.debug("SQD file basic information:")
        logger.debug("Meg160 version = %s", full_version)
        logger.debug("System ID      = %i", sysid)
        logger.debug("System name    = %s", system_name)
        logger.debug("Model name     = %s", model_name)
        logger.debug("Channel count  = %i", channel_count)
        logger.debug("Comment        = %s", comment)
        logger.debug("Dewar style    = %i", dewar_style)
        logger.debug("FLL type       = %i", fll_type)
        logger.debug("Trigger type   = %i", trigger_type)
        logger.debug("A/D board type = %i", adboard_type)
        logger.debug("ADC range      = +/-%s[V]", adc_range / 2.)
        logger.debug("ADC allocate   = %i[bit]", adc_allocated)
        logger.debug("ADC bit        = %i[bit]", adc_stored)
        # MGH description: 'acquisition (megacq) VectorView system at NMR-MGH'
        description = \
            f'{system_name} ({sysid}) {full_version} {model_name}'
        assert adc_allocated % 8 == 0
        sqd['dtype'] = np.dtype(f'<i{adc_allocated // 8}')

        # check that we can read this file
        if fll_type not in KIT.FLL_SETTINGS:
            fll_types = sorted(KIT.FLL_SETTINGS.keys())
            use_fll_type = fll_types[
                np.searchsorted(fll_types, fll_type) - 1]
            warn('Unknown site filter settings (FLL) for system '
                 '"%s" model "%s" (ID %s), will assume FLL %d->%d, check '
                 'your data for correctness, including channel scales and '
                 'filter settings!'
                 % (system_name, model_name, sysid, fll_type, use_fll_type))
            fll_type = use_fll_type

        #
        # channel information (4)
        #
        chan_dir = dirs[KIT.DIR_INDEX_CHANNELS]
        chan_offset, chan_size = chan_dir['offset'], chan_dir['size']
        sqd['channels'] = channels = []
        exg_gains = list()
        for i in range(channel_count):
            fid.seek(chan_offset + chan_size * i)
            channel_type, = np.fromfile(fid, INT32, 1)
            # System 52 mislabeled reference channels as NULL. This was fixed
            # in system 53; not sure about 51...
            if sysid == 52 and i < 160 and channel_type == KIT.CHANNEL_NULL:
                channel_type = KIT.CHANNEL_MAGNETOMETER_REFERENCE

            if channel_type in KIT.CHANNELS_MEG:
                if channel_type not in KIT.CH_TO_FIFF_COIL:
                    raise NotImplementedError(
                        "KIT channel type %i can not be read. Please contact "
                        "the mne-python developers." % channel_type)
                channels.append({
                    'type': channel_type,
                    # (x, y, z, theta, phi) for all MEG channels. Some channel
                    # types have additional information which we're not using.
                    'loc': np.fromfile(fid, dtype=FLOAT64, count=5),
                })
                if channel_type in KIT.CHANNEL_NAME_NCHAR:
                    fid.seek(16, SEEK_CUR)  # misc fields
                    channels[-1]['name'] = _read_name(fid, channel_type)
            elif channel_type in KIT.CHANNELS_MISC:
                channel_no, = np.fromfile(fid, INT32, 1)
                fid.seek(4, SEEK_CUR)
                name = _read_name(fid, channel_type)
                channels.append({
                    'type': channel_type,
                    'no': channel_no,
                    'name': name,
                })
                if channel_type in (KIT.CHANNEL_EEG, KIT.CHANNEL_ECG):
                    offset = 6 if channel_type == KIT.CHANNEL_EEG else 8
                    fid.seek(offset, SEEK_CUR)
                    exg_gains.append(np.fromfile(fid, FLOAT64, 1)[0])
            elif channel_type == KIT.CHANNEL_NULL:
                channels.append({'type': channel_type})
            else:
                raise IOError("Unknown KIT channel type: %i" % channel_type)
        exg_gains = np.array(exg_gains)

        #
        # Channel sensitivity information: (5)
        #

        # only sensor channels requires gain. the additional misc channels
        # (trigger channels, audio and voice channels) are passed
        # through unaffected
        fid.seek(dirs[KIT.DIR_INDEX_CALIBRATION]['offset'])
        # (offset [Volt], gain [Tesla/Volt]) for each channel
        sensitivity = np.fromfile(fid, dtype=FLOAT64, count=channel_count * 2)
        sensitivity.shape = (channel_count, 2)
        channel_offset, channel_gain = sensitivity.T
        assert (channel_offset == 0).all()  # otherwise we have a problem

        #
        # amplifier gain (7)
        #
        fid.seek(dirs[KIT.DIR_INDEX_AMP_FILTER]['offset'])
        amp_data = np.fromfile(fid, INT32, 1)[0]
        if fll_type >= 100:  # Kapper Type
            # gain:             mask           bit
            gain1 = (amp_data & 0x00007000) >> 12
            gain2 = (amp_data & 0x70000000) >> 28
            gain3 = (amp_data & 0x07000000) >> 24
            amp_gain = (KIT.GAINS[gain1] * KIT.GAINS[gain2] * KIT.GAINS[gain3])
            # filter settings
            hpf = (amp_data & 0x00000700) >> 8
            lpf = (amp_data & 0x00070000) >> 16
            bef = (amp_data & 0x00000003) >> 0
        else:  # Hanger Type
            # gain
            input_gain = (amp_data & 0x1800) >> 11
            output_gain = (amp_data & 0x0007) >> 0
            amp_gain = KIT.GAINS[input_gain] * KIT.GAINS[output_gain]
            # filter settings
            hpf = (amp_data & 0x007) >> 4
            lpf = (amp_data & 0x0700) >> 8
            bef = (amp_data & 0xc000) >> 14
        hpf_options, lpf_options, bef_options = KIT.FLL_SETTINGS[fll_type]
        sqd['highpass'] = KIT.HPFS[hpf_options][hpf]
        sqd['lowpass'] = KIT.LPFS[lpf_options][lpf]
        sqd['notch'] = KIT.BEFS[bef_options][bef]

        #
        # Acquisition Parameters (8)
        #
        fid.seek(dirs[KIT.DIR_INDEX_ACQ_COND]['offset'])
        sqd['acq_type'], = acq_type, = np.fromfile(fid, INT32, 1)
        sqd['sfreq'], = np.fromfile(fid, FLOAT64, 1)
        if acq_type == KIT.CONTINUOUS:
            # samples_count, = np.fromfile(fid, INT32, 1)
            fid.seek(KIT.INT, SEEK_CUR)
            sqd['n_samples'], = np.fromfile(fid, INT32, 1)
        elif acq_type == KIT.EVOKED or acq_type == KIT.EPOCHS:
            sqd['frame_length'], = np.fromfile(fid, INT32, 1)
            sqd['pretrigger_length'], = np.fromfile(fid, INT32, 1)
            sqd['average_count'], = np.fromfile(fid, INT32, 1)
            sqd['n_epochs'], = np.fromfile(fid, INT32, 1)
            if acq_type == KIT.EVOKED:
                sqd['n_samples'] = sqd['frame_length']
            else:
                sqd['n_samples'] = sqd['frame_length'] * sqd['n_epochs']
        else:
            raise IOError("Invalid acquisition type: %i. Your file is neither "
                          "continuous nor epoched data." % (acq_type,))

        #
        # digitization information (12 and 26)
        #
        dig_dir = dirs[KIT.DIR_INDEX_DIG_POINTS]
        cor_dir = dirs[KIT.DIR_INDEX_COREG]
        dig = dict()
        hsp = list()
        if dig_dir['count'] > 0 and cor_dir['count'] > 0:
            # directories (0)
            fid.seek(dig_dir['offset'])
            for _ in range(dig_dir['count']):
                name = _read_name(fid, n=8).strip()
                # Sometimes there are mismatches (e.g., AFz vs AFZ) between
                # the channel name and its digitized, name, so let's be case
                # insensitive. It will also prevent collisions with HSP
                name = name.lower()
                rr = np.fromfile(fid, FLOAT64, 3)
                if name:
                    assert name not in dig
                    dig[name] = rr
                else:
                    hsp.append(rr)

            # nasion, lpa, rpa, HPI in native space
            elp = []
            for key in (
                    'fidnz', 'fidt9', 'fidt10',
                    'hpi_1', 'hpi_2', 'hpi_3', 'hpi_4', 'hpi_5'):
                if key in dig and np.isfinite(dig[key]).all():
                    elp.append(dig.pop(key))
            elp = np.array(elp)
            hsp = np.array(hsp, float).reshape(-1, 3)
            if elp.shape not in ((6, 3), (7, 3), (8, 3)):
                raise RuntimeError(
                    f'Fewer than 3 HPI coils found, got {len(elp) - 3}')
            # coregistration
            fid.seek(cor_dir['offset'])
            mrk = np.zeros((elp.shape[0] - 3, 3))
            meg_done = [True] * 5
            for _ in range(cor_dir['count']):
                done = np.fromfile(fid, INT32, 1)[0]
                fid.seek(16 * KIT.DOUBLE +  # meg_to_mri
                         16 * KIT.DOUBLE,  # mri_to_meg
                         SEEK_CUR)
                marker_count = np.fromfile(fid, INT32, 1)[0]
                if not done:
                    continue
                assert marker_count >= len(mrk)
                for mi in range(len(mrk)):
                    mri_type, meg_type, mri_done, this_meg_done = \
                        np.fromfile(fid, INT32, 4)
                    meg_done[mi] = bool(this_meg_done)
                    fid.seek(3 * KIT.DOUBLE, SEEK_CUR)  # mri_pos
                    mrk[mi] = np.fromfile(fid, FLOAT64, 3)
                fid.seek(256, SEEK_CUR)  # marker_file (char)
            if not all(meg_done):
                logger.info(f'Keeping {sum(meg_done)}/{len(meg_done)} HPI '
                            'coils that were digitized')
                elp = elp[[True] * 3 + meg_done]
                mrk = mrk[meg_done]
            sqd.update(hsp=hsp, elp=elp, mrk=mrk)

    # precompute conversion factor for reading data
    if unsupported_format:
        if sysid not in LEGACY_AMP_PARAMS:
            raise IOError("Legacy parameters for system ID %i unavailable" %
                          (sysid,))
        adc_range, adc_stored = LEGACY_AMP_PARAMS[sysid]
    is_meg = np.array([ch['type'] in KIT.CHANNELS_MEG for ch in channels])
    ad_to_volt = adc_range / (2. ** adc_stored)
    ad_to_tesla = ad_to_volt / amp_gain * channel_gain
    conv_factor = np.where(is_meg, ad_to_tesla, ad_to_volt)
    # XXX this is a bit of a hack. Should probably do this more cleanly at
    # some point... the 2 ** (adc_stored - 14) was emperically determined using
    # the test files with known amplitudes. The conv_factors need to be
    # replaced by these values otherwise we're off by a factor off 5000.0
    # for the EEG data.
    is_exg = [ch['type'] in (KIT.CHANNEL_EEG, KIT.CHANNEL_ECG)
              for ch in channels]
    exg_gains /= 2. ** (adc_stored - 14)
    exg_gains[exg_gains == 0] = ad_to_volt
    conv_factor[is_exg] = exg_gains
    sqd['conv_factor'] = conv_factor[:, np.newaxis]

    # Create raw.info dict for raw fif object with SQD data
    info = _empty_info(float(sqd['sfreq']))
    info.update(meas_date=_stamp_to_dt((create_time, 0)),
                lowpass=sqd['lowpass'],
                highpass=sqd['highpass'], kit_system_id=sysid,
                description=description)

    # Creates a list of dicts of meg channels for raw.info
    logger.info('Setting channel info structure...')
    info['chs'] = fiff_channels = []
    channel_index = defaultdict(lambda: 0)
    sqd['eeg_dig'] = OrderedDict()
    for idx, ch in enumerate(channels, 1):
        if ch['type'] in KIT.CHANNELS_MEG:
            ch_name = ch.get('name', '')
            if ch_name == '' or standardize_names:
                ch_name = 'MEG %03d' % idx
            # create three orthogonal vector
            # ch_angles[0]: theta, ch_angles[1]: phi
            theta, phi = np.radians(ch['loc'][3:])
            x = sin(theta) * cos(phi)
            y = sin(theta) * sin(phi)
            z = cos(theta)
            vec_z = np.array([x, y, z])
            vec_z /= np.linalg.norm(vec_z)
            vec_x = np.zeros(vec_z.size, dtype=np.float64)
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
            vec_x /= np.linalg.norm(vec_x)
            vec_y = np.cross(vec_z, vec_x)
            # transform to Neuromag like coordinate space
            vecs = np.vstack((ch['loc'][:3], vec_x, vec_y, vec_z))
            vecs = apply_trans(als_ras_trans, vecs)
            unit = FIFF.FIFF_UNIT_T
            loc = vecs.ravel()
        else:
            ch_type_label = KIT.CH_LABEL[ch['type']]
            channel_index[ch_type_label] += 1
            ch_type_index = channel_index[ch_type_label]
            ch_name = ch.get('name', '')
            eeg_name = ch_name.lower()
            # some files have all EEG labeled as EEG
            if ch_name in ('', 'EEG') or standardize_names:
                ch_name = '%s %03i' % (ch_type_label, ch_type_index)
            unit = FIFF.FIFF_UNIT_V
            loc = np.zeros(12)
            if eeg_name and eeg_name in dig:
                loc[:3] = sqd['eeg_dig'][eeg_name] = dig[eeg_name]
        fiff_channels.append(dict(
            cal=KIT.CALIB_FACTOR, logno=idx, scanno=idx, range=KIT.RANGE,
            unit=unit, unit_mul=KIT.UNIT_MUL, ch_name=ch_name,
            coord_frame=FIFF.FIFFV_COORD_DEVICE,
            coil_type=KIT.CH_TO_FIFF_COIL[ch['type']],
            kind=KIT.CH_TO_FIFF_KIND[ch['type']], loc=loc))
    info._unlocked = False
    info._update_redundant()
    return info, sqd


def _read_name(fid, ch_type=None, n=None):
    n = n if ch_type is None else KIT.CHANNEL_NAME_NCHAR[ch_type]
    return fid.read(n).split(b'\x00')[0].decode('utf-8')


@fill_doc
def read_raw_kit(input_fname, mrk=None, elp=None, hsp=None, stim='>',
                 slope='-', stimthresh=1, preload=False, stim_code='binary',
                 allow_unknown_format=False, standardize_names=False,
                 verbose=None):
    """Reader function for Ricoh/KIT conversion to FIF.

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
    %(preload)s
    stim_code : 'binary' | 'channel'
        How to decode trigger values from stim channels. 'binary' read stim
        channel events as binary code, 'channel' encodes channel number.
    allow_unknown_format : bool
        Force reading old data that is not officially supported. Alternatively,
        read and re-save the data with the KIT MEG Laboratory application.
    %(standardize_names)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawKIT
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
                  preload=preload, stim_code=stim_code,
                  allow_unknown_format=allow_unknown_format,
                  standardize_names=standardize_names, verbose=verbose)


@fill_doc
def read_epochs_kit(input_fname, events, event_id=None, mrk=None, elp=None,
                    hsp=None, allow_unknown_format=False,
                    standardize_names=False, verbose=None):
    """Reader function for Ricoh/KIT epochs files.

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
    allow_unknown_format : bool
        Force reading old data that is not officially supported. Alternatively,
        read and re-save the data with the KIT MEG Laboratory application.
    %(standardize_names)s
    %(verbose)s

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
                       allow_unknown_format=allow_unknown_format,
                       standardize_names=standardize_names,
                       verbose=verbose)
    return epochs
