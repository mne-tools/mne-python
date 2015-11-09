# -*- coding: utf-8 -*-
"""Conversion tool from Brain Vision EEG to FIF"""

# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Christian Brodbeck <christianbrodbeck@nyu.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os
import time
import re
import warnings

import numpy as np

from ...utils import verbose, logger
from ..constants import FIFF
from ..meas_info import _empty_info
from ..base import _BaseRaw, _check_update_montage
from ..reference import add_reference_channels

from ...externals.six import StringIO, u
from ...externals.six.moves import configparser


class RawBrainVision(_BaseRaw):
    """Raw object from Brain Vision EEG file

    Parameters
    ----------
    vhdr_fname : str
        Path to the EEG header file.
    montage : str | None | instance of Montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. Values should correspond to the vhdr file.
        Default is ``('HEOGL', 'HEOGR', 'VEOGb')``.
    misc : list or tuple
        Names of channels or list of indices that should be designated
        MISC channels. Values should correspond to the electrodes
        in the vhdr file. Default is ``()``.
    reference : None | str
        **Deprecated**, use `add_reference_channel` instead.
        Name of the electrode which served as the reference in the recording.
        If a name is provided, a corresponding channel is added and its data
        is set to 0. This is useful for later re-referencing. The name should
        correspond to a name in elp_names. Data must be preloaded.
    scale : float
        The scaling factor for EEG data. Units are in volts. Default scale
        factor is 1. For microvolts, the scale factor would be 1e-6. This is
        used when the header file does not specify the scale factor.
    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.
    response_trig_shift : int | None
        An integer that will be added to all response triggers when reading
        events (stimulus triggers will be unaffected). If None, response
        triggers will be ignored. Default is 0 for backwards compatibility, but
        typically another value or None will be necessary.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    @verbose
    def __init__(self, vhdr_fname, montage=None,
                 eog=('HEOGL', 'HEOGR', 'VEOGb'), misc=(), reference=None,
                 scale=1., preload=False, response_trig_shift=0, verbose=None):
        # Channel info and events
        logger.info('Extracting parameters from %s...' % vhdr_fname)
        vhdr_fname = os.path.abspath(vhdr_fname)
        info, fmt, self._order, events = _get_vhdr_info(
            vhdr_fname, eog, misc, response_trig_shift, scale)
        _check_update_montage(info, montage)
        with open(info['filename'], 'rb') as f:
            f.seek(0, os.SEEK_END)
            n_samples = f.tell()
        dtype_bytes = _fmt_byte_dict[fmt]
        self.preload = False  # so the event-setting works
        self.set_brainvision_events(events)
        last_samps = [(n_samples // (dtype_bytes * (info['nchan'] - 1))) - 1]
        super(RawBrainVision, self).__init__(
            info, last_samps=last_samps, filenames=[info['filename']],
            orig_format=fmt, preload=preload, verbose=verbose)

        # add reference
        if reference is not None:
            warnings.warn('reference is deprecated and will be removed in '
                          'v0.11. Use add_reference_channels instead.')
            if preload is False:
                raise ValueError("Preload must be set to True if reference is "
                                 "specified.")
            add_reference_channels(self, reference, copy=False)

    def _read_segment_file(self, data, idx, offset, fi, start, stop,
                           cals, mult):
        """Read a chunk of raw data"""
        # read data
        n_data_ch = len(self.ch_names) - 1
        n_times = stop - start + 1
        pointer = start * n_data_ch * _fmt_byte_dict[self.orig_format]
        with open(self._filenames[fi], 'rb') as f:
            f.seek(pointer)
            # extract data
            data_buffer = np.fromfile(
                f, dtype=_fmt_dtype_dict[self.orig_format],
                count=n_times * n_data_ch)
        data_buffer = data_buffer.reshape((n_data_ch, n_times),
                                          order=self._order)

        data_ = np.empty((n_data_ch + 1, n_times), dtype=np.float64)
        data_[:-1] = data_buffer  # cast to float64
        del data_buffer
        data_[-1] = _synthesize_stim_channel(self._events, start, stop + 1)
        data_ *= self._cals[:, np.newaxis]
        data[:, offset:offset + stop - start + 1] = \
            np.dot(mult, data_) if mult is not None else data_[idx]

    def get_brainvision_events(self):
        """Retrieve the events associated with the Brain Vision Raw object

        Returns
        -------
        events : array, shape (n_events, 3)
            Events, each row consisting of an (onset, duration, trigger)
            sequence.
        """
        return self._events.copy()

    def set_brainvision_events(self, events):
        """Set the events and update the synthesized stim channel

        Parameters
        ----------
        events : array, shape (n_events, 3)
            Events, each row consisting of an (onset, duration, trigger)
            sequence.
        """
        events = np.array(events, int)
        if events.ndim != 2 or events.shape[1] != 3:
            raise ValueError("[n_events x 3] shaped array required")
        # update events
        self._events = events
        if self.preload:
            start = self.first_samp
            stop = self.last_samp + 1
            self._data[-1] = _synthesize_stim_channel(events, start, stop)


def _read_vmrk_events(fname, response_trig_shift=0):
    """Read events from a vmrk file

    Parameters
    ----------
    fname : str
        vmrk file to be read.
    response_trig_shift : int | None
        Integer to shift response triggers by. None ignores response triggers.

    Returns
    -------
    events : array, shape (n_events, 3)
        An array containing the whole recording's events, each row representing
        an event as (onset, duration, trigger) sequence.
    """
    # read vmrk file
    with open(fname) as fid:
        txt = fid.read()

    header = txt.split('\n')[0].strip()
    start_tag = 'Brain Vision Data Exchange Marker File'
    if not header.startswith(start_tag):
        raise ValueError("vmrk file should start with %r" % start_tag)
    end_tag = 'Version 1.0'
    if not header.endswith(end_tag):
        raise ValueError("vmrk file should be %r" % end_tag)
    if (response_trig_shift is not None and
            not isinstance(response_trig_shift, int)):
        raise TypeError("response_trig_shift must be an integer or None")

    # extract Marker Infos block
    m = re.search("\[Marker Infos\]", txt)
    if not m:
        return np.zeros(0)
    mk_txt = txt[m.end():]
    m = re.search("\[.*\]", mk_txt)
    if m:
        mk_txt = mk_txt[:m.start()]

    # extract event information
    items = re.findall("^Mk\d+=(.*)", mk_txt, re.MULTILINE)
    events = []
    for info in items:
        mtype, mdesc, onset, duration = info.split(',')[:4]
        try:
            trigger = int(re.findall('[A-Za-z]*\s*?(\d+)', mdesc)[0])
            if mdesc[0].lower() == 's' or response_trig_shift is not None:
                if mdesc[0].lower() == 'r':
                    trigger += response_trig_shift
                onset = int(onset)
                duration = int(duration)
                events.append((onset, duration, trigger))
        except IndexError:
            pass

    events = np.array(events).reshape(-1, 3)
    return events


def _synthesize_stim_channel(events, start, stop):
    """Synthesize a stim channel from events read from a vmrk file

    Parameters
    ----------
    events : array, shape (n_events, 3)
        Each row representing an event as (onset, duration, trigger) sequence
        (the format returned by _read_vmrk_events).
    start : int
        First sample to return.
    stop : int
        Last sample to return.

    Returns
    -------
    stim_channel : array, shape (n_samples,)
        An array containing the whole recording's event marking
    """
    # select events overlapping buffer
    onset = events[:, 0]
    offset = onset + events[:, 1]
    idx = np.logical_and(onset < stop, offset > start)
    if idx.sum() > 0:  # fix for old numpy
        events = events[idx]

    # make onset relative to buffer
    events[:, 0] -= start

    # fix onsets before buffer start
    idx = events[:, 0] < 0
    events[idx, 0] = 0

    # create output buffer
    stim_channel = np.zeros(stop - start)
    for onset, duration, trigger in events:
        stim_channel[onset:onset + duration] = trigger

    return stim_channel


_orientation_dict = dict(MULTIPLEXED='F', VECTORIZED='C')
_fmt_dict = dict(INT_16='short', INT_32='int', IEEE_FLOAT_32='single')
_fmt_byte_dict = dict(short=2, int=4, single=4)
_fmt_dtype_dict = dict(short='<i2', int='<i4', single='<f4')
_unit_dict = {'V': 1., u'µV': 1e-6}


def _get_vhdr_info(vhdr_fname, eog, misc, response_trig_shift, scale):
    """Extracts all the information from the header file.

    Parameters
    ----------
    vhdr_fname : str
        Raw EEG header to be read.
    eog : list of str
        Names of channels that should be designated EOG channels. Names should
        correspond to the vhdr file.
    misc : list of str
        Names of channels that should be designated MISC channels. Names
        should correspond to the electrodes in the vhdr file.
    response_trig_shift : int | None
        Integer to shift response triggers by. None ignores response triggers.
    scale : float
        The scaling factor for EEG data. Units are in volts. Default scale
        factor is 1.. For microvolts, the scale factor would be 1e-6. This is
        used when the header file does not specify the scale factor.

    Returns
    -------
    info : Info
        The measurement info.
    fmt : str
        The data format in the file.
    edf_info : dict
        A dict containing Brain Vision specific parameters.
    events : array, shape (n_events, 3)
        Events from the corresponding vmrk file.
    """
    scale = float(scale)
    info = _empty_info()

    ext = os.path.splitext(vhdr_fname)[-1]
    if ext != '.vhdr':
        raise IOError("The header file must be given to read the data, "
                      "not the '%s' file." % ext)
    with open(vhdr_fname, 'r') as f:
        # extract the first section to resemble a cfg
        l = f.readline().strip()
        assert l == 'Brain Vision Data Exchange Header File Version 1.0'
        settings = f.read()

    if settings.find('[Comment]') != -1:
        params, settings = settings.split('[Comment]')
    else:
        params, settings = settings, ''
    cfg = configparser.ConfigParser()
    if hasattr(cfg, 'read_file'):  # newer API
        cfg.read_file(StringIO(params))
    else:
        cfg.readfp(StringIO(params))

    # get sampling info
    # Sampling interval is given in microsec
    info['sfreq'] = 1e6 / cfg.getfloat('Common Infos', 'SamplingInterval')

    # check binary format
    assert cfg.get('Common Infos', 'DataFormat') == 'BINARY'
    order = cfg.get('Common Infos', 'DataOrientation')
    if order not in _orientation_dict:
        raise NotImplementedError('Data Orientation %s is not supported'
                                  % order)
    order = _orientation_dict[order]

    fmt = cfg.get('Binary Infos', 'BinaryFormat')
    if fmt not in _fmt_dict:
        raise NotImplementedError('Datatype %s is not supported' % fmt)
    fmt = _fmt_dict[fmt]

    # load channel labels
    info['nchan'] = cfg.getint('Common Infos', 'NumberOfChannels') + 1
    ch_names = [''] * info['nchan']
    cals = np.empty(info['nchan'])
    ranges = np.empty(info['nchan'])
    cals.fill(np.nan)
    for chan, props in cfg.items('Channel Infos'):
        n = int(re.findall(r'ch(\d+)', chan)[0]) - 1
        props = props.split(',')
        if len(props) < 4:
            props += ('V',)
        name, _, resolution, unit = props[:4]
        ch_names[n] = name
        if resolution == "":  # For truncated vhdrs (e.g. EEGLAB export)
            resolution = 0.000001
        unit = unit.replace('\xc2', '')  # Remove unwanted control characters
        cals[n] = float(resolution)
        ranges[n] = _unit_dict.get(u(unit), unit) * scale
    ch_names[-1] = 'STI 014'
    cals[-1] = 1.
    ranges[-1] = 1.
    if np.isnan(cals).any():
        raise RuntimeError('Missing channel units')

    # Attempts to extract filtering info from header. If not found, both are
    # set to zero.
    settings = settings.splitlines()
    idx = None
    if 'Channels' in settings:
        idx = settings.index('Channels')
        settings = settings[idx + 1:]
        for idx, setting in enumerate(settings):
            if re.match('#\s+Name', setting):
                break
            else:
                idx = None

    if idx:
        lowpass = []
        highpass = []
        for i, ch in enumerate(ch_names[:-1], 1):
            line = settings[idx + i].split()
            assert ch in line
            highpass.append(line[5])
            lowpass.append(line[6])
        if len(highpass) == 0:
            info['highpass'] = None
        elif all(highpass):
            if highpass[0] == 'NaN':
                info['highpass'] = None
            elif highpass[0] == 'DC':
                info['highpass'] = 0.
            else:
                info['highpass'] = float(highpass[0])
        else:
            info['highpass'] = np.min(np.array(highpass, dtype=np.float))
            warnings.warn('%s' % ('Channels contain different highpass '
                                  'filters. Highest filter setting will '
                                  'be stored.'))
        if len(lowpass) == 0:
            info['lowpass'] = None
        elif all(lowpass):
            if lowpass[0] == 'NaN':
                info['lowpass'] = None
            else:
                info['lowpass'] = float(lowpass[0])
        else:
            info['lowpass'] = np.min(np.array(lowpass, dtype=np.float))
            warnings.warn('%s' % ('Channels contain different lowpass filters.'
                                  ' Lowest filter setting will be stored.'))

        # Post process highpass and lowpass to take into account units
        header = settings[idx].split('  ')
        header = [h for h in header if len(h)]
        if '[s]' in header[4] and info['highpass'] is not None \
                and (info['highpass'] > 0):
            info['highpass'] = 1. / info['highpass']
        if '[s]' in header[5] and info['lowpass'] is not None:
            info['lowpass'] = 1. / info['lowpass']
    else:
        info['highpass'] = None
        info['lowpass'] = None

    # locate EEG and marker files
    path = os.path.dirname(vhdr_fname)
    info['filename'] = os.path.join(path, cfg.get('Common Infos', 'DataFile'))
    info['meas_date'] = int(time.time())

    # Creates a list of dicts of eeg channels for raw.info
    logger.info('Setting channel info structure...')
    info['chs'] = []
    info['ch_names'] = ch_names
    for idx, ch_name in enumerate(ch_names):
        if ch_name in eog or idx in eog or idx - info['nchan'] in eog:
            kind = FIFF.FIFFV_EOG_CH
            coil_type = FIFF.FIFFV_COIL_NONE
            unit = FIFF.FIFF_UNIT_V
        elif ch_name in misc or idx in misc or idx - info['nchan'] in misc:
            kind = FIFF.FIFFV_MISC_CH
            coil_type = FIFF.FIFFV_COIL_NONE
            unit = FIFF.FIFF_UNIT_V
        elif ch_name == 'STI 014':
            kind = FIFF.FIFFV_STIM_CH
            coil_type = FIFF.FIFFV_COIL_NONE
            unit = FIFF.FIFF_UNIT_NONE
        else:
            kind = FIFF.FIFFV_EEG_CH
            coil_type = FIFF.FIFFV_COIL_EEG
            unit = FIFF.FIFF_UNIT_V
        info['chs'].append(dict(
            ch_name=ch_name, coil_type=coil_type, kind=kind, logno=idx + 1,
            scanno=idx + 1, cal=cals[idx], range=ranges[idx], loc=np.zeros(12),
            unit=unit, unit_mul=0.,  # always zero- mne manual pg. 273
            coord_frame=FIFF.FIFFV_COORD_HEAD))

    # for stim channel
    marker_id = os.path.join(path, cfg.get('Common Infos', 'MarkerFile'))
    events = _read_vmrk_events(marker_id, response_trig_shift)
    info._check_consistency()
    return info, fmt, order, events


def read_raw_brainvision(vhdr_fname, montage=None,
                         eog=('HEOGL', 'HEOGR', 'VEOGb'), misc=(),
                         reference=None, scale=1., preload=False,
                         response_trig_shift=0, verbose=None):
    """Reader for Brain Vision EEG file

    Parameters
    ----------
    vhdr_fname : str
        Path to the EEG header file.
    montage : str | None | instance of Montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    eog : list or tuple of str
        Names of channels or list of indices that should be designated
        EOG channels. Values should correspond to the vhdr file
        Default is ``('HEOGL', 'HEOGR', 'VEOGb')``.
    misc : list or tuple of str
        Names of channels or list of indices that should be designated
        MISC channels. Values should correspond to the electrodes
        in the vhdr file. Default is ``()``.
    reference : None | str
        **Deprecated**, use `add_reference_channel` instead.
        Name of the electrode which served as the reference in the recording.
        If a name is provided, a corresponding channel is added and its data
        is set to 0. This is useful for later re-referencing. The name should
        correspond to a name in elp_names. Data must be preloaded.
    scale : float
        The scaling factor for EEG data. Units are in volts. Default scale
        factor is 1. For microvolts, the scale factor would be 1e-6. This is
        used when the header file does not specify the scale factor.
    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.
    response_trig_shift : int | None
        An integer that will be added to all response triggers when reading
        events (stimulus triggers will be unaffected). If None, response
        triggers will be ignored. Default is 0 for backwards compatibility, but
        typically another value or None will be necessary.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : instance of RawBrainVision
        A Raw object containing BrainVision data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    raw = RawBrainVision(vhdr_fname=vhdr_fname, montage=montage, eog=eog,
                         misc=misc, reference=reference, scale=scale,
                         preload=preload, verbose=verbose,
                         response_trig_shift=response_trig_shift)
    return raw
