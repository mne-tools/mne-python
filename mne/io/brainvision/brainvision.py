# -*- coding: utf-8 -*-
"""Conversion tool from Brain Vision EEG to FIF"""

# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Christian Brodbeck <christianbrodbeck@nyu.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os
import re
import time

import numpy as np

from ...utils import verbose, logger, warn, deprecated
from ..constants import FIFF
from ..meas_info import _empty_info
from ..base import _BaseRaw, _check_update_montage
from ..utils import _read_segments_file, _synthesize_stim_channel

from ...externals.six import StringIO
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
    event_id : dict | None
        The id of special events to consider in addition to those that
        follow the normal Brainvision trigger format ('SXXX').
        If dict, the keys will be mapped to trigger values on the stimulus
        channel. Example: {'SyncStatus': 1; 'Pulse Artifact': 3}. If None
        or an empty dict (default), only stimulus events are added to the
        stimulus channel. Keys are case sensitive.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    @verbose
    def __init__(self, vhdr_fname, montage=None,
                 eog=('HEOGL', 'HEOGR', 'VEOGb'), misc=(),
                 scale=1., preload=False, response_trig_shift=0,
                 event_id=None, verbose=None):
        # Channel info and events
        logger.info('Extracting parameters from %s...' % vhdr_fname)
        vhdr_fname = os.path.abspath(vhdr_fname)
        info, fmt, self._order, mrk_fname, montage = _get_vhdr_info(
            vhdr_fname, eog, misc, scale, montage)
        events = _read_vmrk_events(mrk_fname, event_id, response_trig_shift)
        _check_update_montage(info, montage)
        with open(info['filename'], 'rb') as f:
            f.seek(0, os.SEEK_END)
            n_samples = f.tell()
        dtype_bytes = _fmt_byte_dict[fmt]
        self.preload = False  # so the event-setting works
        last_samps = [(n_samples // (dtype_bytes * (info['nchan'] - 1))) - 1]
        self._create_event_ch(events, last_samps[0] + 1)
        super(RawBrainVision, self).__init__(
            info, last_samps=last_samps, filenames=[info['filename']],
            orig_format=fmt, preload=preload, verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data"""
        # read data
        dtype = _fmt_dtype_dict[self.orig_format]
        n_data_ch = len(self.ch_names) - 1
        _read_segments_file(self, data, idx, fi, start, stop, cals, mult,
                            dtype=dtype, n_channels=n_data_ch,
                            trigger_ch=self._event_ch)

    @deprecated('get_brainvision_events is deprecated and will be removed '
                'in 0.13, use mne.find_events(raw, "STI014") to get properly '
                'formatted events instead')
    def get_brainvision_events(self):
        """Retrieve the events associated with the Brain Vision Raw object

        Returns
        -------
        events : array, shape (n_events, 3)
            Events, each row consisting of an (onset, duration, trigger)
            sequence.
        """
        return self._get_brainvision_events()

    def _get_brainvision_events(self):
        """Retrieve the events associated with the Brain Vision Raw object

        Returns
        -------
        events : array, shape (n_events, 3)
            Events, each row consisting of an (onset, duration, trigger)
            sequence.
        """
        return self._events.copy()

    @deprecated('set_brainvision_events is deprecated and will be removed '
                'in 0.13')
    def set_brainvision_events(self, events):
        """Set the events and update the synthesized stim channel

        Parameters
        ----------
        events : array, shape (n_events, 3)
            Events, each row consisting of an (onset, duration, trigger)
            sequence.
        """
        return self._set_brainvision_events(events)

    def _set_brainvision_events(self, events):
        """Set the events and update the synthesized stim channel

        Parameters
        ----------
        events : array, shape (n_events, 3)
            Events, each row consisting of an (onset, duration, trigger)
            sequence.
        """
        self._create_event_ch(events)

    def _create_event_ch(self, events, n_samp=None):
        """Create the event channel"""
        if n_samp is None:
            n_samp = self.last_samp - self.first_samp + 1
        events = np.array(events, int)
        if events.ndim != 2 or events.shape[1] != 3:
            raise ValueError("[n_events x 3] shaped array required")
        # update events
        self._event_ch = _synthesize_stim_channel(events, n_samp)
        self._events = events
        if self.preload:
            self._data[-1] = self._event_ch


def _read_vmrk_events(fname, event_id=None, response_trig_shift=0):
    """Read events from a vmrk file

    Parameters
    ----------
    fname : str
        vmrk file to be read.
    event_id : dict | None
        The id of special events to consider in addition to those that
        follow the normal Brainvision trigger format ('SXXX').
        If dict, the keys will be mapped to trigger values on the stimulus
        channel. Example: {'SyncStatus': 1; 'Pulse Artifact': 3}. If None
        or an empty dict (default), only stimulus events are added to the
        stimulus channel. Keys are case sensitive.
    response_trig_shift : int | None
        Integer to shift response triggers by. None ignores response triggers.

    Returns
    -------
    events : array, shape (n_events, 3)
        An array containing the whole recording's events, each row representing
        an event as (onset, duration, trigger) sequence.
    """
    if event_id is None:
        event_id = dict()
    # read vmrk file
    with open(fname, 'rb') as fid:
        txt = fid.read().decode('utf-8')

    header = txt.split('\n')[0].strip()
    _check_mrk_version(header)
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
    events, dropped = list(), list()
    for info in items:
        mtype, mdesc, onset, duration = info.split(',')[:4]
        onset = int(onset)
        duration = (int(duration) if duration.isdigit() else 1)
        if mdesc in event_id:
            trigger = event_id[mdesc]
        else:
            try:
                trigger = int(re.findall('[A-Za-z]*\s*?(\d+)', mdesc)[0])
            except IndexError:
                trigger = None
            if mtype.lower().startswith('response'):
                if response_trig_shift is not None:
                    trigger += response_trig_shift
                else:
                    trigger = None
        if trigger:
            events.append((onset, duration, trigger))
        else:
            if len(mdesc) > 0:
                dropped.append(mdesc)

    if len(dropped) > 0:
        dropped = list(set(dropped))
        examples = ", ".join(dropped[:5])
        if len(dropped) > 5:
            examples += ", ..."
        warn("Currently, {0} trigger(s) will be dropped, such as [{1}]. "
             "Consider using ``event_id`` to parse triggers that "
             "do not follow the 'SXXX' pattern.".format(
                 len(dropped), examples))

    events = np.array(events).reshape(-1, 3)
    return events


def _check_hdr_version(header):
    tags = ['Brain Vision Data Exchange Header File Version 1.0',
            'Brain Vision Data Exchange Header File Version 2.0']
    if header not in tags:
        raise ValueError("Currently only support %r, not %r"
                         "Contact MNE-Developers for support."
                         % (str(tags), header))


def _check_mrk_version(header):
    tags = ['Brain Vision Data Exchange Marker File, Version 1.0',
            'Brain Vision Data Exchange Marker File, Version 2.0']
    if header not in tags:
        raise ValueError("Currently only support %r, not %r"
                         "Contact MNE-Developers for support."
                         % (str(tags), header))


_orientation_dict = dict(MULTIPLEXED='F', VECTORIZED='C')
_fmt_dict = dict(INT_16='short', INT_32='int', IEEE_FLOAT_32='single')
_fmt_byte_dict = dict(short=2, int=4, single=4)
_fmt_dtype_dict = dict(short='<i2', int='<i4', single='<f4')
_unit_dict = {'V': 1., u'µV': 1e-6, 'uV': 1e-6}


def _get_vhdr_info(vhdr_fname, eog, misc, scale, montage):
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
    scale : float
        The scaling factor for EEG data. Units are in volts. Default scale
        factor is 1.. For microvolts, the scale factor would be 1e-6. This is
        used when the header file does not specify the scale factor.
    montage : str | True | None | instance of Montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.

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

    ext = os.path.splitext(vhdr_fname)[-1]
    if ext != '.vhdr':
        raise IOError("The header file must be given to read the data, "
                      "not the '%s' file." % ext)
    with open(vhdr_fname, 'rb') as f:
        # extract the first section to resemble a cfg
        header = f.readline().decode('utf-8').strip()
        _check_hdr_version(header)
        settings = f.read().decode('utf-8')

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
    sfreq = 1e6 / cfg.getfloat('Common Infos', 'SamplingInterval')
    info = _empty_info(sfreq)

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
    nchan = cfg.getint('Common Infos', 'NumberOfChannels') + 1
    ch_names = [''] * nchan
    cals = np.empty(nchan)
    ranges = np.empty(nchan)
    cals.fill(np.nan)
    ch_dict = dict()
    for chan, props in cfg.items('Channel Infos'):
        n = int(re.findall(r'ch(\d+)', chan)[0]) - 1
        props = props.split(',')
        if len(props) < 4:
            props += ('V',)
        name, _, resolution, unit = props[:4]
        ch_dict[chan] = name
        ch_names[n] = name
        if resolution == "":
            if not(unit):  # For truncated vhdrs (e.g. EEGLAB export)
                resolution = 0.000001
            else:
                resolution = 1.  # for files with units specified, but not res
        unit = unit.replace(u'\xc2', u'')  # Remove unwanted control characters
        cals[n] = float(resolution)
        ranges[n] = _unit_dict.get(unit, unit) * scale

    # create montage
    if montage is True:
        from ...transforms import _sphere_to_cartesian
        from ...channels.montage import Montage
        montage_pos = list()
        montage_names = list()
        for ch in cfg.items('Coordinates'):
            montage_names.append(ch_dict[ch[0]])
            radius, theta, phi = map(float, ch[1].split(','))
            # 1: radius, 2: theta, 3: phi
            pos = _sphere_to_cartesian(r=radius, theta=theta, phi=phi)
            montage_pos.append(pos)
        montage_sel = np.arange(len(montage_pos))
        montage = Montage(montage_pos, montage_names, 'Brainvision',
                          montage_sel)

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
            pass
        elif all(highpass):
            if highpass[0] == 'NaN':
                pass  # Placeholder for future use. Highpass set in _empty_info
            elif highpass[0] == 'DC':
                info['highpass'] = 0.
            else:
                info['highpass'] = float(highpass[0])
        else:
            info['highpass'] = np.min(np.array(highpass, dtype=np.float))
            warn('Channels contain different highpass filters. Highest filter '
                 'setting will be stored.')
        if len(lowpass) == 0:
            pass
        elif all(lowpass):
            if lowpass[0] == 'NaN':
                pass  # Placeholder for future use. Lowpass set in _empty_info
            else:
                info['lowpass'] = float(lowpass[0])
        else:
            info['lowpass'] = np.min(np.array(lowpass, dtype=np.float))
            warn('Channels contain different lowpass filters. Lowest filter '
                 'setting will be stored.')

        # Post process highpass and lowpass to take into account units
        header = settings[idx].split('  ')
        header = [h for h in header if len(h)]
        if '[s]' in header[4] and (info['highpass'] > 0):
            info['highpass'] = 1. / info['highpass']
        if '[s]' in header[5]:
            info['lowpass'] = 1. / info['lowpass']

    # locate EEG and marker files
    path = os.path.dirname(vhdr_fname)
    info['filename'] = os.path.join(path, cfg.get('Common Infos', 'DataFile'))
    info['meas_date'] = int(time.time())
    info['buffer_size_sec'] = 1.  # reasonable default

    # Creates a list of dicts of eeg channels for raw.info
    logger.info('Setting channel info structure...')
    info['chs'] = []
    for idx, ch_name in enumerate(ch_names):
        if ch_name in eog or idx in eog or idx - nchan in eog:
            kind = FIFF.FIFFV_EOG_CH
            coil_type = FIFF.FIFFV_COIL_NONE
            unit = FIFF.FIFF_UNIT_V
        elif ch_name in misc or idx in misc or idx - nchan in misc:
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
    mrk_fname = os.path.join(path, cfg.get('Common Infos', 'MarkerFile'))
    info._update_redundant()
    info._check_consistency()
    return info, fmt, order, mrk_fname, montage


def read_raw_brainvision(vhdr_fname, montage=None,
                         eog=('HEOGL', 'HEOGR', 'VEOGb'), misc=(),
                         scale=1., preload=False, response_trig_shift=0,
                         event_id=None, verbose=None):
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
    event_id : dict | None
        The id of special events to consider in addition to those that
        follow the normal Brainvision trigger format ('SXXX').
        If dict, the keys will be mapped to trigger values on the stimulus
        channel. Example: {'SyncStatus': 1; 'Pulse Artifact': 3}. If None
        or an empty dict (default), only stimulus events are added to the
        stimulus channel. Keys are case sensitive.
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
                         misc=misc, scale=scale,
                         preload=preload, verbose=verbose, event_id=event_id,
                         response_trig_shift=response_trig_shift)
    return raw
