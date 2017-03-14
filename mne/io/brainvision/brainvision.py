# -*- coding: utf-8 -*-
"""Conversion tool from Brain Vision EEG to FIF."""

# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Christian Brodbeck <christianbrodbeck@nyu.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Phillip Alday <phillip.alday@unisa.edu.au>
#
# License: BSD (3-clause)

import os
import re
import time

import numpy as np

from ...utils import verbose, logger, warn
from ..constants import FIFF
from ..meas_info import _empty_info
from ..base import BaseRaw, _check_update_montage
from ..utils import (_read_segments_file, _synthesize_stim_channel,
                     _mult_cal_one)

from ...externals.six import StringIO, string_types
from ...externals.six.moves import configparser


class RawBrainVision(BaseRaw):
    """Raw object from Brain Vision EEG file.

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
    misc : list or tuple of str | 'auto'
        Names of channels or list of indices that should be designated
        MISC channels. Values should correspond to the electrodes
        in the vhdr file. If 'auto', units in vhdr file are used for inferring
        misc channels. Default is ``'auto'``.
    scale : float
        The scaling factor for EEG data. Unless specified otherwise by
        header file, units are in microvolts. Default scale factor is 1.
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
        follow the normal Brainvision trigger format ('S###').
        If dict, the keys will be mapped to trigger values on the stimulus
        channel. Example: {'SyncStatus': 1; 'Pulse Artifact': 3}. If None
        or an empty dict (default), only stimulus events are added to the
        stimulus channel. Keys are case sensitive.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, vhdr_fname, montage=None,
                 eog=('HEOGL', 'HEOGR', 'VEOGb'), misc='auto',
                 scale=1., preload=False, response_trig_shift=0,
                 event_id=None, verbose=None):  # noqa: D102
        # Channel info and events
        logger.info('Extracting parameters from %s...' % vhdr_fname)
        vhdr_fname = os.path.abspath(vhdr_fname)
        info, data_filename, fmt, self._order, mrk_fname, montage = \
            _get_vhdr_info(vhdr_fname, eog, misc, scale, montage)
        events = _read_vmrk_events(mrk_fname, event_id, response_trig_shift)
        _check_update_montage(info, montage)
        with open(data_filename, 'rb') as f:
            if isinstance(fmt, dict):  # ASCII, this will be slow :(
                n_skip = 0
                for ii in range(int(fmt['skiplines'])):
                    n_skip += len(f.readline())
                offsets = np.cumsum([n_skip] + [len(line) for line in f])
                n_samples = len(offsets) - 1
            else:
                f.seek(0, os.SEEK_END)
                n_samples = f.tell()
                dtype_bytes = _fmt_byte_dict[fmt]
                offsets = None
                n_samples = n_samples // (dtype_bytes * (info['nchan'] - 1))
        self.preload = False  # so the event-setting works
        self._create_event_ch(events, n_samples)
        super(RawBrainVision, self).__init__(
            info, last_samps=[n_samples - 1], filenames=[data_filename],
            orig_format=fmt, preload=preload, verbose=verbose,
            raw_extras=[offsets])

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        # read data
        if isinstance(self.orig_format, string_types):
            dtype = _fmt_dtype_dict[self.orig_format]
            n_data_ch = len(self.ch_names) - 1
            _read_segments_file(self, data, idx, fi, start, stop, cals, mult,
                                dtype=dtype, n_channels=n_data_ch,
                                trigger_ch=self._event_ch)
        else:
            offsets = self._raw_extras[fi]
            with open(self._filenames[fi], 'rb') as fid:
                fid.seek(offsets[start])
                block = np.empty((len(self.ch_names), stop - start))
                for ii in range(stop - start):
                    line = fid.readline().decode('ASCII')
                    line = line.strip().replace(',', '.').split()
                    block[:-1, ii] = list(map(float, line))
            block[-1] = self._event_ch[start:stop]
            _mult_cal_one(data, block, idx, cals, mult)

    def _get_brainvision_events(self):
        """Retrieve the events associated with the Brain Vision Raw object.

        Returns
        -------
        events : array, shape (n_events, 3)
            Events, each row consisting of an (onset, duration, trigger)
            sequence.
        """
        return self._events.copy()

    def _set_brainvision_events(self, events):
        """Set the events and update the synthesized stim channel.

        Parameters
        ----------
        events : array, shape (n_events, 3)
            Events, each row consisting of an (onset, duration, trigger)
            sequence.
        """
        self._create_event_ch(events)

    def _create_event_ch(self, events, n_samp=None):
        """Create the event channel."""
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
    """Read events from a vmrk file.

    Parameters
    ----------
    fname : str
        vmrk file to be read.
    event_id : dict | None
        The id of special events to consider in addition to those that
        follow the normal Brainvision trigger format ('S###').
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
        txt = fid.read()

    # we don't actually need to know the coding for the header line.
    # the characters in it all belong to ASCII and are thus the
    # same in Latin-1 and UTF-8
    header = txt.decode('ascii', 'ignore').split('\n')[0].strip()
    _check_mrk_version(header)
    if (response_trig_shift is not None and
            not isinstance(response_trig_shift, int)):
        raise TypeError("response_trig_shift must be an integer or None")

    # although the markers themselves are guaranteed to be ASCII (they
    # consist of numbers and a few reserved words), we should still
    # decode the file properly here because other (currently unused)
    # blocks, such as that the filename are specifying are not
    # guaranteed to be ASCII.

    codepage = 'utf-8'
    try:
        # if there is an explicit codepage set, use it
        # we pretend like it's ascii when searching for the codepage
        cp_setting = re.search('Codepage=(.+)',
                               txt.decode('ascii', 'ignore'),
                               re.IGNORECASE & re.MULTILINE)
        if cp_setting:
            codepage = cp_setting.group(1).strip()
        txt = txt.decode(codepage)
    except UnicodeDecodeError:
        # if UTF-8 (new standard) or explicit codepage setting fails,
        # fallback to Latin-1, which is Windows default and implicit
        # standard in older recordings
        txt = txt.decode('latin-1')

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
             "do not follow the 'S###' pattern.".format(
                 len(dropped), examples))

    events = np.array(events).reshape(-1, 3)
    return events


def _check_hdr_version(header):
    """Check the header version."""
    tags = ['Brain Vision Data Exchange Header File Version 1.0',
            'Brain Vision Data Exchange Header File Version 2.0']
    if header not in tags:
        raise ValueError("Currently only support %r, not %r"
                         "Contact MNE-Developers for support."
                         % (str(tags), header))


def _check_mrk_version(header):
    """Check the marker version."""
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
_unit_dict = {'V': 1.,  # V stands for Volt
              u'µV': 1e-6,
              'uV': 1e-6,
              'C': 1,  # C stands for celsius
              u'µS': 1e-6,  # S stands for Siemens
              u'uS': 1e-6,
              u'ARU': 1,  # ARU is the unity for the breathing data
              'S': 1,
              'N': 1}  # Newton


def _get_vhdr_info(vhdr_fname, eog, misc, scale, montage):
    """Extract all the information from the header file.

    Parameters
    ----------
    vhdr_fname : str
        Raw EEG header to be read.
    eog : list of str
        Names of channels that should be designated EOG channels. Names should
        correspond to the vhdr file.
    misc : list or tuple of str | 'auto'
        Names of channels or list of indices that should be designated
        MISC channels. Values should correspond to the electrodes
        in the vhdr file. If 'auto', units in vhdr file are used for inferring
        misc channels. Default is ``'auto'``.
    scale : float
        The scaling factor for EEG data. Unless specified otherwise by
        header file, units are in microvolts. Default scale factor is 1.
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
                      "not a file with extension '%s'." % ext)
    with open(vhdr_fname, 'rb') as f:
        # extract the first section to resemble a cfg
        header = f.readline()
        codepage = 'utf-8'
        # we don't actually need to know the coding for the header line.
        # the characters in it all belong to ASCII and are thus the
        # same in Latin-1 and UTF-8
        header = header.decode('ascii', 'ignore').strip()
        _check_hdr_version(header)

        settings = f.read()
        try:
            # if there is an explicit codepage set, use it
            # we pretend like it's ascii when searching for the codepage
            cp_setting = re.search('Codepage=(.+)',
                                   settings.decode('ascii', 'ignore'),
                                   re.IGNORECASE & re.MULTILINE)
            if cp_setting:
                codepage = cp_setting.group(1).strip()
            settings = settings.decode(codepage)
        except UnicodeDecodeError:
            # if UTF-8 (new standard) or explicit codepage setting fails,
            # fallback to Latin-1, which is Windows default and implicit
            # standard in older recordings
            settings = settings.decode('latin-1')

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

    order = cfg.get('Common Infos', 'DataOrientation')
    if order not in _orientation_dict:
        raise NotImplementedError('Data Orientation %s is not supported'
                                  % order)
    order = _orientation_dict[order]

    data_format = cfg.get('Common Infos', 'DataFormat')
    if data_format == 'BINARY':
        fmt = cfg.get('Binary Infos', 'BinaryFormat')
        if fmt not in _fmt_dict:
            raise NotImplementedError('Datatype %s is not supported' % fmt)
        fmt = _fmt_dict[fmt]
    else:
        fmt = dict((key, cfg.get('ASCII Infos', key))
                   for key in cfg.options('ASCII Infos'))

    # load channel labels
    nchan = cfg.getint('Common Infos', 'NumberOfChannels') + 1
    ch_names = [''] * nchan
    cals = np.empty(nchan)
    ranges = np.empty(nchan)
    cals.fill(np.nan)
    ch_dict = dict()
    misc_chs = dict()
    for chan, props in cfg.items('Channel Infos'):
        n = int(re.findall(r'ch(\d+)', chan)[0]) - 1
        props = props.split(',')
        # default to microvolts because that's what the older brainvision
        # standard explicitly assumed; the unit is only allowed to be
        # something else if explicitly stated (cf. EEGLAB export below)
        if len(props) < 4:
            props += (u'µV',)
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
        ranges[n] = _unit_dict.get(unit, 1) * scale
        if unit not in ('V', u'µV', 'uV'):
            misc_chs[name] = (FIFF.FIFF_UNIT_CEL if unit == 'C'
                              else FIFF.FIFF_UNIT_NONE)
    misc = list(misc_chs.keys()) if misc == 'auto' else misc

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
        hp_col, lp_col = 4, 5
        for idx, setting in enumerate(settings):
            if re.match('#\s+Name', setting):
                break
            else:
                idx = None

    # If software filters are active, then they override the hardware setup
    # But we still want to be able to double check the channel names
    # for alignment purposes, we keep track of the hardware setting idx
    idx_amp = idx

    if 'S o f t w a r e  F i l t e r s' in settings:
        idx = settings.index('S o f t w a r e  F i l t e r s')
        for idx, setting in enumerate(settings[idx + 1:], idx + 1):
            if re.match('#\s+Low Cutoff', setting):
                hp_col, lp_col = 1, 2
                warn('Online software filter detected. Using software '
                     'filter settings and ignoring hardware values')
                break
            else:
                idx = idx_amp

    if idx:
        lowpass = []
        highpass = []

        # extract filter units and convert s to Hz if necessary
        # this cannot be done as post-processing as the inverse t-f
        # relationship means that the min/max comparisons don't make sense
        # unless we know the units
        header = re.split('\s\s+', settings[idx])
        hp_s = '[s]' in header[hp_col]
        lp_s = '[s]' in header[lp_col]

        for i, ch in enumerate(ch_names[:-1], 1):
            line = re.split('\s\s+', settings[idx + i])
            # double check alignment with channel by using the hw settings
            # the actual divider is multiple spaces -- for newer BV
            # files, the unit is specified for every channel separated
            # by a single space, while for older files, the unit is
            # specified in the column headers
            if idx == idx_amp:
                line_amp = line
            else:
                line_amp = re.split('\s\s+', settings[idx_amp + i])
            assert ch in line_amp
            highpass.append(line[hp_col])
            lowpass.append(line[lp_col])
        if len(highpass) == 0:
            pass
        elif len(set(highpass)) == 1:
            if highpass[0] in ('NaN', 'Off'):
                pass  # Placeholder for future use. Highpass set in _empty_info
            elif highpass[0] == 'DC':
                info['highpass'] = 0.
            else:
                info['highpass'] = float(highpass[0])
                if hp_s:
                    info['highpass'] = 1. / info['highpass']
        else:
            heterogeneous_hp_filter = True
            if hp_s:
                # We convert channels with disabled filters to having
                # highpass relaxed / no filters
                highpass = [float(filt) if filt not in ('NaN', 'Off', 'DC')
                            else np.Inf for filt in highpass]
                info['highpass'] = np.max(np.array(highpass, dtype=np.float))
                # Coveniently enough 1 / np.Inf = 0.0, so this works for
                # DC / no highpass filter
                info['highpass'] = 1. / info['highpass']

                # not exactly the cleanest use of FP, but this makes us
                # more conservative in *not* warning.
                if info['highpass'] == 0.0 and len(set(highpass)) == 1:
                    # not actually heterogeneous in effect
                    # ... just heterogeneously disabled
                    heterogeneous_hp_filter = False
            else:
                highpass = [float(filt) if filt not in ('NaN', 'Off', 'DC')
                            else 0.0 for filt in highpass]
                info['highpass'] = np.min(np.array(highpass, dtype=np.float))
                if info['highpass'] == 0.0 and len(set(highpass)) == 1:
                    # not actually heterogeneous in effect
                    # ... just heterogeneously disabled
                    heterogeneous_hp_filter = False

            if heterogeneous_hp_filter:
                warn('Channels contain different highpass filters. '
                     'Lowest (weakest) filter setting (%0.2f Hz) '
                     'will be stored.' % info['highpass'])

        if len(lowpass) == 0:
            pass
        elif len(set(lowpass)) == 1:
            if lowpass[0] in ('NaN', 'Off'):
                pass  # Placeholder for future use. Lowpass set in _empty_info
            else:
                info['lowpass'] = float(lowpass[0])
                if lp_s:
                    info['lowpass'] = 1. / info['lowpass']
        else:
            heterogeneous_lp_filter = True
            if lp_s:
                # We convert channels with disabled filters to having
                # infinitely relaxed / no filters
                lowpass = [float(filt) if filt not in ('NaN', 'Off')
                           else 0.0 for filt in lowpass]
                info['lowpass'] = np.min(np.array(lowpass, dtype=np.float))
                try:
                    info['lowpass'] = 1. / info['lowpass']
                except ZeroDivisionError:
                    if len(set(lowpass)) == 1:
                        # No lowpass actually set for the weakest setting
                        # so we set lowpass to the Nyquist frequency
                        info['lowpass'] = info['sfreq'] / 2.
                        # not actually heterogeneous in effect
                        # ... just heterogeneously disabled
                        heterogeneous_lp_filter = False
                    else:
                        # no lowpass filter is the weakest filter,
                        # but it wasn't the only filter
                        pass
            else:
                # We convert channels with disabled filters to having
                # infinitely relaxed / no filters
                lowpass = [float(filt) if filt not in ('NaN', 'Off')
                           else np.Inf for filt in lowpass]
                info['lowpass'] = np.max(np.array(lowpass, dtype=np.float))

                if np.isinf(info['lowpass']):
                    # No lowpass actually set for the weakest setting
                    # so we set lowpass to the Nyquist frequency
                    info['lowpass'] = info['sfreq'] / 2.
                    if len(set(lowpass)) == 1:
                        # not actually heterogeneous in effect
                        # ... just heterogeneously disabled
                        heterogeneous_lp_filter = False

            if heterogeneous_lp_filter:
                # this isn't clean FP, but then again, we only want to provide
                # the Nyquist hint when the lowpass filter was actually
                # calculated from dividing the sampling frequency by 2, so the
                # exact/direct comparison (instead of tolerance) makes sense
                if info['lowpass'] == info['sfreq'] / 2.0:
                    nyquist = ', Nyquist limit'
                else:
                    nyquist = ""
                warn('Channels contain different lowpass filters. '
                     'Highest (weakest) filter setting (%0.2f Hz%s) '
                     'will be stored.' % (info['lowpass'], nyquist))

    # locate EEG and marker files
    path = os.path.dirname(vhdr_fname)
    data_filename = os.path.join(path, cfg.get('Common Infos', 'DataFile'))
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
            if ch_name in misc_chs:
                unit = misc_chs[ch_name]
            else:
                unit = FIFF.FIFF_UNIT_NONE
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
    return info, data_filename, fmt, order, mrk_fname, montage


def read_raw_brainvision(vhdr_fname, montage=None,
                         eog=('HEOGL', 'HEOGR', 'VEOGb'), misc='auto',
                         scale=1., preload=False, response_trig_shift=0,
                         event_id=None, verbose=None):
    """Reader for Brain Vision EEG file.

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
    misc : list or tuple of str | 'auto'
        Names of channels or list of indices that should be designated
        MISC channels. Values should correspond to the electrodes
        in the vhdr file. If 'auto', units in vhdr file are used for inferring
        misc channels. Default is ``'auto'``.
    scale : float
        The scaling factor for EEG data. Unless specified otherwise by
        header file, units are in microvolts. Default scale factor is 1.
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
        follow the normal Brainvision trigger format ('S###').
        If dict, the keys will be mapped to trigger values on the stimulus
        channel. Example: {'SyncStatus': 1; 'Pulse Artifact': 3}. If None
        or an empty dict (default), only stimulus events are added to the
        stimulus channel. Keys are case sensitive.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raw : instance of RawBrainVision
        A Raw object containing BrainVision data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawBrainVision(vhdr_fname=vhdr_fname, montage=montage, eog=eog,
                          misc=misc, scale=scale, preload=preload,
                          response_trig_shift=response_trig_shift,
                          event_id=event_id, verbose=verbose)
