"""Conversion tool from Brain Vision EEG to FIF"""

# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Christian Brodbeck <christianbrodbeck@nyu.edu>
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
        Default is ('HEOGL', 'HEOGR', 'VEOGb').
    misc : list or tuple
        Names of channels or list of indices that should be designated
        MISC channels. Values should correspond to the electrodes
        in the vhdr file. Default is None.
    reference : None | str
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
                 eog=('HEOGL', 'HEOGR', 'VEOGb'), misc=None, reference=None,
                 scale=1., preload=False, response_trig_shift=0, verbose=None):

        if reference is not None and preload is False:
            raise ValueError("Preload must be set to True if reference is "
                             "specified.")
        # Preliminary Raw attributes
        self._events = np.empty((0, 3))
        self.preload = False

        # Channel info and events
        logger.info('Extracting eeg Parameters from %s...' % vhdr_fname)
        vhdr_fname = os.path.abspath(vhdr_fname)
        if not isinstance(scale, (int, float)):
            raise TypeError('Scale factor must be an int or float. '
                            '%s provided' % type(scale))
        info, self._eeg_info, events = _get_eeg_info(vhdr_fname, eog, misc,
                                                     response_trig_shift)
        self._eeg_info['scale'] = float(scale)
        logger.info('Creating Raw.info structure...')
        _check_update_montage(info, montage)
        with open(info['filename'], 'rb') as f:
            f.seek(0, os.SEEK_END)
            n_samples = f.tell()
        dtype = int(self._eeg_info['dtype'][-1])
        last_samps = [(n_samples //
                      (dtype * self._eeg_info['n_eeg_chan'])) - 1]
        super(RawBrainVision, self).__init__(
            info, last_samps=last_samps, filenames=[vhdr_fname],
            verbose=verbose)
        self.set_brainvision_events(events)

        # load data
        if preload:
            self.preload = preload
            logger.info('Reading raw data from %s...' % vhdr_fname)
            self._data, _ = self._read_segment()
            if reference is not None:
                add_reference_channels(self, reference, copy=False)
            assert len(self._data) == self.info['nchan']
            logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs'
                        % (self.first_samp, self.last_samp,
                           float(self.first_samp) / self.info['sfreq'],
                           float(self.last_samp) / self.info['sfreq']))
        logger.info('Ready.')

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
        data : array, shape (n_channels, n_samples)
           The data.
        times : array, shape (n_samples,)
            returns the time values corresponding to the samples.
        """
        if sel is not None:
            if len(sel) == 1 and sel[0] == 0 and start == 0 and stop == 1:
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

        # assemble channel information
        eeg_info = self._eeg_info
        sfreq = self.info['sfreq']
        chs = self.info['chs']
        units = eeg_info['units']
        if len(self._events):
            chs = chs[:-1]
        n_eeg = len(chs)
        cals = np.atleast_2d([chan_info['cal'] for chan_info in chs])
        cals *= eeg_info['scale'] * units

        logger.info('Reading %d ... %d  =  %9.3f ... %9.3f secs...' %
                    (start, stop - 1, start / float(sfreq),
                     (stop - 1) / float(sfreq)))

        # read data
        dtype = np.dtype(eeg_info['dtype'])
        buffer_size = (stop - start)
        pointer = start * n_eeg * dtype.itemsize

        with open(self.info['filename'], 'rb') as f:
            f.seek(pointer)
            # extract data
            data_buffer = np.fromfile(f, dtype=dtype,
                                      count=buffer_size * n_eeg)
        if eeg_info['data_orientation'] == 'MULTIPLEXED':
            data_buffer = data_buffer.reshape((n_eeg, -1), order='F')
        elif eeg_info['data_orientation'] == 'VECTORIZED':
            data_buffer = data_buffer.reshape((n_eeg, -1), order='C')

        n_channels, n_times = data_buffer.shape
        # Total number of channels
        n_channels += int(len(self._events) > 0)

        # Preallocate data array
        data = np.empty((n_channels, n_times), dtype=np.float64)
        data[:len(data_buffer)] = data_buffer  # cast to float64
        data[:len(data_buffer)] *= cals.T
        ch_idx = len(data_buffer)
        del data_buffer

        # stim channel (if applicable)
        if len(self._events):
            data[ch_idx] = _synthesize_stim_channel(self._events, start, stop)
            ch_idx += 1

        if sel is not None:
            data = data.take(sel, axis=0)

        logger.info('[done]')
        times = np.arange(start, stop, dtype=float) / sfreq

        return data, times

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
        """Set the events (automatically updates the synthesized stim channel)

        Parameters
        ----------
        events : array, shape (n_events, 3)
            Events, each row consisting of an (onset, duration, trigger)
            sequence.
        """
        events = np.copy(events)
        if not events.ndim == 2 and events.shape[1] == 3:
            raise ValueError("[n_events x 3] shaped array required")

        # update info based on presence of stim channel
        had_events = bool(len(self._events))
        has_events = bool(len(events))
        if had_events and not has_events:  # remove stim channel
            if self.info['ch_names'][-1] != 'STI 014':
                err = "Last channel is not stim channel; info was modified"
                raise RuntimeError(err)
            self.info['nchan'] -= 1
            del self.info['ch_names'][-1]
            del self.info['chs'][-1]
            if self.preload:
                self._data = self._data[:-1]
        elif has_events and not had_events:  # add stim channel
            idx = len(self.info['chs']) + 1
            chan_info = {'ch_name': 'STI 014',
                         'kind': FIFF.FIFFV_STIM_CH,
                         'coil_type': FIFF.FIFFV_COIL_NONE,
                         'logno': idx,
                         'scanno': idx,
                         'cal': 1,
                         'range': 1,
                         'unit_mul':  0,
                         'unit': FIFF.FIFF_UNIT_NONE,
                         'eeg_loc': np.zeros(3),
                         'loc': np.zeros(12)}
            self.info['nchan'] += 1
            self.info['ch_names'].append(chan_info['ch_name'])
            self.info['chs'].append(chan_info)
            if self.preload:
                shape = (1, self._data.shape[1])
                self._data = np.vstack((self._data, np.empty(shape)))

        # update events
        self._events = events
        if has_events and self.preload:
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


def _get_eeg_info(vhdr_fname, eog, misc, response_trig_shift):
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

    Returns
    -------
    info : Info
        The measurement info.
    edf_info : dict
        A dict containing Brain Vision specific parameters.
    events : array, shape (n_events, 3)
        Events from the corresponding vmrk file.
    """

    if eog is None:
        eog = []
    if misc is None:
        misc = []
    info = _empty_info()
    info['buffer_size_sec'] = 10.
    info['filename'] = vhdr_fname
    eeg_info = {}

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
        params, settings = settings, str()
    cfg = configparser.ConfigParser()
    if hasattr(cfg, 'read_file'):  # newer API
        cfg.read_file(StringIO(params))
    else:
        cfg.readfp(StringIO(params))

    # get sampling info
    # Sampling interval is given in microsec
    sfreq = 1e6 / cfg.getfloat('Common Infos', 'SamplingInterval')
    sfreq = int(sfreq)
    eeg_info['n_eeg_chan'] = n_eeg_chan = cfg.getint('Common Infos',
                                                     'NumberOfChannels')

    # check binary format
    assert cfg.get('Common Infos', 'DataFormat') == 'BINARY'
    eeg_info['data_orientation'] = cfg.get('Common Infos', 'DataOrientation')
    if not (eeg_info['data_orientation'] == 'MULTIPLEXED' or
            eeg_info['data_orientation'] == 'VECTORIZED'):
        raise NotImplementedError('Data Orientation %s is not supported'
                                  % eeg_info['data_orientation'])

    binary_format = cfg.get('Binary Infos', 'BinaryFormat')
    if binary_format == 'INT_16':
        eeg_info['dtype'] = '<i2'
    elif binary_format == 'INT_32':
        eeg_info['dtype'] = '<i4'
    elif binary_format == 'IEEE_FLOAT_32':
        eeg_info['dtype'] = '<f4'
    else:
        raise NotImplementedError('Datatype %s is not supported'
                                  % binary_format)

    # load channel labels
    ch_names = ['UNKNOWN'] * n_eeg_chan
    cals = np.empty(n_eeg_chan)
    cals[:] = np.nan
    units = ['UNKNOWN'] * n_eeg_chan
    for chan, props in cfg.items('Channel Infos'):
        n = int(re.findall(r'ch(\d+)', chan)[0])
        props = props.split(',')
        if len(props) < 4:
            name, _, resolution = props
            unit = 'V'
        else:
            name, _, resolution, unit = props[:4]
        ch_names[n - 1] = name
        if resolution == "":        # For truncated vhdrs (e.g. EEGLAB export)
            cals[n - 1] = 0.000001  # Fill in a default
        else:
            cals[n - 1] = float(resolution)
        unit = unit.replace('\xc2', '')  # Remove unwanted control characters
        if u(unit) == u('\xb5V'):
            units[n - 1] = 1e-6
        elif unit == 'V':
            units[n - 1] = 1.
        else:
            units[n - 1] = unit

    eeg_info['units'] = np.asarray(units, dtype=float)

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
        for i, ch in enumerate(ch_names, 1):
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
    eeg_info['marker_id'] = os.path.join(path, cfg.get('Common Infos',
                                                       'MarkerFile'))
    info['meas_date'] = int(time.time())

    # Creates a list of dicts of eeg channels for raw.info
    logger.info('Setting channel info structure...')
    info['chs'] = []
    info['nchan'] = nchan = n_eeg_chan
    info['ch_names'] = ch_names
    info['sfreq'] = sfreq

    idxs = range(len(ch_names))
    for idx, ch_name, cal in zip(idxs, ch_names, cals):
        if ch_name in eog or idx in eog or idx - nchan in eog:
            kind = FIFF.FIFFV_EOG_CH
            coil_type = FIFF.FIFFV_COIL_NONE
        elif ch_name in misc or idx in misc or idx - nchan in misc:
            kind = FIFF.FIFFV_MISC_CH
            coil_type = FIFF.FIFFV_COIL_NONE
        else:
            kind = FIFF.FIFFV_EEG_CH
            coil_type = FIFF.FIFFV_COIL_EEG
        chan_info = {'ch_name': ch_name,
                     'coil_type': coil_type,
                     'kind': kind,
                     'logno': idx + 1,
                     'scanno': idx + 1,
                     'cal': cal,
                     'range': 1.,
                     'unit_mul': 0.,  # always zero- mne manual pg. 273
                     'unit': FIFF.FIFF_UNIT_V,
                     'coord_frame': FIFF.FIFFV_COORD_HEAD,
                     'eeg_loc': np.zeros(3),
                     'loc': np.zeros(12)}
        info['chs'].append(chan_info)

    # for stim channel
    events = _read_vmrk_events(eeg_info['marker_id'], response_trig_shift)

    return info, eeg_info, events


def read_raw_brainvision(vhdr_fname, montage=None,
                         eog=('HEOGL', 'HEOGR', 'VEOGb'), misc=None,
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
        Default is ('HEOGL', 'HEOGR', 'VEOGb').
    misc : list or tuple of str
        Names of channels or list of indices that should be designated
        MISC channels. Values should correspond to the electrodes
        in the vhdr file. Default is None.
    reference : None | str
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
    raw : Instance of RawBrainVision
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
