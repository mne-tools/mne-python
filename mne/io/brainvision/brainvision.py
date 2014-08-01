"""Conversion tool from Brain Vision EEG to FIF"""

# Authors: Teon Brooks <teon@nyu.edu>
#          Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
import time
import re
import warnings

import numpy as np

from ...coreg import get_ras_to_neuromag_trans, read_elp
from ...transforms import als_ras_trans, apply_trans
from ...utils import verbose, logger
from ..constants import FIFF
from ..meas_info import Info
from ..base import _BaseRaw

from ...externals.six import StringIO, u
from ...externals.six.moves import configparser


class RawBrainVision(_BaseRaw):
    """Raw object from Brain Vision EEG file

    Parameters
    ----------
    vdhr_fname : str
        Path to the EEG header file.
    elp_fname : str | None
        Path to the elp file containing electrode positions.
        If None, sensor locations are (0,0,0).
    elp_names : list | None
        A list of channel names in the same order as the points in the elp
        file. Electrode positions should be specified with the same names as
        in the vhdr file, and fiducials should be specified as "lpa" "nasion",
        "rpa". ELP positions with other names are ignored. If elp_names is not
        None and channels are missing, a KeyError is raised.
    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.
    reference : None | str
        Name of the electrode which served as the reference in the recording.
        If a name is provided, a corresponding channel is added and its data
        is set to 0. This is useful for later re-referencing. The name should
        correspond to a name in elp_names.
    eog : list of str
        Names of channels that should be designated EOG channels. Names should
        correspond to the vhdr file (default: ['HEOGL', 'HEOGR', 'VEOGb']).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    @verbose
    def __init__(self, vhdr_fname, elp_fname=None, elp_names=None,
                 preload=False, reference=None,
                 eog=['HEOGL', 'HEOGR', 'VEOGb'], ch_names=None, verbose=None):
        # backwards compatibility
        if ch_names is not None:
            if elp_names is not None:
                err = ("ch_names is a deprecated parameter, don't specify "
                       "ch_names if elp_names are specified.")
                raise TypeError(err)
            msg = "The ch_names parameter is deprecated. Use elp_names."
            warnings.warn(msg, DeprecationWarning)
            elp_names = ['nasion', 'lpa', 'rpa', None, None, None, None,
                         None] + list(ch_names)

        # Preliminary Raw attributes
        self._events = np.empty((0, 3))
        self.preload = False

        # Channel info and events
        logger.info('Extracting eeg Parameters from %s...' % vhdr_fname)
        vhdr_fname = os.path.abspath(vhdr_fname)
        self.info, self._eeg_info, events = _get_eeg_info(vhdr_fname,
                                                          elp_fname, elp_names,
                                                          reference, eog)
        self.set_brainvision_events(events)
        logger.info('Creating Raw.info structure...')

        # Raw attributes
        self.verbose = verbose
        self._filenames = list()
        self._projector = None
        self.comp = None  # no compensation for EEG
        self.proj = False
        self.first_samp = 0
        with open(self.info['file_id'], 'rb') as f:
            f.seek(0, os.SEEK_END)
            n_samples = f.tell()
        dtype = int(self._eeg_info['dtype'][-1])
        n_chan = self.info['nchan']
        self.last_samp = (n_samples // (dtype * (n_chan - 1))) - 1
        self._reference = reference

        if preload:
            self.preload = preload
            logger.info('Reading raw data from %s...' % vhdr_fname)
            self._data, _ = self._read_segment()
            assert len(self._data) == self.info['nchan']

            # Add time info
            self._times = np.arange(self.first_samp, self.last_samp + 1,
                                    dtype=np.float64)
            self._times /= self.info['sfreq']
            logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs'
                        % (self.first_samp, self.last_samp,
                           float(self.first_samp) / self.info['sfreq'],
                           float(self.last_samp) / self.info['sfreq']))
        logger.info('Ready.')

    def __repr__(self):
        n_chan = self.info['nchan']
        data_range = self.last_samp - self.first_samp + 1
        s = ('%r' % os.path.basename(self.info['file_id']),
             "n_channels x n_times : %s x %s" % (n_chan, data_range))
        return "<RawEEG  |  %s>" % ', '.join(s)

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
        if self._reference:
            chs = chs[:-1]
        if len(self._events):
            chs = chs[:-1]
        n_eeg = len(chs)
        cals = np.atleast_2d([chan_info['cal'] for chan_info in chs])
        mults = np.atleast_2d([chan_info['unit_mul'] for chan_info in chs])

        logger.info('Reading %d ... %d  =  %9.3f ... %9.3f secs...' %
                    (start, stop - 1, start / float(sfreq),
                     (stop - 1) / float(sfreq)))

        # read data
        dtype = np.dtype(eeg_info['dtype'])
        buffer_size = (stop - start)
        pointer = start * n_eeg * dtype.itemsize
        with open(self.info['file_id'], 'rb') as f:
            f.seek(pointer)
            # extract data
            data = np.fromfile(f, dtype=dtype, count=buffer_size * n_eeg)
        if eeg_info['data_orientation'] == 'MULTIPLEXED':
            data = data.reshape((n_eeg, -1), order='F')
        elif eeg_info['data_orientation'] == 'VECTORIZED':
            data = data.reshape((n_eeg, -1), order='C')

        gains = cals * mults
        data = data * gains.T

        # add reference channel and stim channel (if applicable)
        data_segments = [data]
        if self._reference:
            shape = (1, data.shape[1])
            ref_channel = np.zeros(shape)
            data_segments.append(ref_channel)
        if len(self._events):
            stim_channel = _synthesize_stim_channel(self._events, start, stop)
            data_segments.append(stim_channel)
        if len(data_segments) > 1:
            data = np.vstack(data_segments)

        if sel is not None:
            data = data[sel]

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


def _read_vmrk_events(fname):
    """Read events from a vmrk file

    Parameters
    ----------
    fname : str
        vmrk file to be read.

    Returns
    -------
    events : array, shape (n_events, 3)
        An array containing the whole recording's events, each row representing
        an event as (onset, duration, trigger) sequence.
    """
    # read vmrk file
    with open(fname) as fid:
        txt = fid.read()

    start_tag = 'Brain Vision Data Exchange Marker File, Version 1.0'
    if not txt.startswith(start_tag):
        raise ValueError("vmrk file should start with %r" % start_tag)

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
        if mtype == 'Stimulus':
            trigger = int(re.findall('S\s*?(\d+)', mdesc)[0])
            onset = int(onset)
            duration = int(duration)
            events.append((onset, duration, trigger))

    events = np.array(events)
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


def _get_elp_locs(elp_fname, elp_names):
    """Read a Polhemus ascii file

    Parameters
    ----------
    elp_fname : str
        Path to head shape file acquired from Polhemus system and saved in
        ascii format.
    elp_names : list
        A list in order of EEG electrodes found in the Polhemus digitizer file.

    Returns
    -------
    ch_locs : dict
        Dictionary whose keys are the names from elp_names and whose values
        are the coordinates from the elp file transformed to Neuromag space.
    """
    coords_orig = read_elp(elp_fname)
    coords_ras = apply_trans(als_ras_trans, coords_orig)
    chs_ras = dict(zip(elp_names, coords_ras))
    nasion = chs_ras['nasion']
    lpa = chs_ras['lpa']
    rpa = chs_ras['rpa']
    trans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
    coords_neuromag = apply_trans(trans, coords_ras)
    chs_neuromag = dict(zip(elp_names, coords_neuromag))
    return chs_neuromag


def _get_eeg_info(vhdr_fname, elp_fname, elp_names, reference, eog):
    """Extracts all the information from the header file.

    Parameters
    ----------
    vhdr_fname : str
        Raw EEG header to be read.
    elp_fname : str | None
        Path to the elp file containing electrode positions.
        If None, sensor locations are (0, 0, 0).
    elp_names : list | None
        A list of channel names in the same order as the points in the elp
        file. Electrode positions should be specified with the same names as
        in the vhdr file, and fiducials should be specified as "lpa" "nasion",
        "rpa". ELP positions with other names are ignored. If elp_names is not
        None and channels are missing, a KeyError is raised.
    reference : None | str
        Name of the electrode which served as the reference in the recording.
        If a name is provided, a corresponding channel is added and its data
        is set to 0. This is useful for later re-referencing. The name should
        correspond to a name in elp_names.
    eog : list of str
        Names of channels that should be designated EOG channels. Names should
        correspond to the vhdr file.

    Returns
    -------
    info : Info
        The measurement info.
    edf_info : dict
        A dict containing Brain Vision specific parameters.
    events : array, shape (n_events, 3)
        Events from the corresponding vmrk file.
    """

    info = Info()
    # Some keys to be consistent with FIF measurement info
    info['meas_id'] = None
    info['projs'] = []
    info['comps'] = []
    info['bads'] = []
    info['acq_pars'], info['acq_stim'] = None, None
    info['filename'] = vhdr_fname
    info['ctf_head_t'] = None
    info['dev_ctf_t'] = []
    info['dig'] = None
    info['dev_head_t'] = None
    info['proj_id'] = None
    info['proj_name'] = None
    info['experimenter'] = None
    info['description'] = None
    info['buffer_size_sec'] = 10.
    info['orig_blocks'] = None
    info['line_freq'] = None
    info['subject_info'] = None

    eeg_info = {}

    with open(vhdr_fname, 'r') as f:
        # extract the first section to resemble a cfg
        l = f.readline().strip()
        assert l == 'Brain Vision Data Exchange Header File Version 1.0'
        settings = f.read()

    params, settings = settings.split('[Comment]')
    cfg = configparser.ConfigParser()
    if hasattr(cfg, 'read_file'):  # newer API
        cfg.read_file(StringIO(params))
    else:
        cfg.readfp(StringIO(params))

    # get sampling info
    # Sampling interval is given in microsec
    sfreq = 1e6 / cfg.getfloat('Common Infos', 'SamplingInterval')
    sfreq = int(sfreq)
    n_data_chan = cfg.getint('Common Infos', 'NumberOfChannels')
    n_eeg_chan = n_data_chan + bool(reference)

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
        name, _, resolution, unit = props.split(',')[:4]
        ch_names[n - 1] = name
        cals[n - 1] = float(resolution)
        unit = unit.replace('\xc2', '')  # Remove unwanted control characters
        if u(unit) == u('\xb5V'):
            units[n - 1] = 1e-6
        elif unit == 'V':
            units[n - 1] = 0
        else:
            units[n - 1] = unit

    # add reference channel info
    if reference:
        ch_names[-1] = reference
        cals[-1] = cals[-2]
        units[-1] = units[-2]

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
            if ch == reference:
                continue
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
                info['highpass'] = 0
            else:
                info['highpass'] = int(highpass[0])
        else:
            info['highpass'] = np.min(highpass)
            warnings.warn('%s' % ('Channels contain different highpass '
                                  'filters. Highest filter setting will '
                                  'be stored.'))
        if len(lowpass) == 0:
            info['lowpass'] = None
        elif all(lowpass):
            if lowpass[0] == 'NaN':
                info['lowpass'] = None
            else:
                info['lowpass'] = int(lowpass[0])
        else:
            info['lowpass'] = np.min(lowpass)
            warnings.warn('%s' % ('Channels contain different lowpass filters.'
                                  ' Lowest filter setting will be stored.'))
    else:
        info['highpass'] = None
        info['lowpass'] = None

    # locate EEG and marker files
    path = os.path.dirname(vhdr_fname)
    info['file_id'] = os.path.join(path, cfg.get('Common Infos', 'DataFile'))
    eeg_info['marker_id'] = os.path.join(path, cfg.get('Common Infos',
                                                       'MarkerFile'))
    info['meas_date'] = int(time.time())

    # Creates a list of dicts of eeg channels for raw.info
    logger.info('Setting channel info structure...')
    info['chs'] = []
    info['nchan'] = n_eeg_chan
    info['ch_names'] = ch_names
    info['sfreq'] = sfreq
    if elp_fname and elp_names:
        ch_locs = _get_elp_locs(elp_fname, elp_names)
        info['dig'] = [{'r': ch_locs['nasion'],
                        'ident': FIFF.FIFFV_POINT_NASION,
                        'kind': FIFF.FIFFV_POINT_CARDINAL,
                        'coord_frame':  FIFF.FIFFV_COORD_HEAD},
                       {'r': ch_locs['lpa'], 'ident': FIFF.FIFFV_POINT_LPA,
                        'kind': FIFF.FIFFV_POINT_CARDINAL,
                        'coord_frame': FIFF.FIFFV_COORD_HEAD},
                       {'r': ch_locs['rpa'], 'ident': FIFF.FIFFV_POINT_RPA,
                        'kind': FIFF.FIFFV_POINT_CARDINAL,
                        'coord_frame': FIFF.FIFFV_COORD_HEAD}]
    else:
        ch_locs = None

    missing_positions = []
    idxs = range(1, len(ch_names) + 1)
    for idx, ch_name, cal, unit_mul in zip(idxs, ch_names, cals, units):
        is_eog = ch_name in eog
        if ch_locs is None:
            loc = np.zeros(3)
        elif ch_name in ch_locs:
            loc = ch_locs[ch_name]
        else:
            loc = np.zeros(3)
            if not is_eog:
                missing_positions.append(ch_name)

        if is_eog:
            kind = FIFF.FIFFV_EOG_CH
        else:
            kind = FIFF.FIFFV_EEG_CH

        chan_info = {'ch_name': ch_name,
                     'coil_type': FIFF.FIFFV_COIL_EEG,
                     'kind': kind,
                     'logno': idx,
                     'scanno': idx,
                     'cal': cal,
                     'range': 1.,
                     'unit_mul': unit_mul,
                     'unit': FIFF.FIFF_UNIT_V,
                     'coord_frame': FIFF.FIFFV_COORD_HEAD,
                     'eeg_loc': loc,
                     'loc': np.hstack((loc, np.zeros(9)))}

        info['chs'].append(chan_info)

    # raise error if positions are missing
    if missing_positions:
        err = ("The following positions are missing from the ELP "
               "definitions: %s. If those channels lack positions because "
               "they are EOG channels use the eog "
               "parameter" % str(missing_positions))
        raise KeyError(err)

    # for stim channel
    events = _read_vmrk_events(eeg_info['marker_id'])

    return info, eeg_info, events


def read_raw_brainvision(vhdr_fname, elp_fname=None, elp_names=None,
                         preload=False, reference=None,
                         eog=['HEOGL', 'HEOGR', 'VEOGb'], ch_names=None,
                         verbose=None):
    """Reader for Brain Vision EEG file

    Parameters
    ----------
    vhdr_fname : str
        Path to the EEG header file.
    elp_fname : str | None
        Path to the elp file containing electrode positions.
        If None, sensor locations are (0,0,0).
    elp_names : list | None
        A list of channel names in the same order as the points in the elp
        file. Electrode positions should be specified with the same names as
        in the vhdr file, and fiducials should be specified as "lpa" "nasion",
        "rpa". ELP positions with other names are ignored. If elp_names is not
        None and channels are missing, a KeyError is raised.
    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.
    reference : None | str
        Name of the electrode which served as the reference in the recording.
        If a name is provided, a corresponding channel is added and its data
        is set to 0. This is useful for later re-referencing. The name should
        correspond to a name in elp_names.
    eog : list of str
        Names of channels that should be designated EOG channels. Names should
        correspond to the vhdr file (default: ['HEOGL', 'HEOGR', 'VEOGb']).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    raw = RawBrainVision(vhdr_fname, elp_fname, elp_names, preload,
                         reference, eog, ch_names, verbose)
    return raw
