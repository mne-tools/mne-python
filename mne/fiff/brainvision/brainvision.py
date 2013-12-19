"""Conversion tool from Brain Vision EEG to FIF

"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import os
import time
import re
import warnings
from ...externals.six import StringIO, u
from ...externals.six.moves import configparser

import numpy as np

from ...fiff import pick_types
from ...transforms import als_ras_trans, apply_trans
from ...utils import verbose, logger
from ..raw import Raw
from ..meas_info import Info
from ..constants import FIFF
from ...coreg import get_ras_to_neuromag_trans


class RawBrainVision(Raw):
    """Raw object from Brain Vision eeg file

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
    fix_ptb_events : bool
        Fix events produced by psychtoolbox and recorded through a stim
        tracker (KIT setup).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.fiff.Raw : Documentation of attribute and methods.
    """
    @verbose
    def __init__(self, vhdr_fname, elp_fname=None, elp_names=None,
                 preload=False, fix_ptb_events=False, verbose=None):
        logger.info('Extracting eeg Parameters from %s...' % vhdr_fname)
        vhdr_fname = os.path.abspath(vhdr_fname)
        self.info, self._eeg_info, self._events = _get_eeg_info(vhdr_fname,
                                                                elp_fname,
                                                                elp_names,
                                                                fix_ptb_events)
        logger.info('Creating Raw.info structure...')

        # Raw attributes
        self.verbose = verbose
        self._preloaded = False
        self.fids = list()
        self._projector = None
        self.comp = None  # no compensation for EEG
        self.proj = False
        self.first_samp = 0
        f = open(self.info['file_id'])
        f.seek(0, os.SEEK_END)
        n_samples = f.tell()
        dtype = int(self._eeg_info['dtype'][-1])
        n_chan = self.info['nchan']
        self.last_samp = (n_samples // (dtype * (n_chan - 1))) - 1

        if preload:
            self._preloaded = preload
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
        data : array, [channels x samples]
           the data matrix (channels x samples).

        times : array, [samples]
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

        eeg_info = self._eeg_info
        sfreq = self.info['sfreq']
        n_chan = self.info['nchan']
        cals = np.array([chan_info['cal'] for chan_info in self.info['chs']])
        mults = np.array([chan_info['unit_mul'] for chan_info
                          in self.info['chs']])
        picks = pick_types(self.info, meg=False, eeg=True, exclude=[])
        n_eeg = picks.size
        cals = np.atleast_2d(cals[picks])
        mults = np.atleast_2d(mults[picks])

        if start >= stop:
            raise ValueError('No data in this range')

        logger.info('Reading %d ... %d  =  %9.3f ... %9.3f secs...' %
                    (start, stop - 1, start / float(sfreq),
                     (stop - 1) / float(sfreq)))

        with open(self.info['file_id'], 'rb') as f:
            buffer_size = (stop - start)
            pointer = start * n_chan
            f.seek(pointer)
            # extract data
            data = np.fromfile(f, dtype=eeg_info['dtype'],
                               count=buffer_size * n_eeg)
        if eeg_info['data_orientation'] == 'MULTIPLEXED':
            data = data.reshape((n_eeg, -1), order='F')
        elif eeg_info['data_orientation'] == 'VECTORIZED':
            data = data.reshape((n_eeg, -1), order='C')

        gains = cals * mults
        data = data * gains.T

        if len(self._events):
            stim_channel = _synthesize_stim_channel(self._events, start, stop)
            data = np.vstack((data, stim_channel))

        if sel is not None:
            data = data[sel]

        logger.info('[done]')
        times = np.arange(start, stop, dtype=float) / sfreq

        return data, times


def _read_vmrk_events(fname):
    """Read events from a vmrk file

    Parameters
    ----------
    fname : str
        vmrk file to be read.

    Returns
    -------
    events : array, [n_events x 3]
        An array containing the whole recording's events, each row representing
        an event as (onset, offset, trigger) sequence.
    """
    # read vmrk file
    with open(fname) as f:
        # setup config reader
        l = f.readline().strip()
        assert l == 'Brain Vision Data Exchange Marker File, Version 1.0'
        cfg = configparser.SafeConfigParser()
        cfg.readfp(f)

    # extract event informtion
    events = []
    for _, info in cfg.items('Marker Infos'):
        mtype, mdesc, onset, duration = info.split(',')[:4]
        if mtype == 'Stimulus':
            trigger = int(re.findall('S\s?(\d+)', mdesc)[0])
            onset = int(onset)
            duration = int(duration)
            offset = onset + duration
            events.append((onset, offset, trigger))

    events = np.array(events)
    return events


def _synthesize_stim_channel(events, start, stop):
    """Synthesize a stim channel from events read from a vmrk file

    Parameters
    ----------
    events : array, [n_events x 3]
        Each row representing an event as (onset, offset, trigger) sequence
        (the format returned by _read_vmrk_events).
    start : int
        First sample to return.
    stop : int
        Last sample to return.

    Returns
    -------
    stim_channel : array, [n_samples]
        An array containing the whole recording's event marking
    """
    # select events overlapping buffer
    idx = np.logical_and(events[:, 0] < stop, events[:, 1] > start)
    events = events[idx]

    # make onset and offset relative to buffer
    events[:, :2] -= start

    # fix start
    idx = events[:, 0] < 0
    events[idx, 0] = 0

    # create output buffer
    stim_channel = np.zeros(stop - start)
    for onset, offset, trigger in events:
        stim_channel[onset:offset] = trigger

    return stim_channel


def _fix_kit_ptb_events(events):
    """Fix events from a vmrk file recorded with psychtoolbox/stim tracker

    Parameters
    ----------
    events : array, [n_events x 3]
        Events read from a vmrk file, as they are returned by
        _read_vmrk_events.

    Returns
    -------
    fixed_events : array, [n_good_events x 3]
        Fixed events.
    """
    if not np.all(np.diff(events[:, :2], axis=1) == 1):
        err = ("Not KIT psychtoolbox input data (not all durations are 1)")
        raise ValueError(err)

    # invert trigger codes
    events[:, 2] = np.invert(events[:, 2].astype(np.uint8))

    # extend duration until next tigger start
    events[:-1, 1] = events[1:, 0]

    # remove 0 events
    idx = np.nonzero(events[:, 2])[0]
    events = events[idx]

    return events


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
    pattern = re.compile(r'(\-?\d+\.\d+)\s+(\-?\d+\.\d+)\s+(\-?\d+\.\d+)')
    with open(elp_fname) as fid:
        elp = pattern.findall(fid.read())
    coords_orig = np.array(elp, dtype=float)
    coords_ras = apply_trans(als_ras_trans, coords_orig)
    chs_ras = dict(zip(elp_names, coords_ras))
    nasion = chs_ras['nasion']
    lpa = chs_ras['lpa']
    rpa = chs_ras['rpa']
    trans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
    coords_neuromag = apply_trans(trans, coords_ras)
    chs_neuromag = dict(zip(elp_names, coords_neuromag))
    return chs_neuromag


def _get_eeg_info(vhdr_fname, elp_fname=None, elp_names=None,
                  fix_ptb_events=False):
    """Extracts all the information from the header file.

    Parameters
    ----------
    vhdr_fname : str
        Raw EEG header to be read.
    elp_fname : str | None
        Path to the elp file containing electrode positions.
        If None, sensor locations are (0,0,0).
    elp_names : list | None
        A list of channel names in the same order as the points in the elp
        file. Electrode positions should be specified with the same names as
        in the vhdr file, and fiducials should be specified as "lpa" "nasion",
        "rpa". ELP positions with other names are ignored. If elp_names is not
        None and channels are missing, a KeyError is raised.
    fix_ptb_events : bool
        Fix events produced by psychtoolbox and recorded through a stim
        tracker (KIT setup).

    Returns
    -------
    info : instance of Info
        The measurement info.
    edf_info : dict
        A dict containing Brain Vision specific parameters.
    events : array, [n_events x 3]
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
    info['filenames'] = []
    info['dig'] = None
    info['dev_head_t'] = None
    info['proj_id'] = None
    info['proj_name'] = None
    info['experimenter'] = None
    info['description'] = None
    info['buffer_size_sec'] = 10.
    info['orig_blocks'] = None
    info['orig_fid_str'] = None

    eeg_info = {}

    with open(vhdr_fname, 'r') as f:
        # extract the first section to resemble a cfg
        l = f.readline().strip()
        assert l == 'Brain Vision Data Exchange Header File Version 1.0'
        settings = f.read()

    params, settings = settings.split('[Comment]')
    cfg = configparser.SafeConfigParser()
    cfg.readfp(StringIO(params))

    # get sampling info
    # Sampling interval is given in microsec
    sfreq = 1e6 / cfg.getfloat('Common Infos', 'SamplingInterval')
    sfreq = int(sfreq)
    n_chan = cfg.getint('Common Infos', 'NumberOfChannels')

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
    ch_names = ['UNKNOWN'] * n_chan
    cals = np.ones(n_chan) * np.nan
    units = []
    for chan, props in cfg.items('Channel Infos'):
        n = int(re.findall(r'ch(\d+)', chan)[0])
        name, _, resolution, unit = props.split(',')[:4]
        ch_names[n - 1] = name
        cals[n - 1] = resolution
        unit = unit.replace('\xc2', '')  # Remove unwanted control characters
        if u(unit) == u('\xb5V'):
            units.append(1e-6)
        elif unit == 'V':
            units.append(0)
        else:
            units.append(unit)

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
    info['nchan'] = n_chan
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

    for idx, ch_info in enumerate(zip(ch_names, cals, units), 1):
        ch_name, cal, unit_mul = ch_info
        chan_info = {}
        chan_info['ch_name'] = ch_name
        chan_info['kind'] = FIFF.FIFFV_EEG_CH
        chan_info['coil_type'] = FIFF.FIFFV_COIL_EEG
        chan_info['logno'] = idx
        chan_info['scanno'] = idx
        chan_info['cal'] = cal
        chan_info['range'] = 1.
        chan_info['unit_mul'] = unit_mul
        chan_info['unit'] = FIFF.FIFF_UNIT_V
        chan_info['coord_frame'] = FIFF.FIFFV_COORD_HEAD
        missing_positions = []
        if ch_locs is None:
            loc = np.zeros(3)
        elif ch_name in ch_locs:
            loc = ch_locs[ch_name]
        else:
            loc = np.zeros(3)
            missing_positions.append(ch_name)
        chan_info['eeg_loc'] = loc
        chan_info['loc'] = np.hstack((loc, np.zeros(9)))
        info['chs'].append(chan_info)

    # raise error if positions are missing
    if missing_positions:
        err = ("The following positions are missing from the ELP "
               "definitions: %s" % str(missing_positions))
        raise KeyError(err)

    # for stim channel
    events = _read_vmrk_events(eeg_info['marker_id'])
    if fix_ptb_events:
        events = _fix_kit_ptb_events(events)

    if len(events):
        chan_info = {}
        chan_info['ch_name'] = 'STI 014'
        chan_info['kind'] = FIFF.FIFFV_STIM_CH
        chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
        chan_info['logno'] = idx + 1
        chan_info['scanno'] = idx + 1
        chan_info['cal'] = 1
        chan_info['range'] = 1
        chan_info['unit_mul'] = 0
        chan_info['unit'] = FIFF.FIFF_UNIT_NONE
        chan_info['eeg_loc'] = np.zeros(3)
        chan_info['loc'] = np.zeros(12)
        info['nchan'] = n_chan + 1
        info['ch_names'].append(chan_info['ch_name'])
        info['chs'].append(chan_info)

    return info, eeg_info, events


def read_raw_brainvision(vhdr_fname, elp_fname=None, elp_names=None,
                         preload=False, fix_ptb_events=False, verbose=None):
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
    fix_ptb_events : bool
        Fix events produced by psychtoolbox and recorded through a stim
        tracker (KIT setup).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.fiff.Raw : Documentation of attribute and methods.
    """
    raw = RawBrainVision(vhdr_fname, elp_fname, elp_names, preload,
                         fix_ptb_events, verbose)
    return raw
