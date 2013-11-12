"""Conversion tool from Brain Vision EEG to FIF

"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import os
import time
import re
import warnings
from StringIO import StringIO
from ConfigParser import SafeConfigParser

import numpy as np

from ...fiff import pick_types
from ...transforms import als_ras_trans, apply_trans
from ...utils import verbose, logger
from ..raw import Raw
from ..meas_info import Info
from ..constants import FIFF
from ...coreg import get_ras_to_neuromag_trans


class RawEEG(Raw):
    """Raw object from EEG file

    Parameters
    ----------
    input_fname : str
        Path to the EEG header file.

    elp : str | None
        Path to the elp file containing electrode positions.
        If None, sensor locations are (0,0,0).

    elp_chs : list | None
        A list of channel names in order of collection of electrode position
        digitization.

    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.

    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.fiff.Raw : Documentation of attribute and methods.
    """
    @verbose
    def __init__(self, input_fname, elp=None, elp_chs=None,
                 preload=False, verbose=None):
        logger.info('Extracting eeg Parameters from %s...' % input_fname)
        input_fname = os.path.abspath(input_fname)
        self.info = _get_eeg_info(input_fname, elp, elp_chs)
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
        nsamples = f.tell()
        self.last_samp = nsamples / (2 * (self.info['nchan'] - 1))

        if preload:
            self._preloaded = preload
            logger.info('Reading raw data from %s...' % input_fname)
            self._data, _ = self._read_segment()
            assert len(self._data) == self.info['nchan']

            # Add time info
            self._times = np.arange(self.first_samp, self.last_samp,
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
        if sel is None:
            sel = range(self.info['nchan'])
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
            data = np.fromfile(f, dtype='<i2', count=buffer_size * n_eeg)
            data = data.reshape((n_eeg, -1), order='F')

            gains = cals * mults
            data = data * gains.T

        stim_channel = np.zeros(data.shape[1])
        evts = _read_vmrk(self.info['marker_id'])
        stim_channel[:evts.size] = evts
        stim_channel = stim_channel[start:stop]

        data = np.vstack((data, stim_channel))
        data = data[sel]

        logger.info('[done]')
        times = np.arange(start, stop, dtype=float) / sfreq

        return data, times


def _read_vmrk(fname):
    """Extracts the event markers for vmrk file

    Parameters
    ----------
    fname : str
        vmrk file to be read.

    Returns
    -------
    stim_channel : np.array
        An array containing the whole recording's event marking
    """

    with open(fname) as f:
    # setup config reader
        assert (f.readline().strip() ==
                'Brain Vision Data Exchange Marker File, Version 1.0')

        cfg = SafeConfigParser()
        cfg.readfp(f)
    events = []
    for (marker, info) in cfg.items('Marker Infos'):
        mtype, mdesc, offset, duration = info.split(',')[:4]
        if mtype == 'Stimulus':
            trigger = int(re.findall('S\s?(\d+)', mdesc)[0])
            offset, duration = int(offset), int(duration)
            events.append((trigger, offset, offset + duration))
    stim_channel = np.zeros(events[-1][2])
    for event in events:
        stim_channel[event[1]:event[2]] = trigger

    return stim_channel


def _get_elp_locs(fname, elp_chs):
    """Read a Polhemus ascii file

    Parameters
    ----------
    fname : str
        Path to head shape file acquired from Polhemus system and saved in
        ascii format.

    Returns
    -------
    ch_locs : numpy.array, shape = (n_points, 3)
        Electrode points in Neuromag space.
    """
    pattern = re.compile(r'(\-?\d+\.\d+)\s+(\-?\d+\.\d+)\s+(\-?\d+\.\d+)')
    with open(fname) as fid:
        elp = pattern.findall(fid.read())
    elp = np.array(elp, dtype=float)
    elp = apply_trans(als_ras_trans, elp)
    nasion, lpa, rpa = elp[:3]
    trans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
    elp = apply_trans(trans, elp[8:])
    ch_locs = dict(zip(elp_chs, elp))
    fid = nasion, lpa, rpa

    return fid, ch_locs


def _get_eeg_info(fname, elp=None, elp_chs=None, preload=False):
    """Extracts all the information from the HDR file.

    Parameters
    ----------
    fname : str
        Raw EEG file to be read.

    elp : str | None
        Path to the elp file containing electrode positions.
        If None, sensor locations are (0,0,0).

    elp_chs : list | None
        A list of channel names in order of collection of electrode position
        digitization.

    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.

    Returns
    -------
    info : instance of Info
        The measurement info.
    """

    info = Info()
    # Some keys to be consistent with FIF measurement info
    info['meas_id'] = None
    info['projs'] = []
    info['comps'] = []
    info['bads'] = []
    info['acq_pars'], info['acq_stim'] = None, None
    info['filename'] = fname
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

    with open(fname, 'rb') as f:
        # extract the first section to resemble a cfg
        assert (f.readline().strip() ==
                'Brain Vision Data Exchange Header File Version 1.0')
        settings = f.read()

    params, settings = settings.split('[Comment]')
    cfg = SafeConfigParser()
    cfg.readfp(StringIO(params))

    # get sampling info
    sfreq = re.findall('Sampling Rate\s\[Hz\]:\s(\d+)', settings)
    sfreq = int(sfreq[0])
    n_chan = cfg.getint('Common Infos', 'NumberOfChannels')

    # check binary format
    assert cfg.get('Common Infos', 'DataOrientation') == 'MULTIPLEXED'
    assert cfg.get('Common Infos', 'DataFormat') == 'BINARY'
    assert cfg.get('Binary Infos', 'BinaryFormat') == 'INT_16'

    # load channel labels
    ch_names = ['UNKNOWN'] * n_chan
    cals = np.ones(n_chan) * np.nan
    for chan, props in cfg.items('Channel Infos'):
        n = int(re.findall(r'ch(\d+)', chan)[0])
        name, _, resolution, _ = props.split(',')[:4]
        ch_names[n - 1] = name
        cals[n - 1] = resolution

    # locate EEG and marker files
    path = os.path.dirname(fname)
    info['file_id'] = os.path.join(path, cfg.get('Common Infos', 'DataFile'))
    info['marker_id'] = os.path.join(path, cfg.get('Common Infos',
                                                   'MarkerFile'))
    info['meas_date'] = int(time.time())

    settings = settings.splitlines()
    idx = settings.index('Channels') + 2
    header = settings[idx].split()
    assert '#' in header
    lowpass = []
    highpass = []
    units = []
    for i, ch in enumerate(ch_names, 1):
        line = settings[idx + i].split()
        assert ch in line
        if line[4] == '\xc2\xb5V':
            units.append(1e-6)
        else:
            units.append(line[4])
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

    # Creates a list of dicts of eeg channels for raw.info
    logger.info('Setting channel info structure...')
    info['chs'] = []
    info['nchan'] = n_chan + 1
    info['ch_names'] = ch_names
    info['sfreq'] = sfreq
    if elp and elp_chs:
        fid, ch_locs = _get_elp_locs(elp, elp_chs)
        nasion, lpa, rpa = fid
        info['dig'] = [{'r': nasion, 'ident': FIFF.FIFFV_POINT_NASION,
                        'kind': FIFF.FIFFV_POINT_CARDINAL,
                        'coord_frame':  FIFF.FIFFV_COORD_HEAD},
                       {'r': lpa, 'ident': FIFF.FIFFV_POINT_LPA,
                        'kind': FIFF.FIFFV_POINT_CARDINAL,
                        'coord_frame': FIFF.FIFFV_COORD_HEAD},
                       {'r': rpa, 'ident': FIFF.FIFFV_POINT_RPA,
                        'kind': FIFF.FIFFV_POINT_CARDINAL,
                        'coord_frame': FIFF.FIFFV_COORD_HEAD}]

    for idx, ch_info in enumerate(zip(ch_names, cals, units), 1):
        ch_name, cal, unit_mul = ch_info
        chan_info = {}
        chan_info['cal'] = cal
        chan_info['range'] = 1.
        chan_info['logno'] = idx
        chan_info['scanno'] = idx
        chan_info['unit_mul'] = unit_mul
        chan_info['ch_name'] = ch_name
        chan_info['unit'] = FIFF.FIFF_UNIT_V
        chan_info['coord_frame'] = FIFF.FIFFV_COORD_HEAD
        chan_info['coil_type'] = FIFF.FIFFV_COIL_EEG
        chan_info['kind'] = FIFF.FIFFV_EEG_CH
        if ch_locs:
            if ch_name in ch_locs:
                chan_info['eeg_loc'] = ch_locs['ch_name']
        else:
            chan_info['eeg_loc'] = np.zeros(3)
        chan_info['loc'] = np.zeros(12)
        chan_info['loc'][:3] = chan_info['eeg_loc']
        info['chs'].append(chan_info)
    # for stim channel
    chan_info = {}
    chan_info['range'] = 1
    chan_info['cal'] = 1
    chan_info['unit_mul'] = 0
    chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
    chan_info['unit'] = FIFF.FIFF_UNIT_NONE
    chan_info['kind'] = FIFF.FIFFV_STIM_CH
    chan_info['ch_name'] = 'STI 014'
    info['ch_names'].append(chan_info['ch_name'])
    info['chs'].append(chan_info)

    return info


def read_raw_eeg(input_fname, elp=None, elp_chs=None,
                 preload=False, verbose=None):
    """Reader for Brain Vision EEG file

    Parameters
    ----------
    input_fname : str
        Path to the EEG header file.

    elp : str | None
        Path to the elp file containing electrode positions.
        If None, sensor locations are (0,0,0).

    elp_chs : list | None
        A list of channel names in order of collection of electrode position
        digitization.

    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.

    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.fiff.Raw : Documentation of attribute and methods.
    """
    return RawEEG(input_fname=input_fname, elp=elp, elp_chs=None,
                  preload=preload, verbose=verbose)
