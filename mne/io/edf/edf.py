"""Conversion tool from EDF+,BDF to FIF

"""

# Authors: Teon Brooks <teon@nyu.edu>
#          Martin Billinger <martin.billinger@tugraz.at>
#
# License: BSD (3-clause)

import os
import calendar
import datetime
import re
import warnings
from math import ceil, floor
import warnings

import numpy as np
from scipy.interpolate import interp1d

from ...utils import verbose, logger
from ..base import _BaseRaw, _check_update_montage
from ..meas_info import Info
from ..pick import pick_types
from ..constants import FIFF
from ...filter import resample
from ...externals.six.moves import zip


class RawEDF(_BaseRaw):
    """Raw object from EDF+,BDF file

    Parameters
    ----------
    input_fname : str
        Path to the EDF+,BDF file.
    montage : str | None | instance of Montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0).
    eog : list or tuple of str
        Names of channels or list of indices that should be designated 
        EOG channels. Values should correspond to the electrodes in the 
        edf file. Default is None.
    misc : list or tuple of str
        Names of channels or list of indices that should be designated 
        MISC channels. Values should correspond to the electrodes in the
        edf file. Default is None.
    stim_channel : str | int | None
        The channel name or channel index (starting at 0).
        -1 corresponds to the last channel (default).
        If None, there will be no stim channel added.
    annot : str | None
        Path to annotation file.
        If None, no derived stim channel will be added (for files requiring
        annotation file to interpret stim channel).
    annotmap : str | None
        Path to annotation map file containing mapping from label to trigger.
        Must be specified if annot is not None.
    tal_channel : int | None
        The channel index (starting at 0).
        Index of the channel containing EDF+ annotations.
        -1 corresponds to the last channel.
        If None, the annotation channel is not used.
        Note: this is overruled by the annotation file if specified.
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
    def __init__(self, input_fname, montage, eog=None, misc=None,
                 stim_channel=-1, annot=None, annotmap=None, tal_channel=None,
                 preload=False, verbose=None):
        logger.info('Extracting edf Parameters from %s...' % input_fname)
        input_fname = os.path.abspath(input_fname)
        self.info, self._edf_info = _get_edf_info(input_fname, stim_channel,
                                                  annot, annotmap, tal_channel,
                                                  eog, misc, preload)
        logger.info('Creating Raw.info structure...')
        _check_update_montage(self.info, montage)

        if bool(annot) != bool(annotmap):
            warnings.warn(("Stimulus Channel will not be annotated. "
                           "Both 'annot' and 'annotmap' must be specified."))

        # Raw attributes
        self.verbose = verbose
        self.preload = False
        self._filenames = list()
        self._projector = None
        self.first_samp = 0
        self.last_samp = self._edf_info['nsamples'] - 1
        self.comp = None  # no compensation for EDF
        self.proj = False
        self._first_samps = np.array([self.first_samp])
        self._last_samps = np.array([self.last_samp])
        self._raw_lengths = np.array([self._edf_info['nsamples']])
        self.rawdirs = np.array([])
        self.cals = np.array([ch['cal'] for ch in self.info['chs']])
        self.orig_format = 'int'

        if preload:
            self.preload = preload
            logger.info('Reading raw data from %s...' % input_fname)
            self._data, _ = self._read_segment()
            assert len(self._data) == self.info['nchan']

            # Add time info
            self.first_samp, self.last_samp = 0, self._data.shape[1] - 1
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
        return "<RawEDF  |  %s>" % ', '.join(s)

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

        sfreq = self.info['sfreq']
        n_chan = self.info['nchan']
        data_size = self._edf_info['data_size']
        data_offset = self._edf_info['data_offset']
        stim_channel = self._edf_info['stim_channel']
        tal_channel = self._edf_info['tal_channel']
        annot = self._edf_info['annot']
        annotmap = self._edf_info['annotmap']

        # this is used to deal with indexing in the middle of a sampling period
        blockstart = int(floor(float(start) / sfreq) * sfreq)
        blockstop = int(ceil(float(stop) / sfreq) * sfreq)
        if blockstop > self.last_samp:
            blockstop = self.last_samp + 1

        if start >= stop:
            raise ValueError('No data in this range')

        logger.info('Reading %d ... %d  =  %9.3f ... %9.3f secs...' %
                    (start, stop - 1, start / float(sfreq),
                     (stop - 1) / float(sfreq)))

        # gain constructor
        physical_range = np.array([ch['range'] for ch in self.info['chs']])
        cal = np.array([ch['cal'] for ch in self.info['chs']], float)
        gains = np.atleast_2d(self._edf_info['units'] * (physical_range / cal))

        with open(self.info['file_id'], 'rb') as fid:
            # extract data
            fid.seek(data_offset)
            buffer_size = blockstop - blockstart
            pointer = blockstart * n_chan * data_size
            fid.seek(data_offset + pointer)
            datas = np.zeros((n_chan, buffer_size), dtype=float)
            blocks = int(ceil(float(buffer_size) / sfreq))

            # bdf data: 24bit data
            if self._edf_info['subtype'] == '24BIT':
                # loop over 10s increment to not tax the memory
                buffer_step = int(sfreq * 10)
                for k, block in enumerate(range(buffer_size, 0, -buffer_step)):
                    step = buffer_step
                    if block < step:
                        step = block
                    samp = int(step * n_chan * data_size)
                    blocks = int(ceil(float(step) / sfreq))
                    data = np.fromfile(fid, dtype=np.uint8, count=samp)
                    data = data.reshape(-1, 3).astype(np.int32)
                    # this converts to 24-bit little endian integer
                    # # no support in numpy
                    data = (data[:, 0] + (data[:, 1] << 8) +
                            (data[:, 2] << 16))
                    # 24th bit determines the sign
                    data[data >= (1 << 23)] -= (1 << 24)

                    data = data.reshape((int(sfreq), n_chan, blocks),
                                        order='F')
                    for i in range(blocks):
                        start_pt = int((sfreq * i) + (k * buffer_step))
                        stop_pt = int(start_pt + sfreq)
                        datas[:, start_pt:stop_pt] = data[:, :, i].T
            else:
                # complicated edf: various sampling rates within file
                if 'n_samps' in self._edf_info:
                    data = []
                    for i in range(blocks):
                        for samp in n_samps:
                            chan_data = np.fromfile(fid, dtype='<i2',
                                                    count=samp)
                            data.append(chan_data)
                    for j, samp in enumerate(n_samps):
                        chan_data = data[j::n_chan]
                        chan_data = np.hstack(chan_data)
                        if j == tal_channel:
                            # don't resample tal_channel,
                            # pad with zeros instead.
                            n_missing = int(sfreq - samp) * blocks
                            chan_data = np.hstack([chan_data, [0] * n_missing])
                        elif j == stim_channel and samp < sfreq:
                            if annot and annotmap or tal_channel is not None:
                                # don't bother with resampling the stim channel
                                # because it gets overwritten later on.
                                chan_data = np.zeros(sfreq)
                            else:
                                warnings.warn('Interpolating stim channel. '
                                              'Events may jitter.')
                                oldrange = np.linspace(0, 1, samp*blocks + 1,
                                                       True)
                                newrange = np.linspace(0, 1, sfreq*blocks,
                                                       False)
                                chan_data = interp1d(oldrange,
                                                     np.append(chan_data, 0),
                                                     kind='zero')(newrange)
                        elif samp != sfreq:
                            mult = sfreq / samp
                            chan_data = resample(x=chan_data, up=mult,
                                                 down=1, npad=0)
                        stop_pt = chan_data.shape[0]
                        datas[j, :stop_pt] = chan_data
                # simple edf
                else:
                    data = np.fromfile(fid, dtype='<i2',
                                       count=buffer_size * n_chan)
                    data = data.reshape((int(sfreq), n_chan, blocks),
                                        order='F')
                    for i in range(blocks):
                        start_pt = int(sfreq * i)
                        stop_pt = int(start_pt + sfreq)
                        datas[:, start_pt:stop_pt] = data[:, :, i].T
        datas *= gains.T

        if stim_channel is not None:
            if annot and annotmap:
                datas[stim_channel] = 0
                evts = _read_annot(annot, annotmap, sfreq, self.last_samp)
                datas[stim_channel, :evts.size] = evts[start:stop]
            elif tal_channel is not None:
                evts = _parse_tal_channel(datas[tal_channel])
                self._edf_info['events'] = evts

                unique_annots = sorted(set([e[2] for e in evts]))
                mapping = dict((a, n + 1) for n, a in enumerate(unique_annots))

                datas[stim_channel] = 0
                for t_start, t_duration, annotation in evts:
                    evid = mapping[annotation]
                    n_start = int(t_start * sfreq)
                    n_stop = int(t_duration * sfreq) + n_start - 1
                    # make sure events without duration get one sample
                    n_stop = n_stop if n_stop > n_start else n_start + 1
                    if any(datas[stim_channel][n_start:n_stop]):
                        raise NotImplementedError('EDF+ with overlapping '
                                                  'events not supported.')
                    datas[stim_channel][n_start:n_stop] = evid
            else:
                stim = np.array(datas[stim_channel], int)
                mask = 255 * np.ones(stim.shape, int)
                stim = np.bitwise_and(stim, mask)
                datas[stim_channel] = stim
        datastart = start - blockstart
        datastop = stop - blockstart
        datas = datas[sel, datastart:datastop]

        logger.info('[done]')
        times = np.arange(start, stop, dtype=float) / self.info['sfreq']

        return datas, times


def _parse_tal_channel(tal_channel_data):
    """Parse time-stamped annotation lists (TALs) in stim_channel
    and return list of events.

    Parameters
    ----------
    tal_channel_data : ndarray, shape = [n_samples]
        channel data in EDF+ TAL format

    Returns
    -------
    events : list
        List of events. Each event contains [start, duration, annotation].

    References
    ----------
    http://www.edfplus.info/specs/edfplus.html#tal
    """

    # convert tal_channel to an ascii string
    tals = bytearray()
    for s in tal_channel_data:
        i = int(s)
        tals.extend([i % 256, i // 256])

    regex_tal = '([+-]\d+\.?\d*)(\x15(\d+\.?\d*))?(\x14.*?)\x14\x00'
    tal_list = re.findall(regex_tal, tals.decode('ascii'))
    events = []
    for ev in tal_list:
        onset = float(ev[0])
        duration = float(ev[2]) if ev[2] else 0
        for annotation in ev[3].split('\x14')[1:]:
            if annotation:
                events.append([onset, duration, annotation])

    return events


def _get_edf_info(fname, stim_channel, annot, annotmap, tal_channel,
                  eog, misc, preload):
    """Extracts all the information from the EDF+,BDF file.

    Parameters
    ----------
    fname : str
        Raw EDF+,BDF file to be read.
    stim_channel : str | int | None
        The channel name or channel index (starting at 0).
        -1 corresponds to the last channel.
        If None, there will be no stim channel added.
    annot : str | None
        Path to annotation file.
        If None, no derived stim channel will be added (for files requiring
        annotation file to interpret stim channel).
    annotmap : str | None
        Path to annotation map file containing mapping from label to trigger.
        Must be specified if annot is not None.
    tal_channel : int | None
        The channel index (starting at 0).
        Index of the channel containing EDF+ annotations.
        -1 corresponds to the last channel.
        If None, the annotation channel is not used.
        Note: this is overruled by the annotation file if specified.
    eog : list of str | None
        Names of channels that should be designated EOG channels. Names should
        correspond to the electrodes in the edf file. Default is None.
    misc : list of str | None
        Names of channels that should be designated MISC channels. Names
        should correspond to the electrodes in the edf file. Default is None.
    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.

    Returns
    -------
    info : instance of Info
        The measurement info.
    edf_info : dict
        A dict containing all the EDF+,BDF  specific parameters.
    """

    if eog is None:
        eog = []
    if misc is None:
        misc = []
    info = Info()
    info['file_id'] = fname
    # Add info for fif object
    info['meas_id'] = None
    info['projs'] = []
    info['comps'] = []
    info['bads'] = []
    info['acq_pars'], info['acq_stim'] = None, None
    info['filename'] = fname
    info['ctf_head_t'] = None
    info['dev_ctf_t'] = []
    info['dig'] = None
    info['dev_head_t'] = None
    info['proj_id'] = None
    info['proj_name'] = None
    info['experimenter'] = None
    info['line_freq'] = None
    info['subject_info'] = None

    edf_info = dict()
    edf_info['annot'] = annot
    edf_info['annotmap'] = annotmap
    edf_info['events'] = []

    with open(fname, 'rb') as fid:
        assert(fid.tell() == 0)
        fid.seek(8)

        _ = fid.read(80).strip().decode()  # subject id
        _ = fid.read(80).strip().decode()  # recording id
        day, month, year = [int(x) for x in re.findall('(\d+)',
                                                       fid.read(8).decode())]
        hour, minute, sec = [int(x) for x in re.findall('(\d+)',
                                                        fid.read(8).decode())]
        date = datetime.datetime(year + 2000, month, day, hour, minute, sec)
        info['meas_date'] = calendar.timegm(date.utctimetuple())

        edf_info['data_offset'] = header_nbytes = int(fid.read(8).decode())
        subtype = fid.read(44).strip().decode()[:5]
        edf_info['subtype'] = subtype

        edf_info['n_records'] = n_records = int(fid.read(8).decode())
        # record length in seconds
        edf_info['record_length'] = record_length = float(fid.read(8).decode())
        info['nchan'] = int(fid.read(4).decode())
        channels = list(range(info['nchan']))
        ch_names = [fid.read(16).strip().decode() for _ in channels]
        _ = [fid.read(80).strip().decode() for _ in channels]  # transducer
        units = [fid.read(8).strip().decode() for _ in channels]
        for i, unit in enumerate(units):
            if unit == 'uV':
                units[i] = 1e-6
            else:
                units[i] = 1
        edf_info['units'] = units
        physical_min = np.array([float(fid.read(8).decode())
                                 for _ in channels])
        physical_max = np.array([float(fid.read(8).decode())
                                 for _ in channels])
        digital_min = np.array([float(fid.read(8).decode())
                                for _ in channels])
        digital_max = np.array([float(fid.read(8).decode())
                                for _ in channels])
        prefiltering = [fid.read(80).strip().decode() for _ in channels][:-1]
        highpass = np.ravel([re.findall('HP:\s+(\w+)', filt)
                             for filt in prefiltering])
        lowpass = np.ravel([re.findall('LP:\s+(\w+)', filt)
                            for filt in prefiltering])

        high_pass_default = 0.
        if highpass.size == 0:
            info['highpass'] = high_pass_default
        elif all(highpass):
            if highpass[0] == 'NaN':
                info['highpass'] = high_pass_default
            elif highpass[0] == 'DC':
                info['highpass'] = 0.
            else:
                info['highpass'] = float(highpass[0])
        else:
            info['highpass'] = float(np.min(highpass))
            warnings.warn('%s' % ('Channels contain different highpass'
                                  + 'filters. Highest filter setting will'
                                  + 'be stored.'))

        if lowpass.size == 0:
            info['lowpass'] = None
        elif all(lowpass):
            if lowpass[0] == 'NaN':
                info['lowpass'] = None
            else:
                info['lowpass'] = float(lowpass[0])
        else:
            info['lowpass'] = float(np.min(lowpass))
            warnings.warn('%s' % ('Channels contain different lowpass filters.'
                                  ' Lowest filter setting will be stored.'))
        n_samples_per_record = [int(fid.read(8).decode()) for _ in channels]
        if np.unique(n_samples_per_record).size != 1:
            edf_info['n_samps'] = np.array(n_samples_per_record)
            if not preload:
                raise RuntimeError('%s' % ('Channels contain different'
                                           'sampling rates. '
                                           'Must set preload=True'))

        fid.read(32 * info['nchan']).decode()  # reserved
        assert fid.tell() == header_nbytes

    physical_ranges = physical_max - physical_min
    cals = digital_max - digital_min

    # Some keys to be consistent with FIF measurement info
    info['description'] = None
    info['buffer_size_sec'] = 10.
    info['orig_blocks'] = None

    if edf_info['subtype'] == '24BIT':
        edf_info['data_size'] = 3  # 24-bit (3 byte) integers
    else:
        edf_info['data_size'] = 2  # 16-bit (2 byte) integers

    # Creates a list of dicts of eeg channels for raw.info
    logger.info('Setting channel info structure...')
    info['chs'] = []
    info['ch_names'] = ch_names
    if stim_channel == -1:
        stim_channel = info['nchan'] - 1
    for idx, ch_info in enumerate(zip(ch_names, physical_ranges, cals)):
        ch_name, physical_range, cal = ch_info
        chan_info = {}
        chan_info['cal'] = cal
        chan_info['logno'] = idx
        chan_info['scanno'] = idx + 1
        chan_info['range'] = physical_range
        chan_info['unit_mul'] = 0.
        chan_info['ch_name'] = ch_name
        chan_info['unit'] = FIFF.FIFF_UNIT_V
        chan_info['coord_frame'] = FIFF.FIFFV_COORD_HEAD
        chan_info['coil_type'] = FIFF.FIFFV_COIL_EEG
        chan_info['kind'] = FIFF.FIFFV_EEG_CH
        chan_info['eeg_loc'] = np.zeros(3)
        chan_info['loc'] = np.zeros(12)
        if ch_name in eog or idx in eog:
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['kind'] = FIFF.FIFFV_EOG_CH
        if ch_name in misc or or idx in misc:
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['kind'] = FIFF.FIFFV_MISC_CH
        check1 = stim_channel == ch_name
        check2 = stim_channel == idx
        check3 = info['nchan'] > 1
        stim_check = np.logical_and(np.logical_or(check1, check2), check3)
        if stim_check:
            chan_info['range'] = 1
            chan_info['cal'] = 1
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['unit'] = FIFF.FIFF_UNIT_NONE
            chan_info['kind'] = FIFF.FIFFV_STIM_CH
            chan_info['ch_name'] = 'STI 014'
            info['ch_names'][idx] = chan_info['ch_name']
            if isinstance(stim_channel, str):
                stim_channel = idx
        info['chs'].append(chan_info)
    edf_info['stim_channel'] = stim_channel

    # samples where datablock. not necessarily the same as sampling rate
    picks = pick_types(info, meg=False, eeg=True)
    info['sfreq'] = edf_info['n_samps'][picks].max() / float(record_length)
    edf_info['nsamples'] = n_records * info['sfreq']

    if info['lowpass'] is None:
        info['lowpass'] = info['sfreq'] / 2.

    # TODO: automatic detection of the tal_channel?
    if tal_channel == -1:
        edf_info['tal_channel'] = info['nchan'] - 1
    else:
        edf_info['tal_channel'] = tal_channel

    if tal_channel and not preload:
        raise RuntimeError('%s' % ('EDF+ Annotations (TAL) channel needs to be'
                                   ' parsed completely on loading.'
                                   'Must set preload=True'))

    return info, edf_info


def _read_annot(annot, annotmap, sfreq, data_length):
    """Annotation File Reader

    Parameters
    ----------
    annot : str
        Path to annotation file.
    annotmap : str
        Path to annotation map file containing mapping from label to trigger.
    sfreq : float
        Sampling frequency.
    data_length : int
        Length of the data file.

    Returns
    -------
    stim_channel : ndarray
        An array containing stimulus trigger events.
    """
    pat = '([+/-]\d+.\d+),(\w+)'
    annot = open(annot).read()
    triggers = re.findall(pat, annot)
    times, values = zip(*triggers)
    times = [float(time) * sfreq for time in times]

    pat = '(\w+):(\d+)'
    annotmap = open(annotmap).read()
    mappings = re.findall(pat, annotmap)
    maps = {}
    for mapping in mappings:
        maps[mapping[0]] = mapping[1]
    triggers = [int(maps[value]) for value in values]

    stim_channel = np.zeros(data_length)
    for time, trigger in zip(times, triggers):
        stim_channel[time] = trigger

    return stim_channel


def read_raw_edf(input_fname, montage=None, eog=None, misc=None,
                 stim_channel=-1, annot=None, annotmap=None, tal_channel=None,
                 preload=False, verbose=None):
    """Reader function for EDF+, BDF conversion to FIF

    Parameters
    ----------
    input_fname : str
        Path to the EDF+,BDF file.
    montage : str | None | instance of Montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0).
    eog : list or tuple of str
        Names of channels or list of indices that should be designated 
        EOG channels. Values should correspond to the electrodes in the 
        edf file. Default is None.
    misc : list or tuple of str
        Names of channels or list of indices that should be designated 
        MISC channels. Values should correspond to the electrodes in the
        edf file. Default is None.
    stim_channel : str | int | None
        The channel name or channel index (starting at 0).
        -1 corresponds to the last channel (default).
        If None, there will be no stim channel added.
    annot : str | None
        Path to annotation file.
        If None, no derived stim channel will be added (for files requiring
        annotation file to interpret stim channel).
    annotmap : str | None
        Path to annotation map file containing mapping from label to trigger.
        Must be specified if annot is not None.
    tal_channel : int | None
        The channel index (starting at 0).
        Index of the channel containing EDF+ annotations.
        -1 corresponds to the last channel.
        If None, the annotation channel is not used.
        Note: this is overruled by the annotation file if specified.
    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawEDF(input_fname=input_fname, montage=montage, eog=eog, misc=misc,
                  stim_channel=stim_channel, annot=annot, annotmap=annotmap,
                  tal_channel=tal_channel, preload=preload, verbose=verbose)
