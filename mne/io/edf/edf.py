"""Conversion tool from EDF, EDF+, BDF to FIF

"""

# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Martin Billinger <martin.billinger@tugraz.at>
#
# License: BSD (3-clause)

import os
import calendar
import datetime
import re
import warnings
from math import ceil, floor

import numpy as np

from ...utils import verbose, logger
from ..base import _BaseRaw, _check_update_montage
from ..meas_info import _empty_info
from ..pick import pick_types
from ..constants import FIFF
from ...filter import resample
from ...externals.six.moves import zip


class RawEDF(_BaseRaw):
    """Raw object from EDF, EDF+, BDF file

    Parameters
    ----------
    input_fname : str
        Path to the EDF+,BDF file.
    montage : str | None | instance of Montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0).
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. Values should correspond to the electrodes in the
        edf file. Default is None.
    misc : list or tuple
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
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    @verbose
    def __init__(self, input_fname, montage, eog=None, misc=None,
                 stim_channel=-1, annot=None, annotmap=None,
                 preload=False, verbose=None):
        logger.info('Extracting edf Parameters from %s...' % input_fname)
        input_fname = os.path.abspath(input_fname)
        info, edf_info = _get_edf_info(input_fname, stim_channel,
                                       annot, annotmap,
                                       eog, misc, preload)
        logger.info('Creating Raw.info structure...')
        _check_update_montage(info, montage)

        if bool(annot) != bool(annotmap):
            warnings.warn(("Stimulus Channel will not be annotated. "
                           "Both 'annot' and 'annotmap' must be specified."))

        # Raw attributes
        last_samps = [edf_info['nsamples'] - 1]
        super(RawEDF, self).__init__(
            info, preload, filenames=[input_fname], raw_extras=[edf_info],
            last_samps=last_samps, orig_format='int',
            verbose=verbose)

        logger.info('Ready.')

    @verbose
    def _read_segment_file(self, data, idx, offset, fi, start, stop,
                           cals, mult):
        """Read a chunk of raw data"""
        from scipy.interpolate import interp1d
        if mult is not None:
            # XXX "cals" here does not function the same way as in RawFIF,
            # and for efficiency we want to be able to combine mult and cals
            # so proj support will have to wait until this is resolved
            raise NotImplementedError('mult is not supported yet')
        # RawFIF and RawEDF think of "stop" differently, easiest to increment
        # here and refactor later
        stop += 1
        sel = np.arange(self.info['nchan'])[idx]

        n_samps = self._raw_extras[fi]['n_samps']
        buf_len = self._raw_extras[fi]['max_samp']
        sfreq = self.info['sfreq']
        n_chan = self.info['nchan']
        data_size = self._raw_extras[fi]['data_size']
        data_offset = self._raw_extras[fi]['data_offset']
        stim_channel = self._raw_extras[fi]['stim_channel']
        tal_channel = self._raw_extras[fi]['tal_channel']
        annot = self._raw_extras[fi]['annot']
        annotmap = self._raw_extras[fi]['annotmap']
        subtype = self._raw_extras[fi]['subtype']

        # this is used to deal with indexing in the middle of a sampling period
        blockstart = int(floor(float(start) / buf_len) * buf_len)
        blockstop = int(ceil(float(stop) / buf_len) * buf_len)

        # gain constructor
        physical_range = np.array([ch['range'] for ch in self.info['chs']])
        cal = np.array([ch['cal'] for ch in self.info['chs']])
        gains = np.atleast_2d(self._raw_extras[fi]['units'] *
                              (physical_range / cal))

        # physical dimension in uV
        physical_min = np.atleast_2d(self._raw_extras[fi]['units'] *
                                     self._raw_extras[fi]['physical_min'])
        digital_min = self._raw_extras[fi]['digital_min']

        offsets = np.atleast_2d(physical_min - (digital_min * gains)).T
        if tal_channel is not None:
            offsets[tal_channel] = 0

        read_size = blockstop - blockstart
        this_data = np.empty((len(sel), buf_len))
        data = data[:, offset:offset + (stop - start)]
        """
        Consider this example:

        tmin, tmax = (2, 27)
        read_size = 30
        buf_len = 10
        sfreq = 1.

                        +---------+---------+---------+
        File structure: |  buf0   |   buf1  |   buf2  |
                        +---------+---------+---------+
        File time:      0        10        20        30
                        +---------+---------+---------+
        Requested time:   2                       27

                        |                             |
                    blockstart                    blockstop
                          |                        |
                        start                    stop

        We need 27 - 2 = 25 samples (per channel) to store our data, and
        we need to read from 3 buffers (30 samples) to get all of our data.

        On all reads but the first, the data we read starts at
        the first sample of the buffer. On all reads but the last,
        the data we read ends on the last sample of the buffer.

        We call this_data the variable that stores the current buffer's data,
        and data the variable that stores the total output.

        On the first read, we need to do this::

            >>> data[0:buf_len-2] = this_data[2:buf_len]

        On the second read, we need to do::

            >>> data[1*buf_len-2:2*buf_len-2] = this_data[0:buf_len]

        On the final read, we need to do::

            >>> data[2*buf_len-2:3*buf_len-2-3] = this_data[0:buf_len-3]

        """
        with open(self._filenames[fi], 'rb', buffering=0) as fid:
            # extract data
            fid.seek(data_offset + blockstart * n_chan * data_size)
            n_blk = int(ceil(float(read_size) / buf_len))
            start_offset = start - blockstart
            end_offset = blockstop - stop
            for bi in range(n_blk):
                # Triage start (sidx) and end (eidx) indices for
                # data (d) and read (r)
                if bi == 0:
                    d_sidx = 0
                    r_sidx = start_offset
                else:
                    d_sidx = bi * buf_len - start_offset
                    r_sidx = 0
                if bi == n_blk - 1:
                    d_eidx = data.shape[1]
                    r_eidx = buf_len - end_offset
                else:
                    d_eidx = (bi + 1) * buf_len - start_offset
                    r_eidx = buf_len
                n_buf_samp = r_eidx - r_sidx
                count = 0
                for j, samp in enumerate(n_samps):
                    # bdf data: 24bit data
                    if j not in sel:
                        fid.seek(samp * data_size, 1)
                        continue
                    if samp == buf_len:
                        # use faster version with skips built in
                        if r_sidx > 0:
                            fid.seek(r_sidx * data_size, 1)
                        ch_data = _read_ch(fid, subtype, n_buf_samp, data_size)
                        if r_eidx < buf_len:
                            fid.seek((buf_len - r_eidx) * data_size, 1)
                    else:
                        # read in all the data and triage appropriately
                        ch_data = _read_ch(fid, subtype, samp, data_size)
                        if j == tal_channel:
                            # don't resample tal_channel,
                            # pad with zeros instead.
                            n_missing = int(buf_len - samp)
                            ch_data = np.hstack([ch_data, [0] * n_missing])
                            ch_data = ch_data[r_sidx:r_eidx]
                        elif j == stim_channel:
                            if annot and annotmap or \
                                    tal_channel is not None:
                                # don't bother with resampling the stim ch
                                # because it gets overwritten later on.
                                ch_data = np.zeros(n_buf_samp)
                            else:
                                warnings.warn('Interpolating stim channel.'
                                              ' Events may jitter.')
                                oldrange = np.linspace(0, 1, samp + 1, True)
                                newrange = np.linspace(0, 1, buf_len, False)
                                newrange = newrange[r_sidx:r_eidx]
                                ch_data = interp1d(
                                    oldrange, np.append(ch_data, 0),
                                    kind='zero')(newrange)
                        else:
                            ch_data = resample(ch_data, buf_len, samp,
                                               npad=0)[r_sidx:r_eidx]
                    this_data[count, :n_buf_samp] = ch_data
                    count += 1
                data[:, d_sidx:d_eidx] = this_data[:, :n_buf_samp]
        data *= gains.T[sel]
        data += offsets[sel]

        # only try to read the stim channel if it's not None and it's
        # actually one of the requested channels
        if stim_channel is not None and (sel == stim_channel).sum() > 0:
            stim_channel_idx = np.where(sel == stim_channel)[0]
            if annot and annotmap:
                evts = _read_annot(annot, annotmap, sfreq,
                                   self._last_samps[fi])
                data[stim_channel_idx, :] = evts[start:stop]
            elif tal_channel is not None:
                tal_channel_idx = np.where(sel == tal_channel)[0][0]
                evts = _parse_tal_channel(data[tal_channel_idx])
                self._raw_extras[fi]['events'] = evts

                unique_annots = sorted(set([e[2] for e in evts]))
                mapping = dict((a, n + 1) for n, a in enumerate(unique_annots))

                stim = np.zeros(read_size)
                for t_start, t_duration, annotation in evts:
                    evid = mapping[annotation]
                    n_start = int(t_start * sfreq)
                    n_stop = int(t_duration * sfreq) + n_start - 1
                    # make sure events without duration get one sample
                    n_stop = n_stop if n_stop > n_start else n_start + 1
                    if any(stim[n_start:n_stop]):
                        raise NotImplementedError('EDF+ with overlapping '
                                                  'events not supported.')
                    stim[n_start:n_stop] = evid
                data[stim_channel_idx, :] = stim[start:stop]
            else:
                # Allows support for up to 16-bit trigger values (2 ** 16 - 1)
                stim = np.bitwise_and(data[stim_channel_idx].astype(int),
                                      65535)
                data[stim_channel_idx, :] = stim


def _read_ch(fid, subtype, samp, data_size):
    """Helper to read a number of samples for a single channel"""
    if subtype in ('24BIT', 'bdf'):
        ch_data = np.fromfile(fid, dtype=np.uint8,
                              count=samp * data_size)
        ch_data = ch_data.reshape(-1, 3).astype(np.int32)
        ch_data = ((ch_data[:, 0]) +
                   (ch_data[:, 1] << 8) +
                   (ch_data[:, 2] << 16))
        # 24th bit determines the sign
        ch_data[ch_data >= (1 << 23)] -= (1 << 24)
    # edf data: 16bit data
    else:
        ch_data = np.fromfile(fid, dtype='<i2', count=samp)
    return ch_data


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


def _get_edf_info(fname, stim_channel, annot, annotmap, eog, misc, preload):
    """Extracts all the information from the EDF+,BDF file"""

    if eog is None:
        eog = []
    if misc is None:
        misc = []
    info = _empty_info()
    info['filename'] = fname

    edf_info = dict()
    edf_info['annot'] = annot
    edf_info['annotmap'] = annotmap
    edf_info['events'] = []

    with open(fname, 'rb') as fid:
        assert(fid.tell() == 0)
        fid.seek(8)

        fid.read(80).strip().decode()  # subject id
        fid.read(80).strip().decode()  # recording id
        day, month, year = [int(x) for x in re.findall('(\d+)',
                                                       fid.read(8).decode())]
        hour, minute, sec = [int(x) for x in re.findall('(\d+)',
                                                        fid.read(8).decode())]
        date = datetime.datetime(year + 2000, month, day, hour, minute, sec)
        info['meas_date'] = calendar.timegm(date.utctimetuple())

        edf_info['data_offset'] = header_nbytes = int(fid.read(8).decode())
        subtype = fid.read(44).strip().decode()[:5]
        if len(subtype) > 0:
            edf_info['subtype'] = subtype
        else:
            edf_info['subtype'] = os.path.splitext(fname)[1][1:].lower()

        edf_info['n_records'] = n_records = int(fid.read(8).decode())
        # record length in seconds
        record_length = float(fid.read(8).decode())
        if record_length == 0:
            edf_info['record_length'] = record_length = 1.
            warnings.warn('Header information is incorrect for record length. '
                          'Default record length set to 1.')
        else:
            edf_info['record_length'] = record_length
        info['nchan'] = nchan = int(fid.read(4).decode())
        channels = list(range(info['nchan']))
        ch_names = [fid.read(16).strip().decode() for ch in channels]
        [fid.read(80).strip().decode() for ch in channels]  # transducer
        units = [fid.read(8).strip().decode() for ch in channels]
        for i, unit in enumerate(units):
            if unit == 'uV':
                units[i] = 1e-6
            else:
                units[i] = 1
        edf_info['units'] = units
        physical_min = np.array([float(fid.read(8).decode())
                                 for ch in channels])
        edf_info['physical_min'] = physical_min
        physical_max = np.array([float(fid.read(8).decode())
                                 for ch in channels])
        digital_min = np.array([float(fid.read(8).decode())
                                for ch in channels])
        edf_info['digital_min'] = digital_min
        digital_max = np.array([float(fid.read(8).decode())
                                for ch in channels])
        prefiltering = [fid.read(80).strip().decode() for ch in channels][:-1]
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
            warnings.warn('Channels contain different highpass filters. '
                          'Highest filter setting will be stored.')

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
        # number of samples per record
        n_samps = np.array([int(fid.read(8).decode()) for ch in channels])
        edf_info['n_samps'] = n_samps

        fid.read(32 * info['nchan']).decode()  # reserved
        assert fid.tell() == header_nbytes

    physical_ranges = physical_max - physical_min
    cals = digital_max - digital_min

    # Some keys to be consistent with FIF measurement info
    info['description'] = None
    info['buffer_size_sec'] = 10.

    if edf_info['subtype'] in ('24BIT', 'bdf'):
        edf_info['data_size'] = 3  # 24-bit (3 byte) integers
    else:
        edf_info['data_size'] = 2  # 16-bit (2 byte) integers

    # Creates a list of dicts of eeg channels for raw.info
    logger.info('Setting channel info structure...')
    info['chs'] = []
    info['ch_names'] = ch_names
    tal_ch_name = 'EDF Annotations'
    if tal_ch_name in ch_names:
        tal_channel = ch_names.index(tal_ch_name)
    else:
        tal_channel = None
    edf_info['tal_channel'] = tal_channel
    if tal_channel is not None and stim_channel is not None and not preload:
        raise RuntimeError('%s' % ('EDF+ Annotations (TAL) channel needs to be'
                                   ' parsed completely on loading.'
                                   ' You must set preload parameter to True.'))
    if stim_channel == -1:
        stim_channel = info['nchan'] - 1
    for idx, ch_info in enumerate(zip(ch_names, physical_ranges, cals)):
        ch_name, physical_range, cal = ch_info
        chan_info = {}
        chan_info['cal'] = cal
        chan_info['logno'] = idx + 1
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
        if ch_name in eog or idx in eog or idx - nchan in eog:
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['kind'] = FIFF.FIFFV_EOG_CH
        if ch_name in misc or idx in misc or idx - nchan in misc:
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['kind'] = FIFF.FIFFV_MISC_CH
        check1 = stim_channel == ch_name
        check2 = stim_channel == idx
        check3 = info['nchan'] > 1
        stim_check = np.logical_and(np.logical_or(check1, check2), check3)
        if stim_check:
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['unit'] = FIFF.FIFF_UNIT_NONE
            chan_info['kind'] = FIFF.FIFFV_STIM_CH
            chan_info['ch_name'] = 'STI 014'
            info['ch_names'][idx] = chan_info['ch_name']
            units[idx] = 1
            if isinstance(stim_channel, str):
                stim_channel = idx
        if tal_channel == idx:
            chan_info['range'] = 1
            chan_info['cal'] = 1
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['unit'] = FIFF.FIFF_UNIT_NONE
            chan_info['kind'] = FIFF.FIFFV_MISC_CH
        info['chs'].append(chan_info)
    edf_info['stim_channel'] = stim_channel

    # sfreq defined as the max sampling rate of eeg
    picks = pick_types(info, meg=False, eeg=True)
    if len(picks) == 0:
        edf_info['max_samp'] = max_samp = n_samps.max()
    else:
        edf_info['max_samp'] = max_samp = n_samps[picks].max()
    info['sfreq'] = max_samp / record_length
    edf_info['nsamples'] = int(n_records * max_samp)

    if info['lowpass'] is None:
        info['lowpass'] = info['sfreq'] / 2.

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
                 stim_channel=-1, annot=None, annotmap=None,
                 preload=False, verbose=None):
    """Reader function for EDF+, BDF conversion to FIF

    Parameters
    ----------
    input_fname : str
        Path to the EDF+,BDF file.
    montage : str | None | instance of Montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0).
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. Values should correspond to the electrodes in the
        edf file. Default is None.
    misc : list or tuple
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
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : Instance of RawEDF
        A Raw object containing EDF data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawEDF(input_fname=input_fname, montage=montage, eog=eog, misc=misc,
                  stim_channel=stim_channel, annot=annot, annotmap=annotmap,
                  preload=preload, verbose=verbose)
