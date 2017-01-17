"""Conversion tool from EDF, EDF+, BDF to FIF."""

# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Martin Billinger <martin.billinger@tugraz.at>
#
# License: BSD (3-clause)

import calendar
import datetime
import os
import re

import numpy as np

from ...utils import verbose, logger, warn
from ..utils import _blk_read_lims
from ..base import BaseRaw, _check_update_montage
from ..meas_info import _empty_info
from ..constants import FIFF
from ...filter import resample
from ...externals.six.moves import zip


class RawEDF(BaseRaw):
    """Raw object from EDF, EDF+, BDF file.

    Parameters
    ----------
    input_fname : str
        Path to the EDF+,BDF file.
    montage : str | None | instance of Montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
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
    exclude : list of str
        Channel names to exclude. This can help when reading data with
        different sampling rates to avoid unnecessary resampling.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, input_fname, montage, eog=None, misc=None,
                 stim_channel=-1, annot=None, annotmap=None, exclude=(),
                 preload=False, verbose=None):  # noqa: D102
        logger.info('Extracting edf Parameters from %s...' % input_fname)
        input_fname = os.path.abspath(input_fname)
        info, edf_info = _get_edf_info(input_fname, stim_channel, annot,
                                       annotmap, eog, misc, exclude, preload)
        logger.info('Creating Raw.info structure...')
        _check_update_montage(info, montage)

        if bool(annot) != bool(annotmap):
            warn("Stimulus Channel will not be annotated. Both 'annot' and "
                 "'annotmap' must be specified.")

        # Raw attributes
        last_samps = [edf_info['nsamples'] - 1]
        super(RawEDF, self).__init__(
            info, preload, filenames=[input_fname], raw_extras=[edf_info],
            last_samps=last_samps, orig_format='int',
            verbose=verbose)

        logger.info('Ready.')

    @verbose
    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        from scipy.interpolate import interp1d
        if mult is not None:
            # XXX "cals" here does not function the same way as in RawFIF,
            # and for efficiency we want to be able to combine mult and cals
            # so proj support will have to wait until this is resolved
            raise NotImplementedError('mult is not supported yet')
        exclude = self._raw_extras[fi]['exclude']
        sel = np.arange(self.info['nchan'])[idx]

        n_samps = self._raw_extras[fi]['n_samps']
        buf_len = int(self._raw_extras[fi]['max_samp'])
        sfreq = self.info['sfreq']
        data_size = self._raw_extras[fi]['data_size']
        data_offset = self._raw_extras[fi]['data_offset']
        stim_channel = self._raw_extras[fi]['stim_channel']
        tal_channels = self._raw_extras[fi]['tal_channel']
        annot = self._raw_extras[fi]['annot']
        annotmap = self._raw_extras[fi]['annotmap']
        subtype = self._raw_extras[fi]['subtype']

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
        if tal_channels is not None:
            for tal_channel in tal_channels:
                offsets[tal_channel] = 0

        # This is needed to rearrange the indices to correspond to correct
        # chunks on the file if excluded channels exist:
        selection = sel.copy()
        idx_map = np.argsort(selection)
        for ei in sorted(exclude):
            for ii, si in enumerate(sorted(selection)):
                if si >= ei:
                    selection[idx_map[ii]] += 1
            if tal_channels is not None:
                tal_channels = [tc + 1 if tc >= ei else tc for tc in
                                sorted(tal_channels)]

        # We could read this one EDF block at a time, which would be this:
        ch_offsets = np.cumsum(np.concatenate([[0], n_samps]))
        block_start_idx, r_lims, d_lims = _blk_read_lims(start, stop, buf_len)
        # But to speed it up, we really need to read multiple blocks at once,
        # Otherwise we can end up with e.g. 18,181 chunks for a 20 MB file!
        # Let's do ~10 MB chunks:
        n_per = max(10 * 1024 * 1024 // (ch_offsets[-1] * data_size), 1)
        with open(self._filenames[fi], 'rb', buffering=0) as fid:
            # extract data
            start_offset = (data_offset +
                            block_start_idx * ch_offsets[-1] * data_size)
            for ai in range(0, len(r_lims), n_per):
                block_offset = ai * ch_offsets[-1] * data_size
                n_read = min(len(r_lims) - ai, n_per)
                fid.seek(start_offset + block_offset, 0)
                # Read and reshape to (n_chunks_read, ch0_ch1_ch2_ch3...)
                many_chunk = _read_ch(fid, subtype, ch_offsets[-1] * n_read,
                                      data_size).reshape(n_read, -1)
                for ii, ci in enumerate(selection):
                    # This now has size (n_chunks_read, n_samp[ci])
                    ch_data = many_chunk[:, ch_offsets[ci]:ch_offsets[ci + 1]]
                    r_sidx = r_lims[ai][0]
                    r_eidx = (buf_len * (n_read - 1) +
                              r_lims[ai + n_read - 1][1])
                    d_sidx = d_lims[ai][0]
                    d_eidx = d_lims[ai + n_read - 1][1]
                    if n_samps[ci] != buf_len:
                        if tal_channels is not None and ci in tal_channels:
                            # don't resample tal_channels, zero-pad instead.
                            if n_samps[ci] < buf_len:
                                z = np.zeros((len(ch_data),
                                              buf_len - n_samps[ci]))
                                ch_data = np.append(ch_data, z, -1)
                            else:
                                ch_data = ch_data[:, :buf_len]
                        elif ci == stim_channel:
                            if annot and annotmap or tal_channels is not None:
                                # don't resample, it gets overwritten later
                                ch_data = np.zeros((len(ch_data, buf_len)))
                            else:
                                # Stim channel will be interpolated
                                old = np.linspace(0, 1, n_samps[ci] + 1, True)
                                new = np.linspace(0, 1, buf_len, False)
                                ch_data = np.append(
                                    ch_data, np.zeros((len(ch_data), 1)), -1)
                                ch_data = interp1d(old, ch_data,
                                                   kind='zero', axis=-1)(new)
                        else:
                            # XXX resampling each chunk isn't great,
                            # it forces edge artifacts to appear at
                            # each buffer boundary :(
                            # it can also be very slow...
                            ch_data = resample(
                                ch_data, buf_len, n_samps[ci], npad=0, axis=-1)
                    assert ch_data.shape == (len(ch_data), buf_len)
                    data[ii, d_sidx:d_eidx] = ch_data.ravel()[r_sidx:r_eidx]
        data *= gains.T[sel]
        data += offsets[sel]

        # only try to read the stim channel if it's not None and it's
        # actually one of the requested channels
        read_size = len(r_lims) * buf_len
        if stim_channel is not None and (sel == stim_channel).sum() > 0:
            stim_channel_idx = np.where(sel == stim_channel)[0]
            if annot and annotmap:
                evts = _read_annot(annot, annotmap, sfreq,
                                   self._last_samps[fi])
                data[stim_channel_idx, :] = evts[start:stop + 1]
            elif tal_channels is not None:
                tal_channel_idx = np.intersect1d(sel, tal_channels)
                evts = _parse_tal_channel(np.atleast_2d(data[tal_channel_idx]))
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
                        warn('EDF+ with overlapping events'
                             ' are not fully supported')
                    stim[n_start:n_stop] += evid
                data[stim_channel_idx, :] = stim[start:stop]
            else:
                # Allows support for up to 17-bit trigger values (2 ** 17 - 1)
                stim = np.bitwise_and(data[stim_channel_idx].astype(int),
                                      131071)
                data[stim_channel_idx, :] = stim


def _read_ch(fid, subtype, samp, data_size):
    """Read a number of samples for a single channel."""
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
    """Parse time-stamped annotation lists (TALs) in stim_channel.

    Parameters
    ----------
    tal_channel_data : ndarray, shape = [n_chans, n_samples]
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
    for chan in tal_channel_data:
        for s in chan:
            i = int(s)
            tals.extend(np.uint8([i % 256, i // 256]))

    regex_tal = '([+-]\d+\.?\d*)(\x15(\d+\.?\d*))?(\x14.*?)\x14\x00'
    # use of latin-1 because characters are only encoded for the first 256
    # code points and utf-8 can triggers an "invalid continuation byte" error
    tal_list = re.findall(regex_tal, tals.decode('latin-1'))

    events = []
    for ev in tal_list:
        onset = float(ev[0])
        duration = float(ev[2]) if ev[2] else 0
        for annotation in ev[3].split('\x14')[1:]:
            if annotation:
                events.append([onset, duration, annotation])

    return events


def _get_edf_info(fname, stim_channel, annot, annotmap, eog, misc, exclude,
                  preload):
    """Extract all the information from the EDF+,BDF file."""
    if eog is None:
        eog = []
    if misc is None:
        misc = []

    edf_info = dict()
    edf_info['annot'] = annot
    edf_info['annotmap'] = annotmap
    edf_info['events'] = []

    with open(fname, 'rb') as fid:
        assert(fid.tell() == 0)
        fid.seek(168)  # Seek 8 + 80 bytes for Subject id + 80 bytes for rec id

        day, month, year = [int(x) for x in re.findall('(\d+)',
                                                       fid.read(8).decode())]
        hour, minute, sec = [int(x) for x in re.findall('(\d+)',
                                                        fid.read(8).decode())]
        century = 2000 if year < 50 else 1900
        date = datetime.datetime(year + century, month, day, hour, minute, sec)

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
            warn('Header information is incorrect for record length. Default '
                 'record length set to 1.')
        else:
            edf_info['record_length'] = record_length
        nchan = int(fid.read(4).decode())
        channels = list(range(nchan))
        ch_names = [fid.read(16).strip().decode() for ch in channels]
        exclude = [ch_names.index(idx) for idx in exclude]
        for ch in channels:
            fid.read(80)  # transducer
        units = [fid.read(8).strip().decode() for ch in channels]
        edf_info['units'] = list()
        edf_info['exclude'] = exclude
        include = list()
        for i, unit in enumerate(units):
            if i in exclude:
                continue
            if unit == 'uV':
                edf_info['units'].append(1e-6)
            else:
                edf_info['units'].append(1)
            include.append(i)
        ch_names = [ch_names[idx] for idx in include]
        physical_min = np.array([float(fid.read(8).decode())
                                 for ch in channels])[include]
        edf_info['physical_min'] = physical_min
        physical_max = np.array([float(fid.read(8).decode())
                                 for ch in channels])[include]
        digital_min = np.array([float(fid.read(8).decode())
                                for ch in channels])[include]
        edf_info['digital_min'] = digital_min
        digital_max = np.array([float(fid.read(8).decode())
                                for ch in channels])[include]
        prefiltering = [fid.read(80).strip().decode() for ch in channels][:-1]
        highpass = np.ravel([re.findall('HP:\s+(\w+)', filt)
                             for filt in prefiltering])
        lowpass = np.ravel([re.findall('LP:\s+(\w+)', filt)
                            for filt in prefiltering])

        # number of samples per record
        n_samps = np.array([int(fid.read(8).decode()) for ch
                            in channels])
        edf_info['n_samps'] = n_samps
        n_samps = n_samps[include]

        fid.read(32 * nchan).decode()  # reserved
        assert fid.tell() == header_nbytes

    physical_ranges = physical_max - physical_min
    cals = digital_max - digital_min

    if edf_info['subtype'] in ('24BIT', 'bdf'):
        edf_info['data_size'] = 3  # 24-bit (3 byte) integers
    else:
        edf_info['data_size'] = 2  # 16-bit (2 byte) integers

    # Creates a list of dicts of eeg channels for raw.info
    logger.info('Setting channel info structure...')
    chs = list()

    tal_ch_name = 'EDF Annotations'
    tal_chs = np.where(np.array(ch_names) == tal_ch_name)[0]
    if len(tal_chs) > 0:
        if len(tal_chs) > 1:
            warn('Channel names are not unique, found duplicates for: %s. '
                 'Adding running numbers to duplicate channel names.'
                 % tal_ch_name)
        for idx, tal_ch in enumerate(tal_chs, 1):
            ch_names[tal_ch] = ch_names[tal_ch] + '-%s' % idx
        tal_channel = tal_chs
    else:
        tal_channel = None
    edf_info['tal_channel'] = tal_channel

    if tal_channel is not None and stim_channel is not None and not preload:
        raise RuntimeError('%s' % ('EDF+ Annotations (TAL) channel needs to be'
                                   ' parsed completely on loading.'
                                   ' You must set preload parameter to True.'))
    if stim_channel == -1:
        stim_channel = len(include) - 1
    pick_mask = np.ones(len(ch_names))
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
        chan_info['loc'] = np.zeros(12)
        if ch_name in eog or idx in eog or idx - nchan in eog:
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['kind'] = FIFF.FIFFV_EOG_CH
            pick_mask[idx] = False
        if ch_name in misc or idx in misc or idx - nchan in misc:
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['kind'] = FIFF.FIFFV_MISC_CH
            pick_mask[idx] = False
        check1 = stim_channel == ch_name
        check2 = stim_channel == idx
        check3 = nchan > 1
        stim_check = np.logical_and(np.logical_or(check1, check2), check3)
        if stim_check:
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['unit'] = FIFF.FIFF_UNIT_NONE
            chan_info['kind'] = FIFF.FIFFV_STIM_CH
            pick_mask[idx] = False
            chan_info['ch_name'] = 'STI 014'
            ch_names[idx] = chan_info['ch_name']
            edf_info['units'][idx] = 1
            if isinstance(stim_channel, str):
                stim_channel = idx
        if tal_channel is not None and idx in tal_channel:
            chan_info['range'] = 1
            chan_info['cal'] = 1
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['unit'] = FIFF.FIFF_UNIT_NONE
            chan_info['kind'] = FIFF.FIFFV_MISC_CH
            pick_mask[idx] = False
        chs.append(chan_info)
    edf_info['stim_channel'] = stim_channel

    if any(pick_mask):
        picks = [item for item, mask in zip(range(nchan), pick_mask) if mask]
        edf_info['max_samp'] = max_samp = n_samps[picks].max()
    else:
        edf_info['max_samp'] = max_samp = n_samps.max()
    # sfreq defined as the max sampling rate of eeg
    sfreq = n_samps.max() / record_length
    info = _empty_info(sfreq)
    info['meas_date'] = calendar.timegm(date.utctimetuple())
    info['chs'] = chs

    if highpass.size == 0:
        pass
    elif all(highpass):
        if highpass[0] == 'NaN':
            pass  # Placeholder for future use. Highpass set in _empty_info.
        elif highpass[0] == 'DC':
            info['highpass'] = 0.
        else:
            info['highpass'] = float(highpass[0])
    else:
        info['highpass'] = float(np.max(highpass))
        warn('Channels contain different highpass filters. Highest filter '
             'setting will be stored.')

    if lowpass.size == 0:
        pass
    elif all(lowpass):
        if lowpass[0] == 'NaN':
            pass  # Placeholder for future use. Lowpass set in _empty_info.
        else:
            info['lowpass'] = float(lowpass[0])
    else:
        info['lowpass'] = float(np.min(lowpass))
        warn('Channels contain different lowpass filters. Lowest filter '
             'setting will be stored.')

    # Some keys to be consistent with FIF measurement info
    info['description'] = None
    info['buffer_size_sec'] = 1.
    edf_info['nsamples'] = int(n_records * max_samp)

    # These are the conditions under which a stim channel will be interpolated
    if stim_channel is not None and not (annot and annotmap) and \
            tal_channel is None and n_samps[stim_channel] != int(max_samp):
        warn('Interpolating stim channel. Events may jitter.')
    info._update_redundant()
    return info, edf_info


def _read_annot(annot, annotmap, sfreq, data_length):
    """Annotation File Reader.

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
                 stim_channel=-1, annot=None, annotmap=None, exclude=(),
                 preload=False, verbose=None):
    """Reader function for EDF+, BDF conversion to FIF.

    Parameters
    ----------
    input_fname : str
        Path to the EDF+,BDF file.
    montage : str | None | instance of Montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
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
    exclude : list of str
        Channel names to exclude. This can help when reading data with
        different sampling rates to avoid unnecessary resampling.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

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
                  exclude=exclude, preload=preload, verbose=verbose)
