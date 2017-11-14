"""Conversion tool from EDF, EDF+, BDF to FIF."""

# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Martin Billinger <martin.billinger@tugraz.at>
#          Nicolas Barascud <nicolas.barascud@ens.fr>
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
from ...utils import copy_function_doc_to_method_doc


def find_edf_events(raw):
    """Get original EDF events as read from the header.

    For GDF, the values are returned in form
    [n_events, pos, typ, chn, dur]
    where:

    ========  ===================================  =======
    name      description                          type
    ========  ===================================  =======
    n_events  The number of all events             integer
    pos       Beginnning of the events in samples  array
    typ       The event identifiers                array
    chn       The associated channels (0 for all)  array
    dur       The durations of the events          array
    ========  ===================================  =======

    For EDF+, the values are returned in form
    n_events * [onset, dur, desc]
    where:

    ========  ===================================  =======
    name      description                          type
    ========  ===================================  =======
    onset     Onset of the event in seconds        float
    dur       Duration of the event in seconds     float
    desc      Description of the event             str
    ========  ===================================  =======

    Parameters
    ----------
    raw : Instance of RawEDF
        The raw object for finding the events.

    Returns
    -------
    events : ndarray
        The events as they are in the file header.
    """
    return raw.find_edf_events()


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
    stim_channel : str | int | 'auto' | None
        The channel name or channel index (starting at 0). -1 corresponds to
        the last channel. If None, there will be no stim channel added. If
        'auto' (default), the stim channel will be added as the last channel if
        the header contains ``'EDF Annotations'`` or GDF events (otherwise stim
        channel will not be added).
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

    Notes
    -----
    Biosemi devices trigger codes are encoded in 16-bit format, whereas system
    codes (CMS in/out-of range, battery low, etc.) are coded in bits 16-23 of
    the status channel (see http://www.biosemi.com/faq/trigger_signals.htm).
    To retrieve correct event values (bits 1-16), one could do:

        >>> events = mne.find_events(...)  # doctest:+SKIP
        >>> events[:, 2] &= (2**16 - 1)  # doctest:+SKIP

    The above operation can be carried out directly in :func:`mne.find_events`
    using the ``mask`` and ``mask_type`` parameters
    (see :func:`mne.find_events` for more details).

    It is also possible to retrieve system codes, but no particular effort has
    been made to decode these in MNE. In case it is necessary, for instance to
    check the CMS bit, the following operation can be carried out:

        >>> cms_bit = 20  # doctest:+SKIP
        >>> cms_high = (events[:, 2] & (1 << cms_bit)) != 0  # doctest:+SKIP

    It is worth noting that in some special cases, it may be necessary to
    shift the event values in order to retrieve correct event triggers. This
    depends on the triggering device used to perform the synchronization.
    For instance, some GDF files need a 8 bits shift:

        >>> events[:, 2] >>= 8  # doctest:+SKIP

    In addition, for GDF files, the stimulus channel is constructed from the
    events in the header. The id numbers of overlapping events are simply
    combined through addition. To get the original events from the header,
    use function :func:`mne.io.find_edf_events`.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, input_fname, montage, eog=None, misc=None,
                 stim_channel=True, annot=None, annotmap=None, exclude=(),
                 preload=False, verbose=None):  # noqa: D102
        if stim_channel is True:
            warn("stim_channel will default to 'auto' in version 0.16. "
                 "Set stim_channel explicitly to avoid this warning.",
                 DeprecationWarning)
            stim_channel = 'auto' if input_fname[-4:] == '.gdf' else -1

        logger.info('Extracting edf Parameters from %s...' % input_fname)
        input_fname = os.path.abspath(input_fname)
        info, edf_info = _get_info(input_fname, stim_channel, annot,
                                   annotmap, eog, misc, exclude, preload)
        logger.info('Created Raw.info structure...')
        _check_update_montage(info, montage)

        if bool(annot) != bool(annotmap):
            warn("Stimulus Channel will not be annotated. Both 'annot' and "
                 "'annotmap' must be specified.")

        # Raw attributes
        last_samps = [edf_info['nsamples'] - 1]
        super(RawEDF, self).__init__(
            info, preload, filenames=[input_fname], raw_extras=[edf_info],
            last_samps=last_samps, orig_format='int', verbose=verbose)

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
        dtype = self._raw_extras[fi]['dtype_np']
        dtype_byte = self._raw_extras[fi]['dtype_byte']
        data_offset = self._raw_extras[fi]['data_offset']
        stim_channel = self._raw_extras[fi]['stim_channel']
        tal_channels = self._raw_extras[fi]['tal_channel']
        annot = self._raw_extras[fi]['annot']
        annotmap = self._raw_extras[fi]['annotmap']
        subtype = self._raw_extras[fi]['subtype']
        stim_data = self._raw_extras[fi].get('stim_data', None)  # for GDF

        if np.size(dtype_byte) > 1:
            if len(np.unique(dtype_byte)) > 1:
                warn("Multiple data type not supported")
            dtype = dtype[0]
            dtype_byte = dtype_byte[0]

        # gain constructor
        physical_range = np.array([ch['range'] for ch in self.info['chs']])
        cal = np.array([ch['cal'] for ch in self.info['chs']])
        cal = np.atleast_2d(physical_range / cal)  # physical / digital
        gains = np.atleast_2d(self._raw_extras[fi]['units'])

        # physical dimension in uV
        physical_min = self._raw_extras[fi]['physical_min']
        digital_min = self._raw_extras[fi]['digital_min']

        offsets = np.atleast_2d(physical_min - (digital_min * cal)).T
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
        n_per = max(10 * 1024 * 1024 // (ch_offsets[-1] * dtype_byte), 1)
        with open(self._filenames[fi], 'rb', buffering=0) as fid:

            # Extract data
            start_offset = (data_offset +
                            block_start_idx * ch_offsets[-1] * dtype_byte)
            for ai in range(0, len(r_lims), n_per):
                block_offset = ai * ch_offsets[-1] * dtype_byte
                n_read = min(len(r_lims) - ai, n_per)
                fid.seek(start_offset + block_offset, 0)
                # Read and reshape to (n_chunks_read, ch0_ch1_ch2_ch3...)
                many_chunk = _read_ch(fid, subtype, ch_offsets[-1] * n_read,
                                      dtype_byte, dtype).reshape(n_read, -1)
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
                            if (annot and annotmap or stim_data is not None or
                                    tal_channels is not None):
                                # don't resample, it gets overwritten later
                                ch_data = np.zeros((len(ch_data), buf_len))
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
                            ch_data = resample(ch_data, buf_len, n_samps[ci],
                                               npad=0, axis=-1)
                    assert ch_data.shape == (len(ch_data), buf_len)
                    data[ii, d_sidx:d_eidx] = ch_data.ravel()[r_sidx:r_eidx]

        data *= cal.T[sel]  # scale
        data += offsets[sel]  # offset
        data *= gains.T[sel]  # apply units gain last

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
            elif stim_data is not None:  # GDF events
                data[stim_channel_idx, :] = stim_data[start:stop]
            else:
                stim = np.bitwise_and(data[stim_channel_idx].astype(int),
                                      2**17 - 1)
                data[stim_channel_idx, :] = stim

    @copy_function_doc_to_method_doc(find_edf_events)
    def find_edf_events(self):
        return self._raw_extras[0]['events']


def _read_ch(fid, subtype, samp, dtype_byte, dtype=None):
    """Read a number of samples for a single channel."""
    # BDF
    if subtype in ('24BIT', 'bdf'):
        ch_data = np.fromfile(fid, dtype=dtype, count=samp * dtype_byte)
        ch_data = ch_data.reshape(-1, 3).astype(np.int32)
        ch_data = ((ch_data[:, 0]) +
                   (ch_data[:, 1] << 8) +
                   (ch_data[:, 2] << 16))
        # 24th bit determines the sign
        ch_data[ch_data >= (1 << 23)] -= (1 << 24)

    # GDF data and EDF data
    else:
        ch_data = np.fromfile(fid, dtype=dtype, count=samp)

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

    regex_tal = '([+-]\\d+\\.?\\d*)(\x15(\\d+\\.?\\d*))?(\x14.*?)\x14\x00'
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


def _get_info(fname, stim_channel, annot, annotmap, eog, misc, exclude,
              preload):
    """Extract all the information from the EDF+, BDF or GDF file."""
    if eog is None:
        eog = []
    if misc is None:
        misc = []

    # Read header from file
    ext = os.path.splitext(fname)[1][1:].lower()
    logger.info('%s file detected' % ext.upper())
    if ext in ('bdf', 'edf'):
        edf_info = _read_edf_header(fname, annot, annotmap, exclude)
    elif ext in ('gdf'):
        if annot is not None:
            warn('Annotations not yet supported for GDF files.')
        edf_info = _read_gdf_header(fname, stim_channel, exclude)
        if 'stim_data' not in edf_info and stim_channel == 'auto':
            stim_channel = None  # Cannot construct stim channel.
    else:
        raise NotImplementedError(
            'Only GDF, EDF, and BDF files are supported, got %s.' % ext)

    include = edf_info['include']
    ch_names = edf_info['ch_names']
    n_samps = edf_info['n_samps'][include]
    nchan = edf_info['nchan']
    physical_ranges = edf_info['physical_max'] - edf_info['physical_min']
    cals = edf_info['digital_max'] - edf_info['digital_min']
    if np.any(~np.isfinite(cals)):
        idx = np.where(~np.isfinite(cals))[0]
        warn('Scaling factor is not defined in following channels:\n' +
             ', '.join(ch_names[i] for i in idx))
        cals[idx] = 1
    if 'stim_data' in edf_info and stim_channel == 'auto':  # For GDF events.
        cals = np.append(cals, 1)
    # Check that stimulus channel exists in dataset
    if not ('stim_data' in edf_info or
            'EDF Annotations' in ch_names) and stim_channel == 'auto':
        stim_channel = None
    if stim_channel is not None:
        stim_channel = _check_stim_channel(stim_channel, ch_names, include)

    # Annotations
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

    # Creates a list of dicts of eeg channels for raw.info
    logger.info('Setting channel info structure...')
    chs = list()
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

    # Info structure
    # -------------------------------------------------------------------------

    # sfreq defined as the max sampling rate of eeg (stim_ch not included)
    if stim_channel is None:
        data_samps = n_samps
    else:
        data_samps = np.delete(n_samps, slice(stim_channel, stim_channel + 1))
    sfreq = data_samps.max() * \
        edf_info['record_length'][1] / edf_info['record_length'][0]

    info = _empty_info(sfreq)
    info['meas_date'] = edf_info['meas_date']
    info['chs'] = chs
    info['ch_names'] = ch_names

    # Filter settings
    highpass = edf_info['highpass']
    lowpass = edf_info['lowpass']
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
        pass  # Placeholder for future use. Lowpass set in _empty_info.
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
    edf_info['nsamples'] = int(edf_info['n_records'] * max_samp)

    # These are the conditions under which a stim channel will be interpolated
    if stim_channel is not None and not (annot and annotmap) and \
            tal_channel is None and n_samps[stim_channel] != int(max_samp):
        warn('Interpolating stim channel. Events may jitter.')
    info._update_redundant()

    return info, edf_info


def _read_edf_header(fname, annot, annotmap, exclude):
    """Read header information from EDF+ or BDF file."""
    edf_info = dict()
    edf_info.update(annot=annot, annotmap=annotmap, events=[])

    with open(fname, 'rb') as fid:

        fid.read(8)  # version (unused here)

        # patient ID
        pid = fid.read(80).decode()
        pid = pid.split(' ', 2)
        patient = {}
        if len(pid) >= 2:
            patient['id'] = pid[0]
            patient['name'] = pid[1]

        # Recording ID
        meas_id = {}
        meas_id['recording_id'] = fid.read(80).decode().strip(' \x00')

        day, month, year = [int(x) for x in
                            re.findall(r'(\d+)', fid.read(8).decode())]
        hour, minute, sec = [int(x) for x in
                             re.findall(r'(\d+)', fid.read(8).decode())]
        century = 2000 if year < 50 else 1900
        date = datetime.datetime(year + century, month, day, hour, minute, sec)

        header_nbytes = int(fid.read(8).decode())

        subtype = fid.read(44).strip().decode()[:5]
        if len(subtype) == 0:
            subtype = os.path.splitext(fname)[1][1:].lower()

        n_records = int(fid.read(8).decode())
        record_length = np.array([float(fid.read(8)), 1.])  # in seconds
        if record_length[0] == 0:
            record_length = record_length[0] = 1.
            warn('Header information is incorrect for record length. Default '
                 'record length set to 1.')

        nchan = int(fid.read(4).decode())
        channels = list(range(nchan))
        ch_names = [fid.read(16).strip().decode() for ch in channels]
        exclude = [ch_names.index(idx) for idx in exclude]
        for ch in channels:
            fid.read(80)  # transducer
        units = [fid.read(8).strip().decode() for ch in channels]
        edf_info['units'] = list()
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
        physical_max = np.array([float(fid.read(8).decode())
                                 for ch in channels])[include]
        digital_min = np.array([float(fid.read(8).decode())
                                for ch in channels])[include]
        digital_max = np.array([float(fid.read(8).decode())
                                for ch in channels])[include]
        prefiltering = [fid.read(80).decode().strip(' \x00')
                        for ch in channels][:-1]
        highpass = np.ravel([re.findall(r'HP:\s+(\w+)', filt)
                             for filt in prefiltering])
        lowpass = np.ravel([re.findall(r'LP:\s+(\w+)', filt)
                            for filt in prefiltering])

        # number of samples per record
        n_samps = np.array([int(fid.read(8).decode()) for ch
                            in channels])

        # Populate edf_info
        edf_info.update(
            ch_names=ch_names, data_offset=header_nbytes,
            digital_max=digital_max, digital_min=digital_min, exclude=exclude,
            highpass=highpass, include=include, lowpass=lowpass,
            meas_date=calendar.timegm(date.utctimetuple()),
            n_records=n_records, n_samps=n_samps, nchan=nchan,
            subject_info=patient, physical_max=physical_max,
            physical_min=physical_min, record_length=record_length,
            subtype=subtype)

        fid.read(32 * nchan).decode()  # reserved
        assert fid.tell() == header_nbytes

        fid.seek(0, 2)
        n_bytes = fid.tell()
        n_data_bytes = n_bytes - header_nbytes
        total_samps = (n_data_bytes // 3 if subtype == '24BIT'
                       else n_data_bytes // 2)
        read_records = total_samps // np.sum(n_samps)
        if n_records != read_records:
            warn('Number of records from the header does not match the file '
                 'size (perhaps the recording was not stopped before exiting).'
                 ' Inferring from the file size.')
            edf_info['n_records'] = n_records = read_records

        if subtype in ('24BIT', 'bdf'):
            edf_info['dtype_byte'] = 3  # 24-bit (3 byte) integers
            edf_info['dtype_np'] = np.uint8
        else:
            edf_info['dtype_byte'] = 2  # 16-bit (2 byte) integers
            edf_info['dtype_np'] = np.int16

    return edf_info


def _read_gdf_header(fname, stim_channel, exclude):
    """Read GDF 1.x and GDF 2.x header info."""
    edf_info = dict()
    events = []
    edf_info['annot'] = None
    edf_info['annotmap'] = None
    with open(fname, 'rb') as fid:

        version = fid.read(8).decode()

        gdftype_np = (None, np.int8, np.uint8, np.int16, np.uint16, np.int32,
                      np.uint32, np.int64, np.uint64, None, None, None, None,
                      None, None, None, np.float32, np.float64)
        gdftype_byte = [np.dtype(x).itemsize if x is not None else 0
                        for x in gdftype_np]
        assert sum(gdftype_byte) == 42

        edf_info['type'] = edf_info['subtype'] = version[:3]
        edf_info['number'] = float(version[4:])

        # GDF 1.x
        # ----------------------------------------------------------------------
        if edf_info['number'] < 1.9:

            # patient ID
            pid = fid.read(80).decode()
            pid = pid.split(' ', 2)
            patient = {}
            if len(pid) >= 2:
                patient['id'] = pid[0]
                patient['name'] = pid[1]

            # Recording ID
            meas_id = {}
            meas_id['recording_id'] = fid.read(80).decode().strip(' \x00')

            # date
            tm = fid.read(16).decode().strip(' \x00')
            try:
                if tm[14:16] == '  ':
                    tm = tm[:14] + '00' + tm[16:]
                date = (datetime.datetime(int(tm[0:4]), int(tm[4:6]),
                                          int(tm[6:8]), int(tm[8:10]),
                                          int(tm[10:12]), int(tm[12:14]),
                                          int(tm[14:16]) * pow(10, 4)))
            except Exception:
                date = datetime.datetime(2000, 1, 1)

            header_nbytes = np.fromfile(fid, np.int64, 1)[0]
            meas_id['equipment'] = np.fromfile(fid, np.uint8, 8)[0]
            meas_id['hospital'] = np.fromfile(fid, np.uint8, 8)[0]
            meas_id['technician'] = np.fromfile(fid, np.uint8, 8)[0]
            fid.seek(20, 1)    # 20bytes reserved

            n_records = np.fromfile(fid, np.int64, 1)[0]
            # record length in seconds
            record_length = np.fromfile(fid, np.uint32, 2)
            if record_length[0] == 0:
                record_length[0] = 1.
                warn('Header information is incorrect for record length. '
                     'Default record length set to 1.')
            nchan = np.fromfile(fid, np.uint32, 1)[0]
            channels = list(range(nchan))
            ch_names = [fid.read(16).decode().strip(' \x00')
                        for ch in channels]
            fid.seek(80 * len(channels), 1)  # transducer
            units = [fid.read(8).decode().strip(' \x00') for ch in channels]

            exclude = [ch_names.index(idx) for idx in exclude]
            include = list()
            for i, unit in enumerate(units):
                if unit[:2] == 'uV':
                    units[i] = 1e-6
                else:
                    units[i] = 1
                include.append(i)

            ch_names = [ch_names[idx] for idx in include]
            physical_min = np.fromfile(fid, np.float64, len(channels))
            physical_max = np.fromfile(fid, np.float64, len(channels))
            digital_min = np.fromfile(fid, np.int64, len(channels))
            digital_max = np.fromfile(fid, np.int64, len(channels))
            prefiltering = [fid.read(80).decode().strip(' \x00')
                            for ch in channels][:-1]
            highpass = np.ravel([re.findall(r'HP:\s+(\w+)', filt)
                                 for filt in prefiltering])
            lowpass = np.ravel([re.findall('LP:\\s+(\\w+)', filt)
                                for filt in prefiltering])

            # n samples per record
            n_samps = np.fromfile(fid, np.int32, len(channels))

            # channel data type
            dtype = np.fromfile(fid, np.int32, len(channels))

            # total number of bytes for data
            bytes_tot = np.sum([gdftype_byte[t] * n_samps[i]
                                for i, t in enumerate(dtype)])

            # Populate edf_info
            edf_info.update(
                bytes_tot=bytes_tot, ch_names=ch_names,
                data_offset=header_nbytes, digital_min=digital_min,
                digital_max=digital_max,
                dtype_byte=[gdftype_byte[t] for t in dtype],
                dtype_np=[gdftype_np[t] for t in dtype], exclude=exclude,
                highpass=highpass, include=include, lowpass=lowpass,
                meas_date=calendar.timegm(date.utctimetuple()),
                meas_id=meas_id, n_records=n_records, n_samps=n_samps,
                nchan=nchan, subject_info=patient, physical_max=physical_max,
                physical_min=physical_min, record_length=record_length,
                units=units)

            fid.seek(32 * edf_info['nchan'], 1)  # reserved
            assert fid.tell() == header_nbytes

            # Event table
            # ------------------------------------------------------------------
            etp = header_nbytes + n_records * edf_info['bytes_tot']
            # skip data to go to event table
            fid.seek(etp)
            etmode = np.fromfile(fid, np.uint8, 1)[0]
            if etmode in (1, 3):
                sr = np.fromfile(fid, np.uint8, 3)
                event_sr = sr[0]
                for i in range(1, len(sr)):
                    event_sr = event_sr + sr[i] * 2 ** (i * 8)
                n_events = np.fromfile(fid, np.uint32, 1)[0]
                pos = np.fromfile(fid, np.uint32, n_events) - 1  # 1-based inds
                typ = np.fromfile(fid, np.uint16, n_events)

                if etmode == 3:
                    chn = np.fromfile(fid, np.uint16, n_events)
                    dur = np.fromfile(fid, np.uint32, n_events)
                else:
                    chn = np.zeros(n_events, dtype=np.int32)
                    dur = np.ones(n_events, dtype=np.uint32)
                np.clip(dur, 1, np.inf, out=dur)
                events = [n_events, pos, typ, chn, dur]

        # GDF 2.x
        # ----------------------------------------------------------------------
        else:
            # FIXED HEADER
            handedness = ('Unknown', 'Right', 'Left', 'Equal')
            gender = ('Unknown', 'Male', 'Female')
            scale = ('Unknown', 'No', 'Yes', 'Corrected')

            # date
            pid = fid.read(66).decode()
            pid = pid.split(' ', 2)
            patient = {}
            if len(pid) >= 2:
                patient['id'] = pid[0]
                patient['name'] = pid[1]
            fid.seek(10, 1)  # 10bytes reserved

            # Smoking / Alcohol abuse / drug abuse / medication
            sadm = np.fromfile(fid, np.uint8, 1)[0]
            patient['smoking'] = scale[sadm % 4]
            patient['alcohol_abuse'] = scale[(sadm >> 2) % 4]
            patient['drug_abuse'] = scale[(sadm >> 4) % 4]
            patient['medication'] = scale[(sadm >> 6) % 4]
            patient['weight'] = np.fromfile(fid, np.uint8, 1)[0]
            if patient['weight'] == 0 or patient['weight'] == 255:
                patient['weight'] = None
            patient['height'] = np.fromfile(fid, np.uint8, 1)[0]
            if patient['height'] == 0 or patient['height'] == 255:
                patient['height'] = None

            # Gender / Handedness / Visual Impairment
            ghi = np.fromfile(fid, np.uint8, 1)[0]
            patient['sex'] = gender[ghi % 4]
            patient['handedness'] = handedness[(ghi >> 2) % 4]
            patient['visual'] = scale[(ghi >> 4) % 4]

            # Recording identification
            meas_id = {}
            meas_id['recording_id'] = fid.read(64).decode().strip(' \x00')
            vhsv = np.fromfile(fid, np.uint8, 4)
            loc = {}
            if vhsv[3] == 0:
                loc['vertpre'] = 10 * int(vhsv[0] >> 4) + int(vhsv[0] % 16)
                loc['horzpre'] = 10 * int(vhsv[1] >> 4) + int(vhsv[1] % 16)
                loc['size'] = 10 * int(vhsv[2] >> 4) + int(vhsv[2] % 16)
            else:
                loc['vertpre'] = 29
                loc['horzpre'] = 29
                loc['size'] = 29
            loc['version'] = 0
            loc['latitude'] = \
                float(np.fromfile(fid, np.uint32, 1)[0]) / 3600000
            loc['longitude'] = \
                float(np.fromfile(fid, np.uint32, 1)[0]) / 3600000
            loc['altitude'] = float(np.fromfile(fid, np.int32, 1)[0]) / 100
            meas_id['loc'] = loc

            date = np.fromfile(fid, np.uint64, 1)[0]
            if date == 0:
                date = datetime.datetime(1, 1, 1)
            else:
                date = datetime.datetime(1, 1, 1) + \
                    datetime.timedelta(date * pow(2, -32) - 367)

            birthday = np.fromfile(fid, np.uint64, 1).tolist()[0]
            if birthday == 0:
                birthday = datetime.datetime(1, 1, 1)
            else:
                birthday = (datetime.datetime(1, 1, 1) +
                            datetime.timedelta(birthday * pow(2, -32) - 367))
            patient['birthday'] = birthday
            if patient['birthday'] != datetime.datetime(1, 1, 1, 0, 0):
                today = datetime.datetime.today()
                patient['age'] = today.year - patient['birthday'].year
                today = today.replace(year=patient['birthday'].year)
                if today < patient['birthday']:
                    patient['age'] -= 1
            else:
                patient['age'] = None

            header_nbytes = np.fromfile(fid, np.uint16, 1)[0] * 256

            fid.seek(6, 1)  # 6 bytes reserved
            meas_id['equipment'] = np.fromfile(fid, np.uint8, 8)
            meas_id['ip'] = np.fromfile(fid, np.uint8, 6)
            patient['headsize'] = np.fromfile(fid, np.uint16, 3)
            patient['headsize'] = np.asarray(patient['headsize'], np.float32)
            patient['headsize'] = np.ma.masked_array(
                patient['headsize'],
                np.equal(patient['headsize'], 0), None).filled()
            ref = np.fromfile(fid, np.float32, 3)
            gnd = np.fromfile(fid, np.float32, 3)
            n_records = np.fromfile(fid, np.int64, 1)[0]

            # record length in seconds
            record_length = np.fromfile(fid, np.uint32, 2)
            if record_length[0] == 0:
                record_length[0] = 1.
                warn('Header information is incorrect for record length. '
                     'Default record length set to 1.')

            nchan = np.fromfile(fid, np.uint16, 1)[0]
            fid.seek(2, 1)  # 2bytes reserved

            # Channels (variable header)
            channels = list(range(nchan))
            ch_names = [fid.read(16).decode().strip(' \x00')
                        for ch in channels]
            exclude = [ch_names.index(idx) for idx in exclude]

            fid.seek(80 * len(channels), 1)  # reserved space
            fid.seek(6 * len(channels), 1)  # phys_dim, obsolete

            """The Physical Dimensions are encoded as int16, according to:
            - Units codes :
            https://sourceforge.net/p/biosig/svn/HEAD/tree/trunk/biosig/doc/units.csv
            - Decimal factors codes:
            https://sourceforge.net/p/biosig/svn/HEAD/tree/trunk/biosig/doc/DecimalFactors.txt
            """  # noqa
            units = np.fromfile(fid, np.uint16, len(channels)).tolist()
            unitcodes = np.array(units[:])
            include = list()
            for i, unit in enumerate(units):
                if unit == 4275:  # microvolts
                    units[i] = 1e-6
                elif unit == 512:  # dimensionless
                    units[i] = 1
                elif unit == 0:
                    units[i] = 1  # unrecognized
                else:
                    warn('Unsupported physical dimension for channel %d '
                         '(assuming dimensionless). Please contact the '
                         'MNE-Python developers for support.' % i)
                    units[i] = 1
                include.append(i)

            ch_names = [ch_names[idx] for idx in include]
            physical_min = np.fromfile(fid, np.float64, len(channels))
            physical_max = np.fromfile(fid, np.float64, len(channels))
            digital_min = np.fromfile(fid, np.float64, len(channels))
            digital_max = np.fromfile(fid, np.float64, len(channels))

            fid.seek(68 * len(channels), 1)  # obsolete
            lowpass = np.fromfile(fid, np.float32, len(channels))
            highpass = np.fromfile(fid, np.float32, len(channels))
            notch = np.fromfile(fid, np.float32, len(channels))

            # number of samples per record
            n_samps = np.fromfile(fid, np.int32, len(channels))

            # data type
            dtype = np.fromfile(fid, np.int32, len(channels))

            channel = {}
            channel['xyz'] = [np.fromfile(fid, np.float32, 3)[0]
                              for ch in channels]

            if edf_info['number'] < 2.19:
                impedance = np.fromfile(fid, np.uint8,
                                        len(channels)).astype(float)
                impedance[impedance == 255] = np.nan
                channel['impedance'] = pow(2, impedance / 8)
                fid.seek(19 * len(channels), 1)  # reserved
            else:
                tmp = np.fromfile(fid, np.float32, 5 * len(channels))
                tmp = tmp[::5]
                fZ = tmp[:]
                impedance = tmp[:]
                # channels with no voltage (code 4256) data
                ch = [unitcodes & 65504 != 4256][0]
                impedance[np.where(ch)] = None
                # channel with no impedance (code 4288) data
                ch = [unitcodes & 65504 != 4288][0]
                fZ[np.where(ch)[0]] = None

            assert fid.tell() == header_nbytes

            # total number of bytes for data
            bytes_tot = np.sum([gdftype_byte[t] * n_samps[i]
                                for i, t in enumerate(dtype)])

            # Populate edf_info
            edf_info.update(
                bytes_tot=bytes_tot, ch_names=ch_names,
                data_offset=header_nbytes,
                dtype_byte=[gdftype_byte[t] for t in dtype],
                dtype_np=[gdftype_np[t] for t in dtype],
                digital_min=digital_min, digital_max=digital_max,
                exclude=exclude, gnd=gnd, highpass=highpass, include=include,
                impedance=impedance, lowpass=lowpass,
                meas_date=calendar.timegm(date.utctimetuple()),
                meas_id=meas_id, n_records=n_records, n_samps=n_samps,
                nchan=nchan, notch=notch, subject_info=patient,
                physical_max=physical_max, physical_min=physical_min,
                record_length=record_length, ref=ref, units=units)

            # EVENT TABLE
            # ------------------------------------------------------------------
            etp = edf_info['data_offset'] + edf_info['n_records'] * \
                edf_info['bytes_tot']
            fid.seek(etp)  # skip data to go to event table
            etmode = fid.read(1).decode()
            if etmode != '':
                etmode = np.fromstring(etmode, np.uint8).tolist()[0]

                if edf_info['number'] < 1.94:
                    sr = np.fromfile(fid, np.uint8, 3)
                    event_sr = sr[0]
                    for i in range(1, len(sr)):
                        event_sr = event_sr + sr[i] * 2**(i * 8)
                    n_events = np.fromfile(fid, np.uint32, 1)[0]
                else:
                    ne = np.fromfile(fid, np.uint8, 3)
                    n_events = ne[0]
                    for i in range(1, len(ne)):
                        n_events = n_events + ne[i] * 2**(i * 8)
                    event_sr = np.fromfile(fid, np.float32, 1)[0]

                pos = np.fromfile(fid, np.uint32, n_events) - 1  # 1-based inds
                typ = np.fromfile(fid, np.uint16, n_events)

                if etmode == 3:
                    chn = np.fromfile(fid, np.uint16, n_events)
                    dur = np.fromfile(fid, np.uint32, n_events)
                else:
                    chn = np.zeros(n_events, dtype=np.uint32)
                    dur = np.ones(n_events, dtype=np.uint32)
                np.clip(dur, 1, np.inf, out=dur)
                events = [n_events, pos, typ, chn, dur]
                edf_info['event_sfreq'] = event_sr

    if stim_channel == 'auto' and edf_info['nchan'] not in exclude:
        if len(events) == 0:
            warn('No events found. Cannot construct a stimulus channel.')
            edf_info['events'] = list()
            return edf_info
        edf_info['include'].append(edf_info['nchan'])
        edf_info['n_samps'] = np.append(edf_info['n_samps'], 0)
        edf_info['units'] = np.append(edf_info['units'], 1)
        edf_info['ch_names'] += [u'STI 014']
        edf_info['physical_min'] = np.append(edf_info['physical_min'], 0)
        edf_info['digital_min'] = np.append(edf_info['digital_min'], 0)
        vmax = np.max(events[2])
        edf_info['physical_max'] = np.append(edf_info['physical_max'], vmax)
        edf_info['digital_max'] = np.append(edf_info['digital_max'], vmax)

        data = np.zeros(np.max(n_samps * n_records))
        warn_overlap = False
        for samp, id, dur in zip(events[1], events[2], events[4]):
            if np.sum(data[samp:samp + dur]) > 0:
                warn_overlap = True  # Warn only once.
            data[samp:samp + dur] += id
        if warn_overlap:
            warn('Overlapping events detected. Use find_edf_events for the '
                 'original events.')
        edf_info['stim_data'] = data
    edf_info['events'] = events
    return edf_info


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
    pat = '([+/-]\\d+.\\d+),(\\w+)'
    annot = open(annot).read()
    triggers = re.findall(pat, annot)
    times, values = zip(*triggers)
    times = [float(time) * sfreq for time in times]

    pat = r'(\w+):(\d+)'
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


def _check_stim_channel(stim_channel, ch_names, include):
    """Check that the stimulus channel exists in the current datafile."""
    if isinstance(stim_channel, str):
        if stim_channel == 'auto':
            if 'auto' in ch_names:
                raise ValueError("'auto' exists as a channel name. Change "
                                 "stim_channel parameter!")
            stim_channel = len(include) - 1
        elif stim_channel not in ch_names:
            err = 'Could not find a channel named "{}" in datafile.' \
                  .format(stim_channel)
            casematch = [ch for ch in ch_names
                         if stim_channel.lower().replace(' ', '') ==
                         ch.lower().replace(' ', '')]
            if casematch:
                err += ' Closest match is "{}".'.format(casematch[0])
            raise ValueError(err)
    else:
        if stim_channel == -1:
            stim_channel = len(include) - 1
        elif stim_channel > len(ch_names):
            raise ValueError('Requested stim_channel index ({}) exceeds total '
                             'number of channels in datafile ({})'
                             .format(stim_channel, len(ch_names)))

    return stim_channel


def read_raw_edf(input_fname, montage=None, eog=None, misc=None,
                 stim_channel='auto', annot=None, annotmap=None, exclude=(),
                 preload=False, verbose=None):
    """Reader function for EDF+, BDF, GDF conversion to FIF.

    Parameters
    ----------
    input_fname : str
        Path to the EDF+, BDF, or GDF file.
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
    stim_channel : str | int | 'auto' | None
        The channel name or channel index (starting at 0). -1 corresponds to
        the last channel. If None, there will be no stim channel added. If
        'auto' (default), the stim channel will be added as the last channel if
        the header contains ``'EDF Annotations'`` or GDF events (otherwise stim
        channel will not be added).
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

    Notes
    -----
    Biosemi devices trigger codes are encoded in bits 1-16 of the status
    channel, whereas system codes (CMS in/out-of range, battery low, etc.) are
    coded in bits 16-23 (see http://www.biosemi.com/faq/trigger_signals.htm).
    To retrieve correct event values (bits 1-16), one could do:

        >>> events = mne.find_events(...)  # doctest:+SKIP
        >>> events[:, 2] >>= 8  # doctest:+SKIP

    It is also possible to retrieve system codes, but no particular effort has
    been made to decode these in MNE.

    For GDF files, the stimulus channel is constructed from the events in the
    header. You should use keyword ``stim_channel=-1`` to add it at the end of
    the channel list. The id numbers of overlapping events are simply combined
    through addition. To get the original events from the header, use method
    ``raw.find_edf_events``.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawEDF(input_fname=input_fname, montage=montage, eog=eog, misc=misc,
                  stim_channel=stim_channel, annot=annot, annotmap=annotmap,
                  exclude=exclude, preload=preload, verbose=verbose)
