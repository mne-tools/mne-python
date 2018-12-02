# -*- coding: utf-8 -*-
"""Conversion tool from EDF, EDF+, BDF to FIF."""

# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Martin Billinger <martin.billinger@tugraz.at>
#          Nicolas Barascud <nicolas.barascud@ens.fr>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Joan Massich <mailsik@gmail.com>
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
from ..meas_info import _empty_info, DATE_NONE
from ..constants import FIFF
from ...filter import resample
from ...utils import copy_function_doc_to_method_doc
from ...annotations import Annotations


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
    stim_channel : 'auto' | None | int
        If 'auto' or None, the stim channels present will be read. Channels
        names 'Status' or 'STATUS' will be considered stim channels
        in 'auto' mode. If an int provided then it's the index of the
        stim channel, e.g. -1 for the last channel in the file.

        .. warning:: 0.18 does not allow for stim channel synthesis from
                     the TAL channel called 'EDF Annotations' or
                     'BDF Annotations'. The TAL channel is parsed
                     and put in the raw.annotations attribute.
                     Use :func:`mne.events_from_annotations` to obtain
                     events for the annotations instead.

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
                 stim_channel=None, exclude=(), preload=False,
                 verbose=None):  # noqa: D102
        logger.info('Extracting EDF parameters from %s...' % input_fname)
        input_fname = os.path.abspath(input_fname)
        info, edf_info, orig_units = _get_info(input_fname,
                                               stim_channel, eog, misc,
                                               exclude, preload)
        logger.info('Creating raw.info structure...')
        _check_update_montage(info, montage)

        # Raw attributes
        last_samps = [edf_info['nsamples'] - 1]
        super(RawEDF, self).__init__(
            info, preload, filenames=[input_fname], raw_extras=[edf_info],
            last_samps=last_samps, orig_format='int', orig_units=orig_units,
            verbose=verbose)

        # Read annotations from file and set it
        annot = None
        ext = os.path.splitext(input_fname)[1][1:].lower()
        if ext in ('gdf'):
            events = edf_info.get('events', None)
            # Annotations in GDF: events are stored as the following
            # list: `events = [n_events, pos, typ, chn, dur]` where pos is the
            # latency, dur is the duration in samples. They both are
            # numpy.ndarray
            if events is not None and events[1].shape[0] > 0:
                # For whatever reason, typ has the same content as pos
                # therefore we set an arbitrary description
                desc = 'GDF event'
                annot = Annotations(onset=events[1] / self.info['sfreq'],
                                    duration=events[4] / self.info['sfreq'],
                                    description=desc,
                                    orig_time=None)
        elif len(edf_info['tal_idx']) > 0:
            # XXX : should pass to _read_annotations_edf what channel to read
            # ie tal_channel_idx
            onset, duration, desc = _read_annotations_edf(input_fname)
            if onset:
                # in EDF, annotations are relative to first_samp
                annot = Annotations(onset=onset, duration=duration,
                                    description=desc, orig_time=None)

        if annot is not None:
            self.set_annotations(annot)

    @verbose
    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        from scipy.interpolate import interp1d

        if mult is not None:
            # XXX "cals" here does not function the same way as in RawFIF,
            # and for efficiency we want to be able to combine mult and cals
            # so proj support will have to wait until this is resolved
            raise NotImplementedError('mult is not supported yet')
        n_samps = self._raw_extras[fi]['n_samps']
        buf_len = int(self._raw_extras[fi]['max_samp'])
        dtype = self._raw_extras[fi]['dtype_np']
        dtype_byte = self._raw_extras[fi]['dtype_byte']
        data_offset = self._raw_extras[fi]['data_offset']
        stim_channel = self._raw_extras[fi]['stim_channel']
        orig_sel = self._raw_extras[fi]['sel']
        subtype = self._raw_extras[fi]['subtype']

        if np.size(dtype_byte) > 1:
            if len(np.unique(dtype_byte)) > 1:
                warn("Multiple data type not supported")
            dtype = dtype[0]
            dtype_byte = dtype_byte[0]

        # gain constructor
        physical_range = np.array([ch['range'] for ch in self.info['chs']])
        cal = np.array([ch['cal'] for ch in self.info['chs']])
        assert cal.shape == (len(self.info['chs']),)
        cal = np.atleast_2d(physical_range / cal)  # physical / digital
        gains = np.atleast_2d(self._raw_extras[fi]['units'])

        # physical dimension in uV
        physical_min = self._raw_extras[fi]['physical_min']
        digital_min = self._raw_extras[fi]['digital_min']

        offsets = np.atleast_2d(physical_min - (digital_min * cal)).T
        this_sel = orig_sel[idx]

        # We could read this one EDF block at a time, which would be this:
        ch_offsets = np.cumsum(np.concatenate([[0], n_samps]), dtype=np.int64)
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
                for ii, ci in enumerate(this_sel):
                    # This now has size (n_chunks_read, n_samp[ci])
                    ch_data = many_chunk[:, ch_offsets[ci]:ch_offsets[ci + 1]]
                    r_sidx = r_lims[ai][0]
                    r_eidx = (buf_len * (n_read - 1) +
                              r_lims[ai + n_read - 1][1])
                    d_sidx = d_lims[ai][0]
                    d_eidx = d_lims[ai + n_read - 1][1]
                    if n_samps[ci] != buf_len:
                        if ci == stim_channel:
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

        # only try to read the stim channel if it's not None and it's
        # actually one of the requested channels
        _idx = np.arange(self.info['nchan'])[idx]  # slice -> ints
        if stim_channel is None:  # avoid NumPy comparison to None
            stim_channel_idx = np.array([], int)
        else:
            stim_channel_idx = np.where(_idx == stim_channel)[0]

        if subtype == 'bdf':
            # do not scale stim channel (see gh-5160)
            if stim_channel is None:
                stim_idx = [[]]
            else:
                stim_idx = np.where(np.arange(self.info['nchan']) ==
                                    stim_channel)
            cal[0, stim_idx[0]] = 1
            offsets[stim_idx[0], 0] = 0
            gains[0, stim_idx[0]] = 1
        data *= cal.T[idx]
        data += offsets[idx]
        data *= gains.T[idx]

        if stim_channel is not None and len(stim_channel_idx) > 0:
            stim = np.bitwise_and(data[stim_channel_idx].astype(int),
                                  2**17 - 1)
            data[stim_channel_idx, :] = stim

    @copy_function_doc_to_method_doc(find_edf_events)
    def find_edf_events(self):
        return self._raw_extras[0]['events']


def _read_ch(fid, subtype, samp, dtype_byte, dtype=None):
    """Read a number of samples for a single channel."""
    # BDF
    if subtype == 'bdf':
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


def _get_info(fname, stim_channel, eog, misc, exclude, preload):
    """Extract all the information from the EDF+, BDF or GDF file."""
    if stim_channel is not None:
        if stim_channel:
            _msg = ('The synthesis of the stim channel is not supported since'
                    ' 0.18. Please set `stim_channel` to False and use'
                    ' `mne.events_from_annotations` instead')
            raise RuntimeError(_msg)
        else:
            warn('stim_channel parameter is deprecated and will be removed in'
                 ' 0.19.', DeprecationWarning)
            stim_channel = None

    if eog is None:
        eog = []
    if misc is None:
        misc = []

    # Read header from file
    ext = os.path.splitext(fname)[1][1:].lower()
    logger.info('%s file detected' % ext.upper())
    if ext in ('bdf', 'edf'):
        edf_info, orig_units = _read_edf_header(fname, exclude)
    elif ext in ('gdf'):
        edf_info = _read_gdf_header(fname, stim_channel, exclude)

        # orig_units not yet implemented for gdf
        orig_units = None

    else:
        raise NotImplementedError(
            'Only GDF, EDF, and BDF files are supported, got %s.' % ext)

    sel = edf_info['sel']  # selection of channels not excluded
    ch_names = edf_info['ch_names']  # of length len(sel)
    n_samps = edf_info['n_samps'][sel]
    nchan = edf_info['nchan']
    physical_ranges = edf_info['physical_max'] - edf_info['physical_min']
    cals = edf_info['digital_max'] - edf_info['digital_min']
    bad_idx = np.where((~np.isfinite(cals)) | (cals == 0))[0]
    if len(bad_idx) > 0:
        warn('Scaling factor is not defined in following channels:\n' +
             ', '.join(ch_names[i] for i in bad_idx))
        cals[bad_idx] = 1
    bad_idx = np.where(physical_ranges == 0)[0]
    if len(bad_idx) > 0:
        warn('Physical range is not defined in following channels:\n' +
             ', '.join(ch_names[i] for i in bad_idx))
        physical_ranges[bad_idx] = 1
    stim_channel, stim_ch_name = \
        _check_stim_channel(stim_channel, ch_names, sel)

    # Creates a list of dicts of eeg channels for raw.info
    logger.info('Setting channel info structure...')
    chs = list()
    pick_mask = np.ones(len(ch_names))

    for idx, ch_info in enumerate(zip(ch_names, physical_ranges, cals)):
        ch_name, physical_range, cal = ch_info
        chan_info = {}
        logger.debug('  %s: range=%s cal=%s' % (ch_name, physical_range, cal))
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
        elif ch_name in misc or idx in misc or idx - nchan in misc:
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['kind'] = FIFF.FIFFV_MISC_CH
            pick_mask[idx] = False
        elif stim_channel == idx:
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['unit'] = FIFF.FIFF_UNIT_NONE
            chan_info['kind'] = FIFF.FIFFV_STIM_CH
            pick_mask[idx] = False
            chan_info['ch_name'] = ch_name
            ch_names[idx] = chan_info['ch_name']
            edf_info['units'][idx] = 1
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
            hp = highpass[0]
            try:
                hp = float(hp)
            except Exception:
                hp = 0.
            info['highpass'] = hp
    else:
        info['highpass'] = float(np.max(highpass))
        warn('Channels contain different highpass filters. Highest filter '
             'setting will be stored.')
    if np.isnan(info['highpass']):
        info['highpass'] = 0.
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
    if np.isnan(info['lowpass']):
        info['lowpass'] = info['sfreq'] / 2.

    # Some keys to be consistent with FIF measurement info
    info['description'] = None
    edf_info['nsamples'] = int(edf_info['n_records'] * max_samp)

    info._update_redundant()

    return info, edf_info, orig_units


def _read_edf_header(fname, exclude):
    """Read header information from EDF+ or BDF file."""
    edf_info = {'events': []}

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
        meas_date = (calendar.timegm(date.utctimetuple()), 0)

        header_nbytes = int(fid.read(8).decode())

        # The following 44 bytes sometimes identify the file type, but this is
        # not guaranteed. Therefore, we skip this field and use the file
        # extension to determine the subtype (EDF or BDF, which differ in the
        # number of bytes they use for the data records; EDF uses 2 bytes
        # whereas BDF uses 3 bytes).
        fid.read(44)
        subtype = os.path.splitext(fname)[1][1:].lower()

        n_records = int(fid.read(8).decode())
        record_length = fid.read(8).decode().strip('\x00').strip()
        record_length = np.array([float(record_length), 1.])  # in seconds
        if record_length[0] == 0:
            record_length = record_length[0] = 1.
            warn('Header information is incorrect for record length. Default '
                 'record length set to 1.')

        nchan = int(fid.read(4).decode())
        channels = list(range(nchan))
        ch_names = [fid.read(16).strip().decode() for ch in channels]
        exclude = _find_exclude_idx(ch_names, exclude)
        tal_idx = _find_tal_idx(ch_names)
        exclude = np.concatenate([exclude, tal_idx])
        sel = np.setdiff1d(np.arange(len(ch_names)), exclude)
        for ch in channels:
            fid.read(80)  # transducer
        units = [fid.read(8).strip().decode() for ch in channels]
        orig_units = dict(zip(ch_names, units))
        edf_info['units'] = list()
        for i, unit in enumerate(units):
            if i in exclude:
                continue
            if unit == 'uV':
                edf_info['units'].append(1e-6)
            else:
                edf_info['units'].append(1)
        ch_names = [ch_names[idx] for idx in sel]

        physical_min = np.array([float(fid.read(8).decode())
                                 for ch in channels])[sel]
        physical_max = np.array([float(fid.read(8).decode())
                                 for ch in channels])[sel]
        digital_min = np.array([float(fid.read(8).decode())
                                for ch in channels])[sel]
        digital_max = np.array([float(fid.read(8).decode())
                                for ch in channels])[sel]
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
            digital_max=digital_max, digital_min=digital_min,
            highpass=highpass, sel=sel, lowpass=lowpass, meas_date=meas_date,
            n_records=n_records, n_samps=n_samps, nchan=nchan,
            subject_info=patient, physical_max=physical_max,
            physical_min=physical_min, record_length=record_length,
            subtype=subtype, tal_idx=tal_idx)

        fid.read(32 * nchan).decode()  # reserved
        assert fid.tell() == header_nbytes

        fid.seek(0, 2)
        n_bytes = fid.tell()
        n_data_bytes = n_bytes - header_nbytes
        total_samps = (n_data_bytes // 3 if subtype == 'bdf'
                       else n_data_bytes // 2)
        read_records = total_samps // np.sum(n_samps)
        if n_records != read_records:
            warn('Number of records from the header does not match the file '
                 'size (perhaps the recording was not stopped before exiting).'
                 ' Inferring from the file size.')
            edf_info['n_records'] = n_records = read_records

        if subtype == 'bdf':
            edf_info['dtype_byte'] = 3  # 24-bit (3 byte) integers
            edf_info['dtype_np'] = np.uint8
        else:
            edf_info['dtype_byte'] = 2  # 16-bit (2 byte) integers
            edf_info['dtype_np'] = np.int16

    return edf_info, orig_units


def _read_gdf_header(fname, stim_channel, exclude):
    """Read GDF 1.x and GDF 2.x header info."""
    edf_info = dict()
    events = None
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
        meas_date = DATE_NONE

        # GDF 1.x
        # ---------------------------------------------------------------------
        if edf_info['number'] < 1.9:

            # patient ID
            pid = fid.read(80).decode('latin-1')
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
                date = datetime.datetime(int(tm[0:4]), int(tm[4:6]),
                                         int(tm[6:8]), int(tm[8:10]),
                                         int(tm[10:12]), int(tm[12:14]),
                                         int(tm[14:16]) * pow(10, 4))
                meas_date = (calendar.timegm(date.utctimetuple()), 0)
            except Exception:
                pass

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
            ch_names = [fid.read(16).decode('latin-1').strip(' \x00')
                        for ch in channels]
            fid.seek(80 * len(channels), 1)  # transducer
            units = [fid.read(8).decode('latin-1').strip(' \x00')
                     for ch in channels]
            exclude = _find_exclude_idx(ch_names, exclude)
            sel = list()
            for i, unit in enumerate(units):
                if unit[:2] == 'uV':
                    units[i] = 1e-6
                else:
                    units[i] = 1
                sel.append(i)

            ch_names = [ch_names[idx] for idx in sel]
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
                highpass=highpass, sel=sel, lowpass=lowpass,
                meas_date=meas_date,
                meas_id=meas_id, n_records=n_records, n_samps=n_samps,
                nchan=nchan, subject_info=patient, physical_max=physical_max,
                physical_min=physical_min, record_length=record_length,
                units=units)

            fid.seek(32 * edf_info['nchan'], 1)  # reserved
            assert fid.tell() == header_nbytes

            # Event table
            # -----------------------------------------------------------------
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
        # ---------------------------------------------------------------------
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
            if date != 0:
                date = datetime.datetime(1, 1, 1) + \
                    datetime.timedelta(date * pow(2, -32) - 367)
                meas_date = (calendar.timegm(date.utctimetuple()), 0)

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
            exclude = _find_exclude_idx(ch_names, exclude)

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
            sel = list()
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
                sel.append(i)

            ch_names = [ch_names[idx] for idx in sel]
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
                exclude=exclude, gnd=gnd, highpass=highpass, sel=sel,
                impedance=impedance, lowpass=lowpass, meas_date=meas_date,
                meas_id=meas_id, n_records=n_records, n_samps=n_samps,
                nchan=nchan, notch=notch, subject_info=patient,
                physical_max=physical_max, physical_min=physical_min,
                record_length=record_length, ref=ref, units=units)

            # EVENT TABLE
            # -----------------------------------------------------------------
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

    edf_info.update(events=events, sel=np.arange(len(edf_info['ch_names'])))

    return edf_info


def _check_stim_channel(stim_channel, ch_names, sel):
    """Check that the stimulus channel exists in the current datafile."""
    if stim_channel is False:
        return None, None
    if stim_channel is None:
        stim_channel = 'auto'

    if isinstance(stim_channel, str):
        if stim_channel == 'auto':
            if 'auto' in ch_names:
                raise ValueError("'auto' exists as a channel name. Change "
                                 "stim_channel parameter!")
            if 'STATUS' in ch_names:
                stim_channel_idx = ch_names.index('STATUS')
            elif 'Status' in ch_names:
                stim_channel_idx = ch_names.index('Status')
            else:
                stim_channel_idx = None
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
            stim_channel_idx = len(sel) - 1
        elif stim_channel > len(ch_names):
            raise ValueError('Requested stim_channel index ({}) exceeds total '
                             'number of channels in datafile ({})'
                             .format(stim_channel, len(ch_names)))

    if stim_channel_idx is None:
        stim_ch_name = None
    else:
        stim_ch_name = ch_names[stim_channel_idx]

    return stim_channel_idx, stim_ch_name


def _find_exclude_idx(ch_names, exclude):
    """Find the index of all channels to exclude.

    If there are several channels called "A" and we want to exclude "A",
    then add (the index of) all "A" channels to the exclusion list.
    """
    return [idx for idx, ch in enumerate(ch_names) if ch in exclude]


def _find_tal_idx(ch_names):
    # Annotations / TAL Channels
    accepted_tal_ch_names = ['EDF Annotations', 'BDF Annotations']
    tal_channel_idx = np.where(np.in1d(ch_names, accepted_tal_ch_names))[0]
    return tal_channel_idx


def read_raw_edf(input_fname, montage=None, eog=None, misc=None,
                 stim_channel=None, exclude=(), preload=False, verbose=None):
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
    stim_channel : 'auto' | None | int
        If 'auto' or None, the stim channels present will be read. Channels
        names 'Status' or 'STATUS' will be considered stim channels
        in 'auto' mode. If an int provided then it's the index of the
        stim channel, e.g. -1 for the last channel in the file.

        .. warning:: 0.18 does not allow for stim channel synthesis from
                     the TAL channel called 'EDF Annotations' or
                     'BDF Annotations'. The TAL channel is parsed
                     and put in the raw.annotations attribute.
                     Use :func:`mne.events_from_annotations` to obtain
                     events for the annotations instead.

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
                  stim_channel=stim_channel, exclude=exclude, preload=preload,
                  verbose=verbose)


def _read_annotations_edf(annotations):
    """Annotation File Reader.

    Parameters
    ----------
    annotations : ndarray (n_chans, n_samples) | str
        Channel data in EDF+ TAL format or path to annotation file.

    Returns
    -------
    onset : array of float, shape (n_annotations,)
        The starting time of annotations in seconds after ``orig_time``.
    duration : array of float, shape (n_annotations,)
        Durations of the annotations in seconds.
    description : array of str, shape (n_annotations,)
        Array of strings containing description for each annotation. If a
        string, all the annotations are given the same description. To reject
        epochs, use description starting with keyword 'bad'. See example above.
    """
    pat = '([+-]\\d+\\.?\\d*)(\x15(\\d+\\.?\\d*))?(\x14.*?)\x14\x00'
    if isinstance(annotations, str):
        with open(annotations, encoding='latin-1') as annot_file:
            triggers = re.findall(pat, annot_file.read())
    else:
        tals = bytearray()
        for chan in annotations:
            for s in chan:
                i = int(s)
                tals.extend(np.uint8([i % 256, i // 256]))
        # use of latin-1 because characters are only encoded for the first 256
        # code points and utf-8 can triggers an "invalid continuation byte"
        # error
        triggers = re.findall(pat, tals.decode('latin-1'))

    events = []
    for ev in triggers:
        onset = float(ev[0])
        duration = float(ev[2]) if ev[2] else 0
        for description in ev[3].split('\x14')[1:]:
            if description:
                events.append([onset, duration, description])

    return zip(*events) if events else (list(), list(), list())


def _get_edf_default_event_id(descriptions):
    mapping = dict((a, n) for n, a in
                   enumerate(sorted(set(descriptions)), start=1))
    return mapping
