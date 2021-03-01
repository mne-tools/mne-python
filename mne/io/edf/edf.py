# -*- coding: utf-8 -*-
"""Reading tools from EDF, EDF+, BDF, and GDF."""

# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Martin Billinger <martin.billinger@tugraz.at>
#          Nicolas Barascud <nicolas.barascud@ens.fr>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Joan Massich <mailsik@gmail.com>
#          Clemens Brunner <clemens.brunner@gmail.com>
#          Jeroen Van Der Donckt (IDlab - imec) <jeroen.vanderdonckt@ugent.be>
#
# License: BSD (3-clause)

from datetime import datetime, timezone, timedelta
import os
import re

import numpy as np

from ...utils import verbose, logger, warn
from ..utils import _blk_read_lims, _mult_cal_one
from ..base import BaseRaw
from ..meas_info import _empty_info, _unique_channel_names
from ..constants import FIFF
from ...filter import resample
from ...utils import fill_doc
from ...annotations import Annotations


@fill_doc
class RawEDF(BaseRaw):
    """Raw object from EDF, EDF+ or BDF file.

    Parameters
    ----------
    input_fname : str
        Path to the EDF, EDF+ or BDF file.
    eog : list or tuple
        Names of channels or list of indices that should be designated EOG
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    misc : list or tuple
        Names of channels or list of indices that should be designated MISC
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    stim_channel : 'auto' | str | list of str | int | list of int
        Defaults to 'auto', which means that channels named 'status' or
        'trigger' (case insensitive) are set to STIM. If str (or list of str),
        all channels matching the name(s) are set to STIM. If int (or list of
        ints), the channels corresponding to the indices are set to STIM.

        .. warning:: 0.18 does not allow for stim channel synthesis from TAL
                     channels called 'EDF Annotations' or 'BDF Annotations'
                     anymore. Instead, TAL channels are parsed and extracted
                     annotations are stored in raw.annotations. Use
                     :func:`mne.events_from_annotations` to obtain events from
                     these annotations.

    exclude : list of str
        Channel names to exclude. This can help when reading data with
        different sampling rates to avoid unnecessary resampling.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    mne.io.read_raw_edf : Recommended way to read EDF/EDF+ files.
    mne.io.read_raw_bdf : Recommended way to read BDF files.

    Notes
    -----
    Biosemi devices trigger codes are encoded in 16-bit format, whereas system
    codes (CMS in/out-of range, battery low, etc.) are coded in bits 16-23 of
    the status channel (see http://www.biosemi.com/faq/trigger_signals.htm).
    To retrieve correct event values (bits 1-16), one could do:

        >>> events = mne.find_events(...)  # doctest:+SKIP
        >>> events[:, 2] &= (2**16 - 1)  # doctest:+SKIP

    The above operation can be carried out directly in :func:`mne.find_events`
    using the ``mask`` and ``mask_type`` parameters (see
    :func:`mne.find_events` for more details).

    It is also possible to retrieve system codes, but no particular effort has
    been made to decode these in MNE. In case it is necessary, for instance to
    check the CMS bit, the following operation can be carried out:

        >>> cms_bit = 20  # doctest:+SKIP
        >>> cms_high = (events[:, 2] & (1 << cms_bit)) != 0  # doctest:+SKIP

    It is worth noting that in some special cases, it may be necessary to shift
    event values in order to retrieve correct event triggers. This depends on
    the triggering device used to perform the synchronization. For instance, in
    some files events need to be shifted by 8 bits:

        >>> events[:, 2] >>= 8  # doctest:+SKIP

    TAL channels called 'EDF Annotations' or 'BDF Annotations' are parsed and
    extracted annotations are stored in raw.annotations. Use
    :func:`mne.events_from_annotations` to obtain events from these
    annotations.

    If channels named 'status' or 'trigger' are present, they are considered as
    STIM channels by default. Use func:`mne.find_events` to parse events
    encoded in such analog stim channels.
    """

    @verbose
    def __init__(self, input_fname, eog=None, misc=None,
                 stim_channel='auto', exclude=(), preload=False, verbose=None):
        logger.info('Extracting EDF parameters from {}...'.format(input_fname))
        input_fname = os.path.abspath(input_fname)
        info, edf_info, orig_units = _get_info(input_fname,
                                               stim_channel, eog, misc,
                                               exclude, preload)
        logger.info('Creating raw.info structure...')

        # Raw attributes
        last_samps = [edf_info['nsamples'] - 1]
        super().__init__(info, preload, filenames=[input_fname],
                         raw_extras=[edf_info], last_samps=last_samps,
                         orig_format='int', orig_units=orig_units,
                         verbose=verbose)

        # Read annotations from file and set it
        onset, duration, desc = list(), list(), list()
        if len(edf_info['tal_idx']) > 0:
            # Read TAL data exploiting the header info (no regexp)
            idx = np.empty(0, int)
            tal_data = self._read_segment_file(
                np.empty((0, self.n_times)), idx, 0, 0, int(self.n_times),
                np.ones((len(idx), 1)), None)
            onset, duration, desc = _read_annotations_edf(tal_data[0])

        self.set_annotations(Annotations(onset=onset, duration=duration,
                                         description=desc, orig_time=None))

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        return _read_segment_file(data, idx, fi, start, stop,
                                  self._raw_extras[fi], self._filenames[fi],
                                  cals, mult)


@fill_doc
class RawGDF(BaseRaw):
    """Raw object from GDF file.

    Parameters
    ----------
    input_fname : str
        Path to the GDF file.
    eog : list or tuple
        Names of channels or list of indices that should be designated EOG
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    misc : list or tuple
        Names of channels or list of indices that should be designated MISC
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    stim_channel : 'auto' | str | list of str | int | list of int
        Defaults to 'auto', which means that channels named 'status' or
        'trigger' (case insensitive) are set to STIM. If str (or list of str),
        all channels matching the name(s) are set to STIM. If int (or list of
        ints), channels corresponding to the indices are set to STIM.
    exclude : list of str
        Channel names to exclude. This can help when reading data with
        different sampling rates to avoid unnecessary resampling.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    mne.io.read_raw_gdf : Recommended way to read GDF files.

    Notes
    -----
    If channels named 'status' or 'trigger' are present, they are considered as
    STIM channels by default. Use func:`mne.find_events` to parse events
    encoded in such analog stim channels.
    """

    @verbose
    def __init__(self, input_fname, eog=None, misc=None,
                 stim_channel='auto', exclude=(), preload=False, verbose=None):
        logger.info('Extracting EDF parameters from {}...'.format(input_fname))
        input_fname = os.path.abspath(input_fname)
        info, edf_info, orig_units = _get_info(input_fname,
                                               stim_channel, eog, misc,
                                               exclude, preload)
        logger.info('Creating raw.info structure...')

        # Raw attributes
        last_samps = [edf_info['nsamples'] - 1]
        super().__init__(info, preload, filenames=[input_fname],
                         raw_extras=[edf_info], last_samps=last_samps,
                         orig_format='int', orig_units=orig_units,
                         verbose=verbose)

        # Read annotations from file and set it
        onset, duration, desc = _get_annotations_gdf(edf_info,
                                                     self.info['sfreq'])

        self.set_annotations(Annotations(onset=onset, duration=duration,
                                         description=desc, orig_time=None))

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        return _read_segment_file(data, idx, fi, start, stop,
                                  self._raw_extras[fi], self._filenames[fi],
                                  cals, mult)


def _read_ch(fid, subtype, samp, dtype_byte, dtype=None):
    """Read a number of samples for a single channel."""
    # BDF
    if subtype == 'bdf':
        ch_data = np.fromfile(fid, dtype=dtype, count=samp * dtype_byte)
        ch_data = ch_data.reshape(-1, 3).astype(INT32)
        ch_data = ((ch_data[:, 0]) +
                   (ch_data[:, 1] << 8) +
                   (ch_data[:, 2] << 16))
        # 24th bit determines the sign
        ch_data[ch_data >= (1 << 23)] -= (1 << 24)

    # GDF data and EDF data
    else:
        ch_data = np.fromfile(fid, dtype=dtype, count=samp)

    return ch_data


def _read_segment_file(data, idx, fi, start, stop, raw_extras, filenames,
                       cals, mult):
    """Read a chunk of raw data."""
    from scipy.interpolate import interp1d

    n_samps = raw_extras['n_samps']
    buf_len = int(raw_extras['max_samp'])
    dtype = raw_extras['dtype_np']
    dtype_byte = raw_extras['dtype_byte']
    data_offset = raw_extras['data_offset']
    stim_channel_idxs = raw_extras['stim_channel_idxs']
    orig_sel = raw_extras['sel']
    tal_idx = raw_extras.get('tal_idx', np.empty(0, int))
    subtype = raw_extras['subtype']
    cal = raw_extras['cal']
    offsets = raw_extras['offsets']
    gains = raw_extras['units']

    read_sel = np.concatenate([orig_sel[idx], tal_idx])
    tal_data = []

    # only try to read the stim channel if it's not None and it's
    # actually one of the requested channels
    idx_arr = np.arange(idx.start, idx.stop) if isinstance(idx, slice) else idx

    # We could read this one EDF block at a time, which would be this:
    ch_offsets = np.cumsum(np.concatenate([[0], n_samps]), dtype=np.int64)
    block_start_idx, r_lims, d_lims = _blk_read_lims(start, stop, buf_len)
    # But to speed it up, we really need to read multiple blocks at once,
    # Otherwise we can end up with e.g. 18,181 chunks for a 20 MB file!
    # Let's do ~10 MB chunks:
    n_per = max(10 * 1024 * 1024 // (ch_offsets[-1] * dtype_byte), 1)
    with open(filenames, 'rb', buffering=0) as fid:

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
            r_sidx = r_lims[ai][0]
            r_eidx = (buf_len * (n_read - 1) + r_lims[ai + n_read - 1][1])
            d_sidx = d_lims[ai][0]
            d_eidx = d_lims[ai + n_read - 1][1]
            one = np.zeros((len(orig_sel), d_eidx - d_sidx), dtype=data.dtype)
            for ii, ci in enumerate(read_sel):
                # This now has size (n_chunks_read, n_samp[ci])
                ch_data = many_chunk[:, ch_offsets[ci]:ch_offsets[ci + 1]]

                if ci in tal_idx:
                    tal_data.append(ch_data)
                    continue

                orig_idx = idx_arr[ii]
                ch_data = ch_data * cal[orig_idx]
                ch_data += offsets[orig_idx]
                ch_data *= gains[orig_idx]

                assert ci == orig_sel[orig_idx]

                if n_samps[ci] != buf_len:
                    if orig_idx in stim_channel_idxs:
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
                            ch_data.astype(np.float64), buf_len, n_samps[ci],
                            npad=0, axis=-1)
                elif orig_idx in stim_channel_idxs:
                    ch_data = np.bitwise_and(ch_data.astype(int), 2**17 - 1)
                one[orig_idx] = ch_data.ravel()[r_sidx:r_eidx]
            _mult_cal_one(data[:, d_sidx:d_eidx], one, idx, cals, mult)

    if len(tal_data) > 1:
        tal_data = np.concatenate([tal.ravel() for tal in tal_data])
        tal_data = tal_data[np.newaxis, :]
    return tal_data


def _read_header(fname, exclude):
    """Unify edf, bdf and gdf _read_header call.

    Parameters
    ----------
    fname : str
        Path to the EDF+, BDF, or GDF file.
    exclude : list of str
        Channel names to exclude. This can help when reading data with
        different sampling rates to avoid unnecessary resampling.

    Returns
    -------
    (edf_info, orig_units) : tuple
    """
    ext = os.path.splitext(fname)[1][1:].lower()
    logger.info('%s file detected' % ext.upper())
    if ext in ('bdf', 'edf'):
        return _read_edf_header(fname, exclude)
    elif ext == 'gdf':
        return _read_gdf_header(fname, exclude), None
    else:
        raise NotImplementedError(
            f'Only GDF, EDF, and BDF files are supported, got {ext}.')


def _get_info(fname, stim_channel, eog, misc, exclude, preload):
    """Extract all the information from the EDF+, BDF or GDF file."""
    eog = eog if eog is not None else []
    misc = misc if misc is not None else []

    edf_info, orig_units = _read_header(fname, exclude)

    # XXX: `tal_ch_names` to pass to `_check_stim_channel` should be computed
    #      from `edf_info['ch_names']` and `edf_info['tal_idx']` but 'tal_idx'
    #      contains stim channels that are not TAL.
    stim_channel_idxs, _ = _check_stim_channel(
        stim_channel, edf_info['ch_names'])

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

    # Creates a list of dicts of eeg channels for raw.info
    logger.info('Setting channel info structure...')
    chs = list()
    pick_mask = np.ones(len(ch_names))

    for idx, ch_name in enumerate(ch_names):
        chan_info = {}
        chan_info['cal'] = 1.
        chan_info['logno'] = idx + 1
        chan_info['scanno'] = idx + 1
        chan_info['range'] = 1.
        chan_info['unit_mul'] = FIFF.FIFF_UNITM_NONE
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
        elif idx in stim_channel_idxs:
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['unit'] = FIFF.FIFF_UNIT_NONE
            chan_info['kind'] = FIFF.FIFFV_STIM_CH
            pick_mask[idx] = False
            chan_info['ch_name'] = ch_name
            ch_names[idx] = chan_info['ch_name']
            edf_info['units'][idx] = 1
        chs.append(chan_info)

    edf_info['stim_channel_idxs'] = stim_channel_idxs

    if any(pick_mask):
        picks = [item for item, mask in zip(range(nchan), pick_mask) if mask]
        edf_info['max_samp'] = max_samp = n_samps[picks].max()
    else:
        edf_info['max_samp'] = max_samp = n_samps.max()

    # Info structure
    # -------------------------------------------------------------------------

    not_stim_ch = [x for x in range(n_samps.shape[0])
                   if x not in stim_channel_idxs]
    sfreq = np.take(n_samps, not_stim_ch).max() * \
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
        if lowpass[0] in ('NaN', '0', '0.0'):
            pass  # Placeholder for future use. Lowpass set in _empty_info.
        else:
            info['lowpass'] = float(lowpass[0])
    else:
        info['lowpass'] = float(np.min(lowpass))
        warn('Channels contain different lowpass filters. Lowest filter '
             'setting will be stored.')
    if np.isnan(info['lowpass']):
        info['lowpass'] = info['sfreq'] / 2.

    if info['highpass'] > info['lowpass']:
        warn(f'Highpass cutoff frequency {info["highpass"]} is greater than '
             f'lowpass cutoff frequency {info["lowpass"]}, '
             'setting values to 0 and Nyquist.')
        info['highpass'] = 0.
        info['lowpass'] = info['sfreq'] / 2.

    # Some keys to be consistent with FIF measurement info
    info['description'] = None
    edf_info['nsamples'] = int(edf_info['n_records'] * max_samp)

    info._update_redundant()

    # Later used for reading
    edf_info['cal'] = physical_ranges / cals

    # physical dimension in µV
    edf_info['offsets'] = (
        edf_info['physical_min'] - edf_info['digital_min'] * edf_info['cal'])
    del edf_info['physical_min']
    del edf_info['digital_min']

    if edf_info['subtype'] == 'bdf':
        edf_info['cal'][stim_channel_idxs] = 1
        edf_info['offsets'][stim_channel_idxs] = 0
        edf_info['units'][stim_channel_idxs] = 1

    return info, edf_info, orig_units


def _parse_prefilter_string(prefiltering):
    """Parse prefilter string from EDF+ and BDF headers."""
    highpass = np.array(
        [v for hp in [re.findall(r'HP:\s*([0-9]+[.]*[0-9]*)', filt)
                      for filt in prefiltering] for v in hp]
    )
    lowpass = np.array(
        [v for hp in [re.findall(r'LP:\s*([0-9]+[.]*[0-9]*)', filt)
                      for filt in prefiltering] for v in hp]
    )
    return highpass, lowpass


def _edf_str(x):
    return x.decode('latin-1').split('\x00')[0]


def _read_edf_header(fname, exclude):
    """Read header information from EDF+ or BDF file."""
    edf_info = {'events': []}

    with open(fname, 'rb') as fid:

        fid.read(8)  # version (unused here)

        # patient ID
        patient = {}
        id_info = fid.read(80).decode('latin-1').rstrip()
        id_info = id_info.split(' ')
        if len(id_info):
            patient['id'] = id_info[0]
            if len(id_info) == 4:
                try:
                    birthdate = datetime.strptime(id_info[2], "%d-%b-%Y")
                except ValueError:
                    birthdate = "X"
                patient['sex'] = id_info[1]
                patient['birthday'] = birthdate
                patient['name'] = id_info[3]

        # Recording ID
        meas_id = {}
        rec_info = fid.read(80).decode('latin-1').rstrip().split(' ')
        valid_startdate = False
        if len(rec_info) == 5:
            try:
                startdate = datetime.strptime(rec_info[1], "%d-%b-%Y")
            except ValueError:
                startdate = "X"
            else:
                valid_startdate = True
            meas_id['startdate'] = startdate
            meas_id['study_id'] = rec_info[2]
            meas_id['technician'] = rec_info[3]
            meas_id['equipment'] = rec_info[4]

        # If startdate available in recording info, use it instead of the
        # file's meas_date since it contains all 4 digits of the year
        if valid_startdate:
            day = meas_id['startdate'].day
            month = meas_id['startdate'].month
            year = meas_id['startdate'].year
            fid.read(8)  # skip file's meas_date
        else:
            meas_date = fid.read(8).decode('latin-1')
            day, month, year = [int(x) for x in meas_date.split('.')]
            year = year + 2000 if year < 85 else year + 1900

        meas_time = fid.read(8).decode('latin-1')
        hour, minute, sec = [int(x) for x in meas_time.split('.')]
        try:
            meas_date = datetime(year, month, day, hour, minute, sec,
                                 tzinfo=timezone.utc)
        except ValueError:
            warn(f'Invalid date encountered ({year:04d}-{month:02d}-'
                 f'{day:02d} {hour:02d}:{minute:02d}:{sec:02d}).')
            meas_date = None

        header_nbytes = int(_edf_str(fid.read(8)))

        # The following 44 bytes sometimes identify the file type, but this is
        # not guaranteed. Therefore, we skip this field and use the file
        # extension to determine the subtype (EDF or BDF, which differ in the
        # number of bytes they use for the data records; EDF uses 2 bytes
        # whereas BDF uses 3 bytes).
        fid.read(44)
        subtype = os.path.splitext(fname)[1][1:].lower()

        n_records = int(_edf_str(fid.read(8)))
        record_length = float(_edf_str(fid.read(8)))
        record_length = np.array([record_length, 1.])  # in seconds
        if record_length[0] == 0:
            record_length = record_length[0] = 1.
            warn('Header information is incorrect for record length. Default '
                 'record length set to 1.')

        nchan = int(_edf_str(fid.read(4)))
        channels = list(range(nchan))
        ch_names = [fid.read(16).strip().decode('latin-1') for ch in channels]
        exclude = _find_exclude_idx(ch_names, exclude)
        tal_idx = _find_tal_idx(ch_names)
        exclude = np.concatenate([exclude, tal_idx])
        sel = np.setdiff1d(np.arange(len(ch_names)), exclude)
        for ch in channels:
            fid.read(80)  # transducer
        units = [fid.read(8).strip().decode('latin-1') for ch in channels]
        edf_info['units'] = list()
        for i, unit in enumerate(units):
            if i in exclude:
                continue
            if unit == 'uV':
                edf_info['units'].append(1e-6)
            elif unit == 'mV':
                edf_info['units'].append(1e-3)
            else:
                edf_info['units'].append(1)
        edf_info['units'] = np.array(edf_info['units'], float)

        ch_names = [ch_names[idx] for idx in sel]
        units = [units[idx] for idx in sel]

        # make sure channel names are unique
        ch_names = _unique_channel_names(ch_names)
        orig_units = dict(zip(ch_names, units))

        physical_min = np.array(
            [float(_edf_str(fid.read(8))) for ch in channels])[sel]
        physical_max = np.array(
            [float(_edf_str(fid.read(8))) for ch in channels])[sel]
        digital_min = np.array(
            [float(_edf_str(fid.read(8))) for ch in channels])[sel]
        digital_max = np.array(
            [float(_edf_str(fid.read(8))) for ch in channels])[sel]
        prefiltering = [_edf_str(fid.read(80)).strip() for ch in channels][:-1]
        highpass, lowpass = _parse_prefilter_string(prefiltering)

        # number of samples per record
        n_samps = np.array([int(_edf_str(fid.read(8))) for ch in channels])

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
            edf_info['n_records'] = read_records
        del n_records

        if subtype == 'bdf':
            edf_info['dtype_byte'] = 3  # 24-bit (3 byte) integers
            edf_info['dtype_np'] = UINT8
        else:
            edf_info['dtype_byte'] = 2  # 16-bit (2 byte) integers
            edf_info['dtype_np'] = INT16

    return edf_info, orig_units


INT8 = '<i1'
UINT8 = '<u1'
INT16 = '<i2'
UINT16 = '<u2'
INT32 = '<i4'
UINT32 = '<u4'
INT64 = '<i8'
UINT64 = '<u8'
FLOAT32 = '<f4'
FLOAT64 = '<f8'
GDFTYPE_NP = (None, INT8, UINT8, INT16, UINT16, INT32, UINT32,
              INT64, UINT64, None, None, None, None,
              None, None, None, FLOAT32, FLOAT64)
GDFTYPE_BYTE = tuple(np.dtype(x).itemsize if x is not None else 0
                     for x in GDFTYPE_NP)


def _check_dtype_byte(types):
    assert sum(GDFTYPE_BYTE) == 42
    dtype_byte = [GDFTYPE_BYTE[t] for t in types]
    dtype_np = [GDFTYPE_NP[t] for t in types]
    if len(np.unique(dtype_byte)) > 1:
        # We will not read it properly, so this should be an error
        raise RuntimeError("Reading multiple data types not supported")
    return dtype_np[0], dtype_byte[0]


def _read_gdf_header(fname, exclude):
    """Read GDF 1.x and GDF 2.x header info."""
    edf_info = dict()
    events = None
    with open(fname, 'rb') as fid:

        version = fid.read(8).decode()
        edf_info['type'] = edf_info['subtype'] = version[:3]
        edf_info['number'] = float(version[4:])
        meas_date = None

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
            meas_id['recording_id'] = _edf_str(fid.read(80)).strip()

            # date
            tm = _edf_str(fid.read(16)).strip()
            try:
                if tm[14:16] == '  ':
                    tm = tm[:14] + '00' + tm[16:]
                meas_date = datetime(
                    int(tm[0:4]), int(tm[4:6]),
                    int(tm[6:8]), int(tm[8:10]),
                    int(tm[10:12]), int(tm[12:14]),
                    int(tm[14:16]) * pow(10, 4),
                    tzinfo=timezone.utc)
            except Exception:
                pass

            header_nbytes = np.fromfile(fid, INT64, 1)[0]
            meas_id['equipment'] = np.fromfile(fid, UINT8, 8)[0]
            meas_id['hospital'] = np.fromfile(fid, UINT8, 8)[0]
            meas_id['technician'] = np.fromfile(fid, UINT8, 8)[0]
            fid.seek(20, 1)    # 20bytes reserved

            n_records = np.fromfile(fid, INT64, 1)[0]
            # record length in seconds
            record_length = np.fromfile(fid, UINT32, 2)
            if record_length[0] == 0:
                record_length[0] = 1.
                warn('Header information is incorrect for record length. '
                     'Default record length set to 1.')
            nchan = np.fromfile(fid, UINT32, 1)[0]
            channels = list(range(nchan))
            ch_names = [_edf_str(fid.read(16)).strip() for ch in channels]
            exclude = _find_exclude_idx(ch_names, exclude)
            sel = np.setdiff1d(np.arange(len(ch_names)), exclude)
            fid.seek(80 * len(channels), 1)  # transducer
            units = [_edf_str(fid.read(8)).strip() for ch in channels]
            edf_info['units'] = list()
            for i, unit in enumerate(units):
                if i in exclude:
                    continue
                if unit[:2] == 'uV':
                    edf_info['units'].append(1e-6)
                else:
                    edf_info['units'].append(1)
            edf_info['units'] = np.array(edf_info['units'], float)

            ch_names = [ch_names[idx] for idx in sel]
            physical_min = np.fromfile(fid, FLOAT64, len(channels))
            physical_max = np.fromfile(fid, FLOAT64, len(channels))
            digital_min = np.fromfile(fid, INT64, len(channels))
            digital_max = np.fromfile(fid, INT64, len(channels))
            prefiltering = [_edf_str(fid.read(80)) for ch in channels][:-1]
            highpass, lowpass = _parse_prefilter_string(prefiltering)

            # n samples per record
            n_samps = np.fromfile(fid, INT32, len(channels))

            # channel data type
            dtype = np.fromfile(fid, INT32, len(channels))

            # total number of bytes for data
            bytes_tot = np.sum([GDFTYPE_BYTE[t] * n_samps[i]
                                for i, t in enumerate(dtype)])

            # Populate edf_info
            dtype_np, dtype_byte = _check_dtype_byte(dtype)
            edf_info.update(
                bytes_tot=bytes_tot, ch_names=ch_names,
                data_offset=header_nbytes, digital_min=digital_min,
                digital_max=digital_max,
                dtype_byte=dtype_byte, dtype_np=dtype_np, exclude=exclude,
                highpass=highpass, sel=sel, lowpass=lowpass,
                meas_date=meas_date,
                meas_id=meas_id, n_records=n_records, n_samps=n_samps,
                nchan=nchan, subject_info=patient, physical_max=physical_max,
                physical_min=physical_min, record_length=record_length)

            fid.seek(32 * edf_info['nchan'], 1)  # reserved
            assert fid.tell() == header_nbytes

            # Event table
            # -----------------------------------------------------------------
            etp = header_nbytes + n_records * edf_info['bytes_tot']
            # skip data to go to event table
            fid.seek(etp)
            etmode = np.fromfile(fid, UINT8, 1)[0]
            if etmode in (1, 3):
                sr = np.fromfile(fid, UINT8, 3)
                event_sr = sr[0]
                for i in range(1, len(sr)):
                    event_sr = event_sr + sr[i] * 2 ** (i * 8)
                n_events = np.fromfile(fid, UINT32, 1)[0]
                pos = np.fromfile(fid, UINT32, n_events) - 1  # 1-based inds
                typ = np.fromfile(fid, UINT16, n_events)

                if etmode == 3:
                    chn = np.fromfile(fid, UINT16, n_events)
                    dur = np.fromfile(fid, UINT32, n_events)
                else:
                    chn = np.zeros(n_events, dtype=np.int32)
                    dur = np.ones(n_events, dtype=UINT32)
                np.maximum(dur, 1, out=dur)
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
            sadm = np.fromfile(fid, UINT8, 1)[0]
            patient['smoking'] = scale[sadm % 4]
            patient['alcohol_abuse'] = scale[(sadm >> 2) % 4]
            patient['drug_abuse'] = scale[(sadm >> 4) % 4]
            patient['medication'] = scale[(sadm >> 6) % 4]
            patient['weight'] = np.fromfile(fid, UINT8, 1)[0]
            if patient['weight'] == 0 or patient['weight'] == 255:
                patient['weight'] = None
            patient['height'] = np.fromfile(fid, UINT8, 1)[0]
            if patient['height'] == 0 or patient['height'] == 255:
                patient['height'] = None

            # Gender / Handedness / Visual Impairment
            ghi = np.fromfile(fid, UINT8, 1)[0]
            patient['sex'] = gender[ghi % 4]
            patient['handedness'] = handedness[(ghi >> 2) % 4]
            patient['visual'] = scale[(ghi >> 4) % 4]

            # Recording identification
            meas_id = {}
            meas_id['recording_id'] = _edf_str(fid.read(64)).strip()
            vhsv = np.fromfile(fid, UINT8, 4)
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
                float(np.fromfile(fid, UINT32, 1)[0]) / 3600000
            loc['longitude'] = \
                float(np.fromfile(fid, UINT32, 1)[0]) / 3600000
            loc['altitude'] = float(np.fromfile(fid, INT32, 1)[0]) / 100
            meas_id['loc'] = loc

            meas_date = np.fromfile(fid, UINT64, 1)[0]
            if meas_date != 0:
                meas_date = (datetime(1, 1, 1, tzinfo=timezone.utc) +
                             timedelta(meas_date * pow(2, -32) - 367))
            else:
                meas_date = None

            birthday = np.fromfile(fid, UINT64, 1).tolist()[0]
            if birthday == 0:
                birthday = datetime(1, 1, 1, tzinfo=timezone.utc)
            else:
                birthday = (datetime(1, 1, 1, tzinfo=timezone.utc) +
                            timedelta(birthday * pow(2, -32) - 367))
            patient['birthday'] = birthday
            if patient['birthday'] != datetime(1, 1, 1, 0, 0,
                                               tzinfo=timezone.utc):
                today = datetime.now(tz=timezone.utc)
                patient['age'] = today.year - patient['birthday'].year
                today = today.replace(year=patient['birthday'].year)
                if today < patient['birthday']:
                    patient['age'] -= 1
            else:
                patient['age'] = None

            header_nbytes = np.fromfile(fid, UINT16, 1)[0] * 256

            fid.seek(6, 1)  # 6 bytes reserved
            meas_id['equipment'] = np.fromfile(fid, UINT8, 8)
            meas_id['ip'] = np.fromfile(fid, UINT8, 6)
            patient['headsize'] = np.fromfile(fid, UINT16, 3)
            patient['headsize'] = np.asarray(patient['headsize'], np.float32)
            patient['headsize'] = np.ma.masked_array(
                patient['headsize'],
                np.equal(patient['headsize'], 0), None).filled()
            ref = np.fromfile(fid, FLOAT32, 3)
            gnd = np.fromfile(fid, FLOAT32, 3)
            n_records = np.fromfile(fid, INT64, 1)[0]

            # record length in seconds
            record_length = np.fromfile(fid, UINT32, 2)
            if record_length[0] == 0:
                record_length[0] = 1.
                warn('Header information is incorrect for record length. '
                     'Default record length set to 1.')

            nchan = np.fromfile(fid, UINT16, 1)[0]
            fid.seek(2, 1)  # 2bytes reserved

            # Channels (variable header)
            channels = list(range(nchan))
            ch_names = [_edf_str(fid.read(16)).strip() for ch in channels]
            exclude = _find_exclude_idx(ch_names, exclude)
            sel = np.setdiff1d(np.arange(len(ch_names)), exclude)

            fid.seek(80 * len(channels), 1)  # reserved space
            fid.seek(6 * len(channels), 1)  # phys_dim, obsolete

            """The Physical Dimensions are encoded as int16, according to:
            - Units codes :
            https://sourceforge.net/p/biosig/svn/HEAD/tree/trunk/biosig/doc/units.csv
            - Decimal factors codes:
            https://sourceforge.net/p/biosig/svn/HEAD/tree/trunk/biosig/doc/DecimalFactors.txt
            """  # noqa
            units = np.fromfile(fid, UINT16, len(channels)).tolist()
            unitcodes = np.array(units[:])
            edf_info['units'] = list()
            for i, unit in enumerate(units):
                if i in exclude:
                    continue
                if unit == 4275:  # microvolts
                    edf_info['units'].append(1e-6)
                elif unit == 4274:  # millivolts
                    edf_info['units'].append(1e-3)
                elif unit == 512:  # dimensionless
                    edf_info['units'].append(1)
                elif unit == 0:
                    edf_info['units'].append(1)  # unrecognized
                else:
                    warn('Unsupported physical dimension for channel %d '
                         '(assuming dimensionless). Please contact the '
                         'MNE-Python developers for support.' % i)
                    edf_info['units'].append(1)
            edf_info['units'] = np.array(edf_info['units'], float)

            ch_names = [ch_names[idx] for idx in sel]
            physical_min = np.fromfile(fid, FLOAT64, len(channels))
            physical_max = np.fromfile(fid, FLOAT64, len(channels))
            digital_min = np.fromfile(fid, FLOAT64, len(channels))
            digital_max = np.fromfile(fid, FLOAT64, len(channels))

            fid.seek(68 * len(channels), 1)  # obsolete
            lowpass = np.fromfile(fid, FLOAT32, len(channels))
            highpass = np.fromfile(fid, FLOAT32, len(channels))
            notch = np.fromfile(fid, FLOAT32, len(channels))

            # number of samples per record
            n_samps = np.fromfile(fid, INT32, len(channels))

            # data type
            dtype = np.fromfile(fid, INT32, len(channels))

            channel = {}
            channel['xyz'] = [np.fromfile(fid, FLOAT32, 3)[0]
                              for ch in channels]

            if edf_info['number'] < 2.19:
                impedance = np.fromfile(fid, UINT8,
                                        len(channels)).astype(float)
                impedance[impedance == 255] = np.nan
                channel['impedance'] = pow(2, impedance / 8)
                fid.seek(19 * len(channels), 1)  # reserved
            else:
                tmp = np.fromfile(fid, FLOAT32, 5 * len(channels))
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
            bytes_tot = np.sum([GDFTYPE_BYTE[t] * n_samps[i]
                                for i, t in enumerate(dtype)])

            # Populate edf_info
            dtype_np, dtype_byte = _check_dtype_byte(dtype)
            edf_info.update(
                bytes_tot=bytes_tot, ch_names=ch_names,
                data_offset=header_nbytes,
                dtype_byte=dtype_byte, dtype_np=dtype_np,
                digital_min=digital_min, digital_max=digital_max,
                exclude=exclude, gnd=gnd, highpass=highpass, sel=sel,
                impedance=impedance, lowpass=lowpass, meas_date=meas_date,
                meas_id=meas_id, n_records=n_records, n_samps=n_samps,
                nchan=nchan, notch=notch, subject_info=patient,
                physical_max=physical_max, physical_min=physical_min,
                record_length=record_length, ref=ref)

            # EVENT TABLE
            # -----------------------------------------------------------------
            etp = edf_info['data_offset'] + edf_info['n_records'] * \
                edf_info['bytes_tot']
            fid.seek(etp)  # skip data to go to event table
            etmode = fid.read(1).decode()
            if etmode != '':
                etmode = np.fromstring(etmode, UINT8).tolist()[0]

                if edf_info['number'] < 1.94:
                    sr = np.fromfile(fid, UINT8, 3)
                    event_sr = sr[0]
                    for i in range(1, len(sr)):
                        event_sr = event_sr + sr[i] * 2**(i * 8)
                    n_events = np.fromfile(fid, UINT32, 1)[0]
                else:
                    ne = np.fromfile(fid, UINT8, 3)
                    n_events = ne[0]
                    for i in range(1, len(ne)):
                        n_events = n_events + ne[i] * 2**(i * 8)
                    event_sr = np.fromfile(fid, FLOAT32, 1)[0]

                pos = np.fromfile(fid, UINT32, n_events) - 1  # 1-based inds
                typ = np.fromfile(fid, UINT16, n_events)

                if etmode == 3:
                    chn = np.fromfile(fid, UINT16, n_events)
                    dur = np.fromfile(fid, UINT32, n_events)
                else:
                    chn = np.zeros(n_events, dtype=np.uint32)
                    dur = np.ones(n_events, dtype=np.uint32)
                np.clip(dur, 1, np.inf, out=dur)
                events = [n_events, pos, typ, chn, dur]
                edf_info['event_sfreq'] = event_sr

    edf_info.update(events=events, sel=np.arange(len(edf_info['ch_names'])))

    return edf_info


def _check_stim_channel(stim_channel, ch_names,
                        tal_ch_names=['EDF Annotations', 'BDF Annotations']):
    """Check that the stimulus channel exists in the current datafile."""
    DEFAULT_STIM_CH_NAMES = ['status', 'trigger']

    if stim_channel is None or stim_channel is False:
        return [], []

    if stim_channel is True:  # convenient aliases
        stim_channel = 'auto'

    elif isinstance(stim_channel, str):
        if stim_channel == 'auto':
            if 'auto' in ch_names:
                warn(RuntimeWarning, "Using `stim_channel='auto'` when auto"
                     " also corresponds to a channel name is ambiguous."
                     " Please use `stim_channel=['auto']`.")
            else:
                valid_stim_ch_names = DEFAULT_STIM_CH_NAMES
        else:
            valid_stim_ch_names = [stim_channel.lower()]

    elif isinstance(stim_channel, int):
        valid_stim_ch_names = [ch_names[stim_channel].lower()]

    elif isinstance(stim_channel, list):
        if all([isinstance(s, str) for s in stim_channel]):
            valid_stim_ch_names = [s.lower() for s in stim_channel]
        elif all([isinstance(s, int) for s in stim_channel]):
            valid_stim_ch_names = [ch_names[s].lower() for s in stim_channel]
        else:
            raise ValueError('Invalid stim_channel')
    else:
        raise ValueError('Invalid stim_channel')

    # Forbid the synthesis of stim channels from TAL Annotations
    tal_ch_names_found = [ch for ch in valid_stim_ch_names
                          if ch in [t.lower() for t in tal_ch_names]]
    if len(tal_ch_names_found):
        _msg = ('The synthesis of the stim channel is not supported'
                ' since 0.18. Please remove {} from `stim_channel`'
                ' and use `mne.events_from_annotations` instead'
                ).format(tal_ch_names_found)
        raise ValueError(_msg)

    ch_names_low = [ch.lower() for ch in ch_names]
    found = list(set(valid_stim_ch_names) & set(ch_names_low))

    if not found:
        return [], []
    else:
        stim_channel_idxs = [ch_names_low.index(f) for f in found]
        names = [ch_names[idx] for idx in stim_channel_idxs]
        return stim_channel_idxs, names


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


@fill_doc
def read_raw_edf(input_fname, eog=None, misc=None,
                 stim_channel='auto', exclude=(), preload=False, verbose=None):
    """Reader function for EDF or EDF+ files.

    Parameters
    ----------
    input_fname : str
        Path to the EDF or EDF+ file.
    eog : list or tuple
        Names of channels or list of indices that should be designated EOG
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    misc : list or tuple
        Names of channels or list of indices that should be designated MISC
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    stim_channel : 'auto' | str | list of str | int | list of int
        Defaults to 'auto', which means that channels named 'status' or
        'trigger' (case insensitive) are set to STIM. If str (or list of str),
        all channels matching the name(s) are set to STIM. If int (or list of
        ints), channels corresponding to the indices are set to STIM.

        .. warning:: 0.18 does not allow for stim channel synthesis from TAL
                     channels called 'EDF Annotations' anymore. Instead, TAL
                     channels are parsed and extracted annotations are stored
                     in raw.annotations. Use
                     :func:`mne.events_from_annotations` to obtain events from
                     these annotations.

    exclude : list of str
        Channel names to exclude. This can help when reading data with
        different sampling rates to avoid unnecessary resampling.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawEDF
        The raw instance.

    See Also
    --------
    mne.io.read_raw_bdf : Reader function for BDF files.
    mne.io.read_raw_gdf : Reader function for GDF files.

    Notes
    -----
    It is worth noting that in some special cases, it may be necessary to shift
    event values in order to retrieve correct event triggers. This depends on
    the triggering device used to perform the synchronization. For instance, in
    some files events need to be shifted by 8 bits:

        >>> events[:, 2] >>= 8  # doctest:+SKIP

    TAL channels called 'EDF Annotations' are parsed and extracted annotations
    are stored in raw.annotations. Use :func:`mne.events_from_annotations` to
    obtain events from these annotations.

    If channels named 'status' or 'trigger' are present, they are considered as
    STIM channels by default. Use func:`mne.find_events` to parse events
    encoded in such analog stim channels.
    """
    input_fname = os.path.abspath(input_fname)
    ext = os.path.splitext(input_fname)[1][1:].lower()
    if ext != 'edf':
        raise NotImplementedError(
            'Only EDF files are supported by read_raw_edf, got %s' % (ext,))
    return RawEDF(input_fname=input_fname, eog=eog, misc=misc,
                  stim_channel=stim_channel, exclude=exclude, preload=preload,
                  verbose=verbose)


@fill_doc
def read_raw_bdf(input_fname, eog=None, misc=None,
                 stim_channel='auto', exclude=(), preload=False, verbose=None):
    """Reader function for BDF files.

    Parameters
    ----------
    input_fname : str
        Path to the BDF file.
    eog : list or tuple
        Names of channels or list of indices that should be designated EOG
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    misc : list or tuple
        Names of channels or list of indices that should be designated MISC
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    stim_channel : 'auto' | str | list of str | int | list of int
        Defaults to 'auto', which means that channels named 'status' or
        'trigger' (case insensitive) are set to STIM. If str (or list of str),
        all channels matching the name(s) are set to STIM. If int (or list of
        ints), channels corresponding to the indices are set to STIM.

        .. warning:: 0.18 does not allow for stim channel synthesis from TAL
                     channels called 'BDF Annotations' anymore. Instead, TAL
                     channels are parsed and extracted annotations are stored
                     in raw.annotations. Use
                     :func:`mne.events_from_annotations` to obtain events from
                     these annotations.

    exclude : list of str
        Channel names to exclude. This can help when reading data with
        different sampling rates to avoid unnecessary resampling.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawEDF
        The raw instance.

    See Also
    --------
    mne.io.read_raw_edf : Reader function for EDF and EDF+ files.
    mne.io.read_raw_gdf : Reader function for GDF files.

    Notes
    -----
    Biosemi devices trigger codes are encoded in 16-bit format, whereas system
    codes (CMS in/out-of range, battery low, etc.) are coded in bits 16-23 of
    the status channel (see http://www.biosemi.com/faq/trigger_signals.htm).
    To retrieve correct event values (bits 1-16), one could do:

        >>> events = mne.find_events(...)  # doctest:+SKIP
        >>> events[:, 2] &= (2**16 - 1)  # doctest:+SKIP

    The above operation can be carried out directly in :func:`mne.find_events`
    using the ``mask`` and ``mask_type`` parameters (see
    :func:`mne.find_events` for more details).

    It is also possible to retrieve system codes, but no particular effort has
    been made to decode these in MNE. In case it is necessary, for instance to
    check the CMS bit, the following operation can be carried out:

        >>> cms_bit = 20  # doctest:+SKIP
        >>> cms_high = (events[:, 2] & (1 << cms_bit)) != 0  # doctest:+SKIP

    It is worth noting that in some special cases, it may be necessary to shift
    event values in order to retrieve correct event triggers. This depends on
    the triggering device used to perform the synchronization. For instance, in
    some files events need to be shifted by 8 bits:

        >>> events[:, 2] >>= 8  # doctest:+SKIP

    TAL channels called 'BDF Annotations' are parsed and extracted annotations
    are stored in raw.annotations. Use :func:`mne.events_from_annotations` to
    obtain events from these annotations.

    If channels named 'status' or 'trigger' are present, they are considered as
    STIM channels by default. Use func:`mne.find_events` to parse events
    encoded in such analog stim channels.
    """
    input_fname = os.path.abspath(input_fname)
    ext = os.path.splitext(input_fname)[1][1:].lower()
    if ext != 'bdf':
        raise NotImplementedError('Only BDF files are supported, got '
                                  '{}.'.format(ext))
    return RawEDF(input_fname=input_fname, eog=eog, misc=misc,
                  stim_channel=stim_channel, exclude=exclude, preload=preload,
                  verbose=verbose)


@fill_doc
def read_raw_gdf(input_fname, eog=None, misc=None,
                 stim_channel='auto', exclude=(), preload=False, verbose=None):
    """Reader function for GDF files.

    Parameters
    ----------
    input_fname : str
        Path to the GDF file.
    eog : list or tuple
        Names of channels or list of indices that should be designated EOG
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    misc : list or tuple
        Names of channels or list of indices that should be designated MISC
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    stim_channel : 'auto' | str | list of str | int | list of int
        Defaults to 'auto', which means that channels named 'status' or
        'trigger' (case insensitive) are set to STIM. If str (or list of str),
        all channels matching the name(s) are set to STIM. If int (or list of
        ints), channels corresponding to the indices are set to STIM.
    exclude : list of str
        Channel names to exclude. This can help when reading data with
        different sampling rates to avoid unnecessary resampling.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawGDF
        The raw instance.

    See Also
    --------
    mne.io.read_raw_edf : Reader function for EDF and EDF+ files.
    mne.io.read_raw_bdf : Reader function for BDF files.

    Notes
    -----
    If channels named 'status' or 'trigger' are present, they are considered as
    STIM channels by default. Use func:`mne.find_events` to parse events
    encoded in such analog stim channels.
    """
    input_fname = os.path.abspath(input_fname)
    ext = os.path.splitext(input_fname)[1][1:].lower()
    if ext != 'gdf':
        raise NotImplementedError('Only GDF files are supported, got '
                                  '{}.'.format(ext))
    return RawGDF(input_fname=input_fname, eog=eog, misc=misc,
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
        annotations = np.atleast_2d(annotations)
        for chan in annotations:
            this_chan = chan.ravel()
            if this_chan.dtype == INT32:  # BDF
                this_chan = this_chan.view(dtype=UINT8)
                this_chan = this_chan.reshape(-1, 4)
                # Why only keep the first 3 bytes as BDF values
                # are stored with 24 bits (not 32)
                this_chan = this_chan[:, :3].ravel()
                # As ravel() returns a 1D array we can add all values at once
                tals.extend(this_chan)
            else:
                this_chan = chan.astype(np.int64)
                # Exploit np vectorized processing
                tals.extend(np.uint8([this_chan % 256, this_chan // 256])
                            .flatten('F'))

        # use of latin-1 because characters are only encoded for the first 256
        # code points and utf-8 can triggers an "invalid continuation byte"
        # error
        triggers = re.findall(pat, tals.decode('latin-1'))

    events = []
    offset = 0.
    for k, ev in enumerate(triggers):
        onset = float(ev[0]) + offset
        duration = float(ev[2]) if ev[2] else 0
        for description in ev[3].split('\x14')[1:]:
            if description:
                events.append([onset, duration, description])
            elif k == 0:
                # The startdate/time of a file is specified in the EDF+ header
                # fields 'startdate of recording' and 'starttime of recording'.
                # These fields must indicate the absolute second in which the
                # start of the first data record falls. So, the first TAL in
                # the first data record always starts with +0.X, indicating
                # that the first data record starts a fraction, X, of a second
                # after the startdate/time that is specified in the EDF+
                # header. If X=0, then the .X may be omitted.
                offset = -onset

    return zip(*events) if events else (list(), list(), list())


def _get_edf_default_event_id(descriptions):
    mapping = {a: n for n, a in enumerate(sorted(set(descriptions)), start=1)}
    return mapping


def _get_annotations_gdf(edf_info, sfreq):
    onset, duration, desc = list(), list(), list()
    events = edf_info.get('events', None)
    # Annotations in GDF: events are stored as the following
    # list: `events = [n_events, pos, typ, chn, dur]` where pos is the
    # latency, dur is the duration in samples. They both are
    # numpy.ndarray
    if events is not None and events[1].shape[0] > 0:
        onset = events[1] / sfreq
        duration = events[4] / sfreq
        desc = events[2]

    return onset, duration, desc
