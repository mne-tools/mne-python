# Author: Porloy Das <pdas6@mgh.harvard.edu>
import os
from datetime import datetime, timezone
from warnings import warn

import numpy as np

from .constants import FIFF
from .meas_info import _empty_info
from .base import BaseRaw
from .utils import _read_segments_file #, _mult_cal_one
from ..utils import logger


# common channel type names mapped to internal ch types
CH_TYPE_MAPPING = {
    'EEG': FIFF.FIFFV_EEG_CH,
    'SEEG': FIFF.FIFFV_SEEG_CH,
    'CC': FIFF.FIFFV_SEEG_CH,
    'ECOG': FIFF.FIFFV_ECOG_CH,
    'DBS': FIFF.FIFFV_DBS_CH,
    'EOG': FIFF.FIFFV_EOG_CH,
    'ECG': FIFF.FIFFV_ECG_CH,
    'EMG': FIFF.FIFFV_EMG_CH,
    'BIO': FIFF.FIFFV_BIO_CH,
    'RESP': FIFF.FIFFV_RESP_CH,
    'TEMP': FIFF.FIFFV_TEMPERATURE_CH,
    'MISC': FIFF.FIFFV_MISC_CH,
    'SAO2': FIFF.FIFFV_BIO_CH,
}


_unit_dict = {'V': 1.,  # V stands for Volt
              'µV': 1e-6,
              'uV': 1e-6,
              'mV': 1e-3
}


nsx_header_dict = {
    'basic_2.1': [
            ('file_id', 'S8'),
            # label of sampling group (e.g. "1kS/s" or "LFP Low")
            ('label', 'S16'),
            # number of 1/30000 seconds between data points
            # (e.g., if sampling rate "1 kS/s", period equals "30")
            ('period', 'uint32'),
            ('channel_count', 'uint32')],
    'basic':[
            ('file_id', 'S8'),  # achFileType
            # file specification split into major and minor version number
            ('ver_major', 'uint8'),
            ('ver_minor', 'uint8'),
            # bytes of basic & extended header
            ('bytes_in_headers', 'uint32'),
            # label of the sampling group (e.g., "1 kS/s" or "LFP low")
            ('label', 'S16'),
            ('comment', 'S256'),
            ('period', 'uint32'),
            ('timestamp_resolution', 'uint32'),
            # time origin: 2byte uint16 values for ...
            ('year', 'uint16'),
            ('month', 'uint16'),
            ('weekday', 'uint16'),
            ('day', 'uint16'),
            ('hour', 'uint16'),
            ('minute', 'uint16'),
            ('second', 'uint16'),
            ('millisecond', 'uint16'),
            # number of channel_count match number of extended headers
            ('channel_count', 'uint32')],
    'extended': [
            ('type', 'S2'),
            ('electrode_id', 'uint16'),
            ('electrode_label', 'S16'),
            # used front-end amplifier bank (e.g., A, B, C, D)
            ('physical_connector', 'uint8'),
            # used connector pin (e.g., 1-37 on bank A, B, C or D)
            ('connector_pin', 'uint8'),
            # digital and analog value ranges of the signal
            ('min_digital_val', 'int16'),
            ('max_digital_val', 'int16'),
            ('min_analog_val', 'int16'),
            ('max_analog_val', 'int16'),
            # units of the analog range values ("mV" or "uV")
            ('units', 'S16'),
            # filter settings used to create nsx from source signal
            ('hi_freq_corner', 'uint32'),
            ('hi_freq_order', 'uint32'),
            ('hi_freq_type', 'uint16'),  # 0=None, 1=Butterworth
            ('lo_freq_corner', 'uint32'),
            ('lo_freq_order', 'uint32'),
            ('lo_freq_type', 'uint16')],  # 0=None, 1=Butterworth,
    'data>2.1<3': [
            ('header', 'uint8'),
            ('timestamp', 'uint32'),    
            ('nb_data_points', 'uint32')],
    'data>=3': [
            ('header', 'uint8'),
            ('timestamp', 'uint64'),     
            ('nb_data_points', 'uint32')],
}


class RawNSX(BaseRaw):
    def __init__(self, nsx_fname, stim_channel=True,
                 eog=[], misc=[], preload=False, verbose=None):
        
        (info, data_fname, fmt, n_samples, orig_format, data_headers, orig_units) = \
                _get_hdr_info(nsx_fname, stim_channel=stim_channel, 
                        eog=eog, misc=misc, exclude=None)
        raw_extras = dict(data_headers=data_headers, fmt=fmt, orig_format=orig_format)
        super(RawNSX, self).__init__(
            info, last_samps=[n_samples - 1], filenames=[data_fname],
            orig_format=orig_format, preload=preload, verbose=verbose,
            raw_extras=[raw_extras], orig_units=orig_units)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        n_data_ch = self._raw_extras[fi]['orig_nchan']
        dtype = self._raw_extras[fi]['orig_format']
        beginnings = self._raw_extras[fi]['data_headers']['beginnings']
        endings = self._raw_extras[fi]['data_headers']['endings']
        i_start = np.searchsorted(beginnings, start) - 1
        if i_start == -1 : i_start = 0  # Corner case
        i_stop = np.searchsorted(endings, stop)
        if i_start > i_stop:
            raise ValueError
        elif i_start == i_stop:
            offset = self._raw_extras[fi]['data_headers']['offsets'][i_start]
            _read_segments_file(self, data, idx, fi, start, stop, cals, mult,
                        dtype, n_channels=n_data_ch, offset=offset, trigger_ch=None)
        else:  # i_start < i_stop
            raise NotImplementedError(f'')
            cont_segments = [(start, endings[0]), ]
            for ii in range(i_start+1, i_stop):
                cont_segments.append((beginnings[ii], endings[ii]))
            cont_segments.append((beginnings[i_stop], stop))
            offsets = self._raw_extras[fi]['data_headers']['offsets'][i_start:i_stop+1]
            edge_samps = []
            for (_start, _stop), offset in zip(cont_segments, offsets):
                _read_segments_file(self, data, idx, fi, _start, _stop, cals, mult,
                        dtype, n_channels=n_data_ch, offset=offset, trigger_ch=None)
            # add bad annotations at the discontinuous sample.
                edge_samps.append(_stop - start)
            
            annotations = self.annotations
            assert annotations.orig_time == self.info['meas_date']
            edge_samps = np.cumsum(edge_samps)
            for edge_samp in edge_samps:
                onset = _sync_onset(self, (edge_samp) / self.info['sfreq'], True)
                self.annotations.append(onset, 0., 'BAD boundary')
                self.annotations.append(onset, 0., 'EDGE boundary')


def _sync_onset(raw, onset, inverse=False):
    """Adjust onsets in relation to raw data."""
    # Copied from https://github.com/mne-tools/mne-python/blob/78e099d9c117c01a0545697a6bbb26521659556e/mne/annotations.py#L935 
    offset = (-1 if inverse else 1) * raw._first_time
    assert raw.info['meas_date'] == raw.annotations.orig_time
    annot_start = onset - offset
    return annot_start


def _get_file_size(filename):
    """
    Returns the file size in bytes for the given file.
    """
    with open(filename, 'rb') as fid:
        fid.seek(0, os.SEEK_END)
        file_size = fid.tell()
    return file_size

def _read_header(filename):
    basic_header = {}
    nsx_file_id = np.fromfile(filename, count=1,
                                dtype=[('file_id', 'S8')])[0]['file_id'].decode()
    if nsx_file_id == 'NEURALSG':
        dtype0 = nsx_header_dict['basic_2.1']
        dtype1 = [('electrode_id', 'uint32')]
        basic_header['timestamp_resolution'] = 30000
        basic_header['bytes_in_headers'] = (
            32 + 4 * basic_header["channel_count"]
        )
        basic_header['ver_major'] = 2
        basic_header['ver_minor'] = 1
    elif nsx_file_id in ['NEURALCD', 'BRSMPGRP']:
        dtype0 = nsx_header_dict['basic']
        dtype1 = nsx_header_dict['extended']

    nsx_file_header = np.fromfile(filename, count=1, dtype=dtype0)[0]
    basic_header.update({name: nsx_file_header[name] for name in nsx_file_header.dtype.names})
    
    offset_dtype0 = np.dtype(dtype0).itemsize
    shape = nsx_file_header['channel_count']
    basic_header['extended'] = np.memmap(filename, shape=shape, offset=offset_dtype0, dtype=dtype1, mode='r')

    if nsx_file_id == 'NEURALSG':
        basic_header['highpass'] = np.nan * np.ones(basic_header['channel_count'])
        basic_header['lowpass'] = np.nan * np.ones(basic_header['channel_count'])
    elif nsx_file_id in ['NEURALCD', 'BRSMPGRP']:
        # The values are stored in mV 
        # See: https://blackrockneurotech.com/research/wp-content/ifu/LB-0023-7.00_NEV_File_Format.pdf)
        basic_header['highpass'] = basic_header['extended']['hi_freq_corner'] * 1e-3
        basic_header['lowpass'] = basic_header['extended']['lo_freq_corner'] * 1e-3

    ver_major, ver_minor = basic_header.pop('ver_major'), basic_header.pop('ver_minor') 
    basic_header['spec'] = '{}.{}'.format(ver_major, ver_minor)
    
    data_header = {}
    index = 0
    offset = basic_header['bytes_in_headers']
    filesize = _get_file_size(filename)
    if float(basic_header['spec']) > 2.1:
        if float(basic_header['spec']) < 3.0:
            dtype2 = nsx_header_dict['data>2.1<3']
            data_bytes = 2
        else:
            dtype2 = nsx_header_dict['data>=3']
            data_bytes = 4
        while offset < filesize:
            dh = np.memmap(filename, dtype=dtype2, shape=1, offset=offset, mode='r')[0]
            data_header[index] = {
                    'header': dh['header'],
                    'timestamp': dh['timestamp'],
                    'nb_data_points': dh['nb_data_points'],
                    'offset_to_data_block': offset + dh.dtype.itemsize
            }
            # data size = number of data points * (data_bytes * number of channels)
            # use of `int` avoids overflow problem
            data_size = int(dh['nb_data_points']) *\
                int(basic_header['channel_count']) * data_bytes
            # define new offset (to possible next data block)
            offset = data_header[index]['offset_to_data_block'] + data_size

            index += 1
    else:
        data_header[index] = {
                'header': 1,
                'timestamp': 0,
                'nb_data_points': (filesize - offset) // (2 * basic_header['channel_count']),
                'offset_to_data_block': offset
        }
    basic_header['data_header'] = data_header
        
    try:
        time_origin = datetime(*[basic_header[xx] for xx in ('year', 'month', 'day',
                    'hour', 'minute', 'second','millisecond')], tzinfo=timezone.utc)
        [basic_header.pop(xx) for xx in ('year', 'month', 'day', 'hour', 'minute', 'second','millisecond')]
    except KeyError:
        time_origin = None
    basic_header['meas_date'] = time_origin

    return basic_header


def _get_hdr_info(filename, stim_channel=True, eog=[], misc=[], exclude=None):
    eog = eog if eog is not None else []
    misc = misc if misc is not None else []
    
    nsx_info = _read_header(filename)
    if nsx_info['spec'] == 2.1:
        ch_names = []
        for elid in list(nsx_info['extended']):
            if elid < 129:
                ch_names.append('chan%i' % elid)
            else:
                ch_names.append('ainp%i' % (elid - 129 + 1))
        ch_units = [''] * len(ch_names)
        ch_types = ['CC'] * len(ch_names)
        warn("Cannot rescale to voltage, raw data will be returned.", UserWarning)
        
        
    else:
        ch_names = list(nsx_info['extended']['electrode_label'])
        ch_types = list(nsx_info['extended']['type']) 
        ch_units = list(nsx_info['extended']['units']) 
        ch_names, ch_types, ch_units = \
            [list(map(bytes.decode, xx)) for xx in (ch_names, ch_types, ch_units)]

    stim_channel_idxs, _ = _check_stim_channel(stim_channel, ch_names)

    nchan = nsx_info['channel_count']
    logger.info('Setting channel info structure...')
    chs = list()
    pick_mask = np.ones(len(ch_names))

    chs_without_types = list()
    orig_units = {}

    for idx, ch_name in enumerate(ch_names):
        chan_info = {}
        chan_info['logno'] = nsx_info['extended']['electrode_id'][idx]
        chan_info['scanno'] = nsx_info['extended']['electrode_id'][idx]
        chan_info['ch_name'] = ch_name
        ch_unit = ch_units[idx]
        if ch_unit == '':
            chan_info['unit_mul'] = FIFF.FIFF_UNITM_NONE
            chan_info['unit'] = FIFF.FIFF_UNIT_NONE
        else:
            chan_info['unit_mul'] = FIFF.FIFF_UNITM_NONE
            chan_info['unit'] = FIFF.FIFF_UNIT_V
        chan_info['range'] = _unit_dict.get(ch_unit, 1)
        chan_info['cal'] = 1
        chan_info['coord_frame'] = FIFF.FIFFV_COORD_HEAD
        chan_info['coil_type'] = FIFF.FIFFV_COIL_EEG
        chan_info['kind'] = FIFF.FIFFV_EEG_CH
        # montage can't be stored in EDF so channel locs are unknown:
        chan_info['loc'] = np.full(12, np.nan)
        orig_units[ch_name] = ch_unit

        # if the edf info contained channel type information
        # set it now
        ch_type = ch_types[idx]
        if ch_type is not None and ch_type in CH_TYPE_MAPPING:
            chan_info['kind'] = CH_TYPE_MAPPING.get(ch_type)
            if ch_type not in ['EEG', 'ECOG', 'SEEG', 'DBS', 'CC']:
                chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
                pick_mask[idx] = False
        # if user passes in explicit mapping for eog, misc and stim
        # channels set them here
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
        elif ch_type not in CH_TYPE_MAPPING:
            chs_without_types.append(ch_name)
        chs.append(chan_info)

    # warn if channel type was not inferable
    if len(chs_without_types):
        msg = ('Could not determine channel type of the following channels, '
               f'they will be set as EEG:\n{", ".join(chs_without_types)}')
        logger.info(msg)

    sfreq = nsx_info['timestamp_resolution'] / nsx_info['period']
    info = _empty_info(sfreq)
    info['meas_date'] = nsx_info['meas_date']
    info['chs'] = chs
    info['ch_names'] = ch_names

    highpass = nsx_info['highpass'][:128]
    lowpass = nsx_info['lowpass'][:128]
    if np.all(highpass == highpass[0]):
        if highpass[0] == 'NaN':
            # Placeholder for future use. Highpass set in _empty_info.
            pass
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
    
    if np.all(lowpass == lowpass[0]):
        if lowpass[0] in ('NaN', '0', '0.0'):
            # Placeholder for future use. Lowpass set in _empty_info.
            pass
        else:
            info['lowpass'] = float(lowpass[0])
    else:
        info['lowpass'] = float(np.min(lowpass))
        warn('Channels contain different lowpass filters. Lowest filter '
            'setting will be stored.')
    if np.isnan(info['lowpass']):
        info['lowpass'] = info['sfreq'] / 2.

    if info['highpass'] > info['lowpass']:
        warn(f'Highpass cutoff frequency {info["highpass"]} is greater '
             f'than lowpass cutoff frequency {info["lowpass"]}, '
             'setting values to 0 and Nyquist.')
        info['highpass'] = 0.
        info['lowpass'] = info['sfreq'] / 2.
    
    # Some keys to be consistent with FIF measurement info
    info['description'] = None
    
    info._unlocked = False
    info._update_redundant()
    
    data_headers = {}
    orig_format = 'short'
    beginnings = []
    endings = []
    offsets = []
    n_samples = 0
    for data_bl, data_bl_info in nsx_info['data_header'].items():
        if data_bl_info['header'] != 1:
            continue
        start = data_bl_info['timestamp'] // nsx_info['period']
        end = start + data_bl_info['nb_data_points']
        n_samples += data_bl_info['nb_data_points']
        beginnings.append(start)
        endings.append(end)
        offsets.append(data_bl_info['offset_to_data_block'])
    data_headers['beginnings'] = np.array(beginnings)
    data_headers['endings'] = np.array(endings)
    data_headers['offsets'] = np.array(offsets)

    return info, filename, nsx_info['spec'], n_samples, orig_format, data_headers, orig_units


def _check_stim_channel(stim_channel, ch_names):
    """Check that the stimulus channel exists in the current datafile."""
    DEFAULT_STIM_CH_NAMES = ['status', 'trigger']

    if stim_channel is None or stim_channel is False:
        return [], []

    if stim_channel is True:  # convenient aliases
        stim_channel = 'auto'

    if isinstance(stim_channel, str):
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

    ch_names_low = [ch.lower() for ch in ch_names]
    found = list(set(valid_stim_ch_names) & set(ch_names_low))

    if not found:
        return [], []
    else:
        stim_channel_idxs = [ch_names_low.index(f) for f in found]
        names = [ch_names[idx] for idx in stim_channel_idxs]
        return stim_channel_idxs, names