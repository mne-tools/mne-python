"""Conversion tool from Neuroscan CNT to FIF
"""

# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)
from os import path
import datetime
import calendar

import numpy as np

from ...utils import warn, verbose
from ..constants import FIFF
from ..utils import _mult_cal_one
from ..meas_info import _empty_info
from ..base import _BaseRaw


def read_raw_cnt(input_fname, read_blocks=True, preload=False, verbose=None):
    """

    Parameters
    ----------
    input_fname : str
        Path to the data file.
    read_blocks : bool
        Whether to read data in blocks. This is for dealing with different
        kinds of CNT data formats.
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
    Instance of RawCNT.
    """
    return RawCNT(input_fname, read_blocks=read_blocks, preload=preload,
                  verbose=verbose)


def _get_cnt_info(input_fname, read_blocks):
    """Helper for reading the cnt header."""
    dtype = '<i4'
    n_bytes = np.dtype(dtype).itemsize
    offset = 900  # Size of the 'SETUP' header.
    start = 0
    cnt_info = dict()
    # Reading only the fields of interest. Structure of the whole header at
    # http://paulbourke.net/dataformats/eeg/
    with open(input_fname, 'rb', buffering=0) as fid:
        fid.seek(225)
        session_date = ''.join(np.fromfile(fid, dtype='S1', count=10))
        time = ''.join(np.fromfile(fid, dtype='S1', count=12))
        date = session_date.split('/')
        if date[2].startswith('9'):
            date[2] = '19' + date[2]
        elif len(date[2]) == 2:
            date[2] = '20' + date[2]
        time = time.split(':')
        # Assuming mm/dd/yy
        date = datetime.datetime(int(date[2]), int(date[0]), int(date[1]),
                                 int(time[0]), int(time[1]), int(time[2]))
        fid.seek(370)
        n_channels = np.fromfile(fid, dtype='<u2', count=1)[0]

        data_offset = n_channels * start * n_bytes + offset
        fid.seek(376)
        cnt_info['sfreq'] = np.fromfile(fid, dtype='<u2', count=1)[0]
        fid.seek(438)
        cnt_info['lowpass_toggle'] = np.fromfile(fid, 'i1', count=1)[0]
        cnt_info['highpass_toggle'] = np.fromfile(fid, 'i1', count=1)[0]
        fid.seek(869)
        cnt_info['lowcutoff'] = np.fromfile(fid, dtype='f4', count=1)[0]
        cnt_info['lowpoles'] = np.fromfile(fid, dtype='u2', count=1)[0]
        cnt_info['highcutoff'] = np.fromfile(fid, dtype='f4', count=1)[0]
        cnt_info['highpoles'] = np.fromfile(fid, dtype='u2', count=1)[0]

        # Bandpass=0 Notch=1 Highpass=2 Lowpass=3
        cnt_info['filtertype'] = np.fromfile(fid, dtype='i1', count=1)[0]
        # Frequency=0 Time=1
        cnt_info['filterdomain'] = np.fromfile(fid, dtype='i1', count=1)[0]
        fid.seek(886)
        cnt_info['event_offset'] = np.fromfile(fid, dtype='<i4', count=1)[0]
        cnt_info['continuous_seconds'] = np.fromfile(fid, dtype='<f4',
                                                     count=1)[0]
        cnt_info['channel_offset'] = np.fromfile(fid, dtype='<i4', count=1)[0]
        if cnt_info['channel_offset'] > 1 and read_blocks:
            cnt_info['channel_offset'] /= 2
            warn('Reading in data in blocks of %d. If this fails, try using '
                 'read_blocks=False.')
        cnt_info['n_samples'] = (cnt_info['event_offset'] -
                                 (900 + 75 * n_channels)) / (2 * n_channels)
        ch_names = list()
        cals = list()
        baselines = list()  # Baselines are subtracted before scaling the data.
        chs = list()
        for i in range(n_channels):
            fid.seek(data_offset + 75 * i)
            ch_names.append(''.join(np.fromfile(fid, dtype='S1', count=10)))
            fid.seek(data_offset + 75 * i + 47)
            baselines.append(np.fromfile(fid, dtype='i2', count=1)[0])
            fid.seek(data_offset + 75 * i + 59)
            sensitivity = np.fromfile(fid, dtype='f4', count=1)[0]
            fid.seek(data_offset + 75 * i + 71)
            cal = np.fromfile(fid, dtype='f4', count=1)
            cals.append(cal * sensitivity * 1e-6 / 204.8)
    cnt_info['n_channels'] = n_channels

    info = _empty_info(cnt_info['sfreq'])
    if cnt_info['lowpass_toggle'] == 1:
        info['lowpass'] = cnt_info['highcutoff']
    if cnt_info['highpass_toggle'] == 1:
        info['highpass'] = cnt_info['lowcutoff']
    info.update({'filename': input_fname,
                 'meas_date': calendar.timegm(date.utctimetuple()),
                 'description': None, 'buffer_size_sec': 10.})
    for idx, ch_name in enumerate(ch_names):
        ch_coil = FIFF.FIFFV_COIL_EEG
        ch_kind = FIFF.FIFFV_EEG_CH
        chan_info = {'cal': cals[idx], 'logno': idx + 1, 'scanno': idx + 1,
                     'range': 1.0, 'unit_mul': 0., 'ch_name': ch_name,
                     'unit': FIFF.FIFF_UNIT_V,
                     'coord_frame': FIFF.FIFFV_COORD_HEAD, 'loc': np.zeros(12),
                     'coil_type': ch_coil, 'kind': ch_kind}
        chs.append(chan_info)

    # Add the stim channel.
    chan_info = {'cal': 1.0, 'logno': len(chs) + 1, 'scanno': len(chs) + 1,
                 'range': 1.0, 'unit_mul': 0., 'ch_name': 'STI',
                 'unit': FIFF.FIFF_UNITM_NONE,
                 'coord_frame': FIFF.FIFFV_COORD_UNKNOWN, 'loc': np.zeros(12),
                 'coil_type': FIFF.FIFFV_COIL_NONE, 'kind': FIFF.FIFFV_SYST_CH}
    chs.append(chan_info)
    baselines.append(0)  # For stim channel
    cnt_info['baselines'] = np.array(baselines)
    info['chs'] = chs
    return info, cnt_info


class RawCNT(_BaseRaw):
    """Raw object from Neuroscan CNT file.

    Parameters
    ----------
    input_fname : str
        Path to the CNT file.
    read_blocks : bool
        Whether to read data in blocks. This is for dealing with different
        kinds of CNT data formats.
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
    def __init__(self, input_fname, read_blocks=True, preload=False,
                 verbose=None):
        input_fname = path.abspath(input_fname)
        info, cnt_info = _get_cnt_info(input_fname, read_blocks)
        last_samps = [cnt_info['n_samples'] - 1]
        super(RawCNT, self).__init__(
            info, preload, filenames=[input_fname], raw_extras=[cnt_info],
            last_samps=last_samps, orig_format='int',
            verbose=verbose)

    @verbose
    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Take a chunk of raw data, multiply by mult or cals, and store"""
        n_channels = self.info['nchan'] - 1
        n_samples = self._raw_extras[0]['n_samples']
        channel_offset = self._raw_extras[0]['channel_offset']
        baselines = self._raw_extras[0]['baselines']
        event_offset = self._raw_extras[0]['event_offset']
        n_bytes = 2
        # The data is divided into blocks of samples / channel.
        # channel_offset determines the amount of successive samples.
        # Here we use sample offset to align the data because start can be in
        # the middle of these blocks.
        s_offset = start % channel_offset

        with open(self._filenames[fi], 'rb', buffering=0) as fid:
            fid.seek(event_offset)
            event_type = np.fromfile(fid, dtype='<i1', count=1)[0]
            event_size = np.fromfile(fid, dtype='<i4', count=1)
            if event_type == 1:
                bytes = 8
            elif event_type == 2:
                bytes = 19
            else:
                raise IOError('Unexpected event size.')
            n_events = event_size / bytes
            events = list()
            for i in range(n_events):
                fid.seek(event_offset + 9 + i * bytes + 4)
                offset = np.fromfile(fid, dtype='<i4', count=1)
                events.append((offset - 900 - 75 * n_channels) /
                              (n_channels * 2))

            event_ch = np.zeros(n_samples)

            for event in events:
                event_ch[event - 1] = 1

            fid.seek(900 + 75 * n_channels + start * n_channels * n_bytes -
                     s_offset * n_channels * n_bytes)
            data_ = np.empty([n_channels, n_samples])
            # One extra sample set is read here to make sure the desired time
            # window is covered by the blocks.
            for sampleset in range(n_samples / channel_offset + 1):
                if sampleset * channel_offset >= stop - start + s_offset:
                    data_ = data_[:, s_offset:stop - start + s_offset]
                    break
                block = np.fromfile(fid, dtype='<i2',
                                    count=n_channels * channel_offset)
                block = block.reshape(n_channels, channel_offset, order='C')
                block_start = sampleset * channel_offset
                data_[:, block_start:block_start + channel_offset] = block
            event_block = event_ch[start:stop]
            data_ = np.vstack((data_, event_block))

            _mult_cal_one(data, data_ - baselines[:, None], idx, cals,
                          mult=None)
