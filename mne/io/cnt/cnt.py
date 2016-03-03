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
        lowpass_toggle = np.fromfile(fid, 'i1', count=1)[0]
        highpass_toggle = np.fromfile(fid, 'i1', count=1)[0]
        fid.seek(869)
        lowcutoff = np.fromfile(fid, dtype='f4', count=1)[0]
        fid.seek(2, 1)
        highcutoff = np.fromfile(fid, dtype='f4', count=1)[0]

        fid.seek(886)
        event_offset = np.fromfile(fid, dtype='<i4', count=1)[0]
        cnt_info['continuous_seconds'] = np.fromfile(fid, dtype='<f4',
                                                     count=1)[0]
        # Channel offset refers to the size of blocks per channel in the file.
        cnt_info['channel_offset'] = np.fromfile(fid, dtype='<i4', count=1)[0]
        if cnt_info['channel_offset'] > 1 and read_blocks:
            cnt_info['channel_offset'] /= 2  # Data read as 2 byte ints.
            warn('Reading in data in blocks of %d. If this fails, try using '
                 'read_blocks=False.')
        n_samples = (event_offset - (900 + 75 * n_channels)) / (2 * n_channels)
        ch_names, cals, baselines, chs, pos = (list(), list(), list(), list(),
                                               list())
        size = list()
        for ch_idx in range(n_channels):  # ELECTLOC fields
            fid.seek(data_offset + 75 * ch_idx)
            ch_names.append(''.join(np.fromfile(fid, dtype='S1', count=10)))
            fid.seek(data_offset + 75 * ch_idx + 19)
            pos.append(np.fromfile(fid, dtype='f4', count=2))  # x and y pos
            fid.seek(data_offset + 75 * ch_idx + 47)
            # Baselines are subtracted before scaling the data.
            baselines.append(np.fromfile(fid, dtype='i2', count=1)[0])
            fid.seek(data_offset + 75 * ch_idx + 59)
            sensitivity = np.fromfile(fid, dtype='f4', count=1)[0]
            fid.seek(data_offset + 75 * ch_idx + 66)
            size.append(np.fromfile(fid, dtype='u1', count=5))
            fid.seek(data_offset + 75 * ch_idx + 71)
            cal = np.fromfile(fid, dtype='f4', count=1)
            cals.append(cal * sensitivity * 1e-6 / 204.8)

        fid.seek(event_offset)
        event_type = np.fromfile(fid, dtype='<i1', count=1)[0]
        event_size = np.fromfile(fid, dtype='<i4', count=1)
        if event_type == 1:
            event_bytes = 8
        elif event_type == 2:
            event_bytes = 19
        else:
            raise IOError('Unexpected event size.')
        n_events = event_size // event_bytes
        events = list()
        for i in range(n_events):
            fid.seek(event_offset + 9 + i * event_bytes + 4)
            offset = np.fromfile(fid, dtype='<i4', count=1)
            events.append((offset - 900 - 75 * n_channels) //
                          (n_channels * 2))
        stim_channel = np.zeros(n_samples)

        for event in events:
            stim_channel[event - 1] = 1

    info = _empty_info(cnt_info['sfreq'])
    if lowpass_toggle == 1:
        info['lowpass'] = highcutoff
    if highpass_toggle == 1:
        info['highpass'] = lowcutoff
    info.update(filename=input_fname,
                meas_date=calendar.timegm(date.utctimetuple()),
                description=None, buffer_size_sec=10.)

    coords = _topo_to_sphere(pos)
    for idx, ch_name in enumerate(ch_names):
        ch_coil = FIFF.FIFFV_COIL_EEG
        ch_kind = FIFF.FIFFV_EEG_CH
        loc = np.zeros(12)
        loc[:3] = coords[idx]

        chan_info = {'cal': cals[idx], 'logno': idx + 1, 'scanno': idx + 1,
                     'range': 1.0, 'unit_mul': 0., 'ch_name': ch_name,
                     'unit': FIFF.FIFF_UNIT_V,
                     'coord_frame': FIFF.FIFFV_COORD_HEAD, 'loc': loc,
                     'coil_type': ch_coil, 'kind': ch_kind}
        chs.append(chan_info)

    # Add the stim channel.
    chan_info = {'cal': 1.0, 'logno': len(chs) + 1, 'scanno': len(chs) + 1,
                 'range': 1.0, 'unit_mul': 0., 'ch_name': 'STI 014',
                 'unit': FIFF.FIFF_UNIT_NONE,
                 'coord_frame': FIFF.FIFFV_COORD_UNKNOWN, 'loc': np.zeros(12),
                 'coil_type': FIFF.FIFFV_COIL_NONE, 'kind': FIFF.FIFFV_STIM_CH}
    chs.append(chan_info)
    baselines.append(0)  # For stim channel
    cnt_info.update(baselines=np.array(baselines), n_samples=n_samples,
                    n_channels=n_channels, stim_channel=stim_channel)
    info['chs'] = chs
    info._check_consistency()
    return info, cnt_info


def _topo_to_sphere(pos):
    """Helper function for transforming xy-coordinates to sphere.
    Parameters
    ----------
    pos : array of shape (xs, ys)
        Coordinates to transform.

    Returns
    -------
    coords : list
        xyz-coordinates.
    """
    xs = np.array(pos)[:, 0]
    ys = np.array(pos)[:, 1]

    xs -= min(xs)  # First normalize the points.
    ys -= min(ys)
    xs *= (2. / max(xs))
    ys *= (2. / max(ys))
    xs -= 1.  # Values range from -1 to 1
    ys -= 1.
    coords = list()
    for x, y in zip(xs, ys):
        t = np.sqrt(x ** 2 + y ** 2)
        if t > 1:  # Force the points to the surface of a sphere.
            t = 1.
        alpha = np.arccos(t)
        z = np.sin(alpha)
        coords.append([x, y, z])
    return coords


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
        channel_offset = self._raw_extras[0]['channel_offset']
        baselines = self._raw_extras[0]['baselines']
        n_bytes = 2
        # The data is divided into blocks of samples / channel.
        # channel_offset determines the amount of successive samples.
        # Here we use sample offset to align the data because start can be in
        # the middle of these blocks.
        s_offset = start % channel_offset
        sel = np.arange(n_channels + 1)[idx]
        block_size = channel_offset * n_channels  # Size of blocks in file.
        with open(self._filenames[fi], 'rb', buffering=0) as fid:
            fid.seek(900 + n_channels * (75 + (start - s_offset) * n_bytes))
            data_ = np.empty((n_channels + 1, data.shape[1]))
            n_samps = stop - start

            # In case channel offset and start time do not align perfectly,
            # extra sample sets are read here to cover the desired time window.
            # The whole block is read at once and then reshaped to
            # (n_channels, n_samples).
            extra_samps = block_size if (s_offset != 0 or
                                         n_samps % channel_offset != 0) else 0
            if s_offset >= (channel_offset // 2):  # Extend at the end.
                extra_samps += block_size
            count = n_samps // channel_offset * block_size + extra_samps
            n_samps = count // block_size
            samps = np.fromfile(fid, dtype='<i2', count=count)
            samps = samps.reshape((n_samps, n_channels, channel_offset),
                                  order='C')
        # Intermediate shaping to block sizes.
        block = np.zeros((n_channels + 1, channel_offset * n_samps))
        for set_idx, row in enumerate(samps):  # Final shape.
            block[:-1, set_idx * channel_offset:(set_idx +
                                                 1) * channel_offset] = row
        block = block[sel, s_offset:stop - start + s_offset]
        data_[sel] = block
        data_[-1] = self._raw_extras[0]['stim_channel'][start:stop]
        data -= baselines[sel, None]
        _mult_cal_one(data, data_, idx, cals, mult=None)
