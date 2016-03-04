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


def read_raw_cnt(input_fname, eog=(), ecg=(), emg=(), misc=(),
                 read_blocks=True, preload=False, verbose=None):
    """Read CNT data as raw object,

    Note: Channels that are not assigned with keywords ``eog``, ``ecg``,
    ``emg`` and ``misc`` are assigned as eeg channels. All the eeg channels are
    fit to a sphere when computing the z-coordinates for the channels.
    If channels assigned as eeg channels were placed away from the head (i.e.
    x and y coordinates don't fit to a sphere), all the channel locations will
    be distorted.

    Parameters
    ----------
    input_fname : str
        Path to the data file.
    eog : list | tuple | 'auto' | 'header'
        Names of channels or list of indices that should be designated
        EOG channels. If 'header', VEOG and HEOG channels assigned in the file
        header are used. If 'auto', channel names containing 'EOG' are used.
        Defaults to empty tuple.
    ecg : list or tuple | 'auto'
        Names of channels or list of indices that should be designated
        ECG channels. If 'auto', the channel names containing 'ECG' are used.
        Defaults to empty tuple.
    emg : list or tuple
        Names of channels or list of indices that should be designated
        EMG channels. If 'auto', the channel names containing 'EMG' are used.
        Defaults to empty tuple.
    misc : list or tuple
        Names of channels or list of indices that should be designated
        MISC channels. Defaults to empty tuple.
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
    return RawCNT(input_fname, eog=eog, ecg=ecg, emg=emg, misc=misc,
                  read_blocks=read_blocks, preload=preload, verbose=verbose)


def _get_cnt_info(input_fname, eog, ecg, emg, misc, read_blocks):
    """Helper for reading the cnt header."""
    data_offset = 900  # Size of the 'SETUP' header.
    cnt_info = dict()
    # Reading only the fields of interest. Structure of the whole header at
    # http://paulbourke.net/dataformats/eeg/
    with open(input_fname, 'rb', buffering=0) as fid:
        fid.seek(21)
        patient_id = ''.join(np.fromfile(fid, dtype='S1', count=20))
        patient_id = int(patient_id) if patient_id.isdigit() else 0
        fid.seek(121)
        patient_name = ''.join(np.fromfile(fid, dtype='S1', count=20)).split()
        last_name = patient_name[0]
        first_name = patient_name[-1]
        fid.seek(2, 1)
        sex = ''.join(np.fromfile(fid, dtype='S1', count=1))
        if sex == 'M':
            sex = 1
        elif sex == 'F':
            sex = 2
        else:  # can be 'U'
            sex = 0
        hand = ''.join(np.fromfile(fid, dtype='S1', count=1))
        if hand == 'R':
            hand = 1
        elif hand == 'L':
            hand = 2
        else:  # can be 'M' for mixed or 'U'
            hand = 0
        fid.seek(205)
        session_label = ''.join(np.fromfile(fid, dtype='S1', count=20))
        session_date = ''.join(np.fromfile(fid, dtype='S1', count=10))
        time = ''.join(np.fromfile(fid, dtype='S1', count=12))
        date = session_date.split('/')
        if len(date) != 3:
            warn('Could not parse meas date from the header. Setting to 0...')
            meas_date = 0
        else:
            if date[2].startswith('9'):
                date[2] = '19' + date[2]
            elif len(date[2]) == 2:
                date[2] = '20' + date[2]
            time = time.split(':')
            # Assuming mm/dd/yy
            date = datetime.datetime(int(date[2]), int(date[0]), int(date[1]),
                                     int(time[0]), int(time[1]), int(time[2]))
            meas_date = calendar.timegm(date.utctimetuple())
            if meas_date < 0:
                warn('Could not parse meas date from the header. Setting to '
                     '0...')
                meas_date = 0
        fid.seek(370)
        n_channels = np.fromfile(fid, dtype='<u2', count=1)[0]
        fid.seek(376)
        sfreq = np.fromfile(fid, dtype='<u2', count=1)[0]
        if eog == 'header':
            fid.seek(402)
            eog = [idx for idx in np.fromfile(fid, dtype='i2', count=2) if
                   idx >= 0]
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
        else:
            cnt_info['channel_offset'] = 1
        n_samples = (event_offset - (900 + 75 * n_channels)) / (2 * n_channels)
        ch_names, cals, baselines, chs, pos = (list(), list(), list(), list(),
                                               list())
        bads = list()
        for ch_idx in range(n_channels):  # ELECTLOC fields
            fid.seek(data_offset + 75 * ch_idx)
            ch_name = ''.join(np.fromfile(fid, dtype='S1', count=10))
            ch_names.append(ch_name)
            fid.seek(data_offset + 75 * ch_idx + 4)
            if np.fromfile(fid, dtype='u1', count=1)[0]:
                bads.append(ch_name)
            fid.seek(data_offset + 75 * ch_idx + 19)
            pos.append(np.fromfile(fid, dtype='f4', count=2))  # x and y pos
            fid.seek(data_offset + 75 * ch_idx + 47)
            # Baselines are subtracted before scaling the data.
            baselines.append(np.fromfile(fid, dtype='i2', count=1)[0])
            fid.seek(data_offset + 75 * ch_idx + 59)
            sensitivity = np.fromfile(fid, dtype='f4', count=1)[0]
            fid.seek(data_offset + 75 * ch_idx + 71)
            cal = np.fromfile(fid, dtype='f4', count=1)
            cals.append(cal * sensitivity * 1e-6 / 204.8)

        fid.seek(event_offset)
        event_type = np.fromfile(fid, dtype='<i1', count=1)[0]
        event_size = np.fromfile(fid, dtype='<i4', count=1)[0]
        if event_type == 1:
            event_bytes = 8
        elif event_type == 2:
            event_bytes = 19
        else:
            raise IOError('Unexpected event size.')

        n_events = event_size // event_bytes
        stim_channel = np.zeros(n_samples)  # Construct stim channel
        for i in range(n_events):
            fid.seek(event_offset + 9 + i * event_bytes)
            event_id = np.fromfile(fid, dtype='u2', count=1)[0]
            fid.seek(event_offset + 9 + i * event_bytes + 4)
            offset = np.fromfile(fid, dtype='<i4', count=1)[0]
            event_time = (offset - 900 - 75 * n_channels) // (n_channels * 2)
            stim_channel[event_time - 1] = event_id

    info = _empty_info(sfreq)
    if lowpass_toggle == 1:
        info['lowpass'] = highcutoff
    if highpass_toggle == 1:
        info['highpass'] = lowcutoff
    subject_info = {'hand': hand, 'id': patient_id, 'sex': sex,
                    'first_name': first_name, 'last_name': last_name}

    if eog == 'auto':
        eog = [ch for ch in ch_names if 'EOG' in ch.upper()]
    if ecg == 'auto':
        ecg = [ch for ch in ch_names if 'ECG' in ch.upper()]
    if emg == 'auto':
        emg = [ch for ch in ch_names if 'EMG' in ch.upper()]
    eegs = list()
    for idx, ch_name in enumerate(ch_names):
        if ch_name in eog or idx in eog:
            coil_type = FIFF.FIFFV_COIL_NONE
            kind = FIFF.FIFFV_EOG_CH
        elif ch_name in ecg or idx in ecg:
            coil_type = FIFF.FIFFV_COIL_NONE
            kind = FIFF.FIFFV_ECG_CH
        elif ch_name in emg or idx in emg:
            coil_type = FIFF.FIFFV_COIL_NONE
            kind = FIFF.FIFFV_EMG_CH
        elif ch_name in misc or idx in misc:
            coil_type = FIFF.FIFFV_COIL_NONE
            kind = FIFF.FIFFV_MISC_CH
        else:
            coil_type = FIFF.FIFFV_COIL_EEG
            kind = FIFF.FIFFV_EEG_CH
            eegs.append(idx)

        chan_info = {'cal': cals[idx], 'logno': idx + 1, 'scanno': idx + 1,
                     'range': 1.0, 'unit_mul': 0., 'ch_name': ch_name,
                     'unit': FIFF.FIFF_UNIT_V,
                     'coord_frame': FIFF.FIFFV_COORD_HEAD,
                     'coil_type': coil_type, 'kind': kind}
        chs.append(chan_info)

    coords = _topo_to_sphere(pos, eegs)
    locs = np.zeros((len(chs), 12), dtype=float)
    locs[:, :3] = coords
    for ch, loc in zip(chs, locs):
        ch.update(loc=loc)

    # Add the stim channel.
    chan_info = {'cal': 1.0, 'logno': len(chs) + 1, 'scanno': len(chs) + 1,
                 'range': 1.0, 'unit_mul': 0., 'ch_name': 'STI 014',
                 'unit': FIFF.FIFF_UNIT_NONE,
                 'coord_frame': FIFF.FIFFV_COORD_UNKNOWN, 'loc': np.zeros(12),
                 'coil_type': FIFF.FIFFV_COIL_NONE, 'kind': FIFF.FIFFV_STIM_CH}
    chs.append(chan_info)
    baselines.append(0)  # For stim channel
    cnt_info.update(baselines=np.array(baselines), n_samples=n_samples,
                    stim_channel=stim_channel)
    info.update(filename=input_fname, meas_date=np.array([meas_date, 0]),
                description=str(session_label), buffer_size_sec=10., bads=bads,
                subject_info=subject_info, chs=chs)
    info._check_consistency()
    return info, cnt_info


def _topo_to_sphere(pos, eegs):
    """Helper function for transforming xy-coordinates to sphere.
    Parameters
    ----------
    pos : array of shape (chs, 2)
        xy-oordinates to transform.
    eegs : list of int
        Indices of eeg channels that are included when calculating the sphere.

    Returns
    -------
    coords : list of shape (chs, 3)
        xyz-coordinates.
    """
    xs = np.array(pos)[:, 0]
    ys = np.array(pos)[:, 1]

    ys -= min(ys)  # Normalize the points
    xs -= min(xs)
    xs /= max(xs)
    ys /= max(ys)

    xs += 0.5 - np.mean(xs[eegs])  # Centralize the points
    ys += 0.5 - np.mean(ys[eegs])

    xs *= 2  # Values ranging from -1 to 1
    ys *= 2
    xs -= 1.
    ys -= 1.

    sqs = max(np.sqrt((xs[eegs] ** 2) + (ys[eegs] ** 2)))  # Shape to a sphere
    xs /= sqs
    ys /= sqs

    coords = list()
    for x, y in zip(xs, ys):
        r = np.sqrt(x ** 2 + y ** 2)
        if r > 1:  # If a point is outside the sphere set z=0.
            coords.append([x, y, 0])
            continue
        alpha = np.arccos(r)
        z = np.sin(alpha)
        coords.append([x, y, z])
    return coords


class RawCNT(_BaseRaw):
    """Raw object from Neuroscan CNT file.

    Note: Channels that are not assigned with keywords ``eog``, ``ecg``,
    ``emg`` and ``misc`` are assigned as eeg channels. All the eeg channels are
    fit to a sphere when computing the z-coordinates for the channels.
    If channels assigned as eeg channels were placed away from the head (i.e.
    x and y coordinates don't fit to a sphere), all the channel locations will
    be distorted.

    Parameters
    ----------
    input_fname : str
        Path to the CNT file.
    eog : list | tuple
        Names of channels or list of indices that should be designated
        EOG channels. If 'auto', the channel names beginning with
        ``EOG`` are used. Defaults to empty tuple.
    ecg : list or tuple
        Names of channels or list of indices that should be designated
        ECG channels. If 'auto', the channel names beginning with
        ``ECG`` are used. Defaults to empty tuple.
    emg : list or tuple
        Names of channels or list of indices that should be designated
        EMG channels. If 'auto', the channel names beginning with
        ``EMG`` are used. Defaults to empty tuple.
    misc : list or tuple
        Names of channels or list of indices that should be designated
        MISC channels. Defaults to empty tuple.
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
    def __init__(self, input_fname, eog=(), ecg=(), emg=(), misc=(),
                 read_blocks=True, preload=False, verbose=None):
        input_fname = path.abspath(input_fname)
        info, cnt_info = _get_cnt_info(input_fname, eog, ecg, emg, misc,
                                       read_blocks)
        last_samps = [cnt_info['n_samples'] - 1]
        super(RawCNT, self).__init__(
            info, preload, filenames=[input_fname], raw_extras=[cnt_info],
            last_samps=last_samps, orig_format='int',
            verbose=verbose)

    @verbose
    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Take a chunk of raw data, multiply by mult or cals, and store"""
        n_channels = self.info['nchan'] - 1  # Stim channel already read.
        channel_offset = self._raw_extras[0]['channel_offset']
        baselines = self._raw_extras[0]['baselines']
        stim_ch = self._raw_extras[0]['stim_channel']
        n_bytes = 2
        sel = np.arange(n_channels + 1)[idx]
        chunk_size = channel_offset * n_channels  # Size of chunks in file.
        # The data is divided into blocks of samples / channel.
        # channel_offset determines the amount of successive samples.
        # Here we use sample offset to align the data because start can be in
        # the middle of these blocks.
        data_left = (stop - start) * n_channels
        # Read up to 100 MB of data at a time, block_size is in data samples
        block_size = ((int(100e6) // n_bytes) // chunk_size) * chunk_size
        block_size = min(data_left, block_size)
        s_offset = start % channel_offset
        with open(self._filenames[fi], 'rb', buffering=0) as fid:
            fid.seek(900 + n_channels * (75 + (start - s_offset) * n_bytes))
            for sample_start in np.arange(0, data_left,
                                          block_size) // n_channels:
                sample_stop = sample_start + min((block_size // n_channels,
                                                  data_left // n_channels -
                                                  sample_start))
                n_samps = sample_stop - sample_start
                data_ = np.empty((n_channels + 1, n_samps))
                # In case channel offset and start time do not align perfectly,
                # extra sample sets are read here to cover the desired time
                # window. The whole (up to 100 MB) block is read at once and
                # then reshaped to (n_channels, n_samples).
                extra_samps = chunk_size if (s_offset != 0 or n_samps %
                                             channel_offset != 0) else 0
                if s_offset >= (channel_offset / 2.):  # Extend at the end.
                    extra_samps += chunk_size
                count = n_samps // channel_offset * chunk_size + extra_samps
                n_chunks = count // chunk_size
                samps = np.fromfile(fid, dtype='<i2', count=count)
                samps = samps.reshape((n_chunks, n_channels, channel_offset),
                                      order='C')
                # Intermediate shaping to chunk sizes.
                block = np.zeros((n_channels + 1, channel_offset * n_chunks))
                for set_idx, row in enumerate(samps):  # Final shape.
                    block_slice = slice(set_idx * channel_offset,
                                        (set_idx + 1) * channel_offset)
                    block[:-1, block_slice] = row
                block = block[sel, s_offset:n_samps + s_offset]
                data_[sel] = block
                data_[-1] = stim_ch[start + sample_start:start + sample_stop]
                data -= baselines[sel, None]
                _mult_cal_one(data[:, sample_start:sample_stop], data_, idx,
                              cals, mult=None)
