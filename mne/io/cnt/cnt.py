"""Conversion tool from Neuroscan CNT to FIF."""

# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)
from os import path
import datetime
import calendar

import numpy as np

from ...utils import warn, verbose
from ...channels.layout import _topo_to_sphere
from ..constants import FIFF
from ..utils import _mult_cal_one, _find_channels, _create_chs
from ..meas_info import _empty_info
from ..base import BaseRaw, _check_update_montage
from ..utils import read_str


def read_raw_cnt(input_fname, montage, eog=(), misc=(), ecg=(), emg=(),
                 data_format='auto', date_format='mm/dd/yy', preload=False,
                 verbose=None):
    """Read CNT data as raw object.

    .. Note::
        If montage is not provided, the x and y coordinates are read from the
        file header. Channels that are not assigned with keywords ``eog``,
        ``ecg``, ``emg`` and ``misc`` are assigned as eeg channels. All the eeg
        channel locations are fit to a sphere when computing the z-coordinates
        for the channels. If channels assigned as eeg channels have locations
        far away from the head (i.e. x and y coordinates don't fit to a
        sphere), all the channel locations will be distorted. If you are not
        sure that the channel locations in the header are correct, it is
        probably safer to use a (standard) montage. See
        :func:`mne.channels.read_montage`

    Parameters
    ----------
    input_fname : str
        Path to the data file.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions. If None,
        xy sensor locations are read from the header (``x_coord`` and
        ``y_coord`` in ``ELECTLOC``) and fit to a sphere. See the documentation
        of :func:`mne.channels.read_montage` for more information.
    eog : list | tuple | 'auto' | 'header'
        Names of channels or list of indices that should be designated
        EOG channels. If 'header', VEOG and HEOG channels assigned in the file
        header are used. If 'auto', channel names containing 'EOG' are used.
        Defaults to empty tuple.
    misc : list | tuple
        Names of channels or list of indices that should be designated
        MISC channels. Defaults to empty tuple.
    ecg : list | tuple | 'auto'
        Names of channels or list of indices that should be designated
        ECG channels. If 'auto', the channel names containing 'ECG' are used.
        Defaults to empty tuple.
    emg : list | tuple
        Names of channels or list of indices that should be designated
        EMG channels. If 'auto', the channel names containing 'EMG' are used.
        Defaults to empty tuple.
    data_format : 'auto' | 'int16' | 'int32'
        Defines the data format the data is read in. If 'auto', it is
        determined from the file header using ``numsamples`` field.
        Defaults to 'auto'.
    date_format : str
        Format of date in the header. Currently supports 'mm/dd/yy' (default)
        and 'dd/mm/yy'.
    preload : bool | str (default False)
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
    raw : instance of RawCNT.
        The raw data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    .. versionadded:: 0.12
    """
    return RawCNT(input_fname, montage=montage, eog=eog, misc=misc, ecg=ecg,
                  emg=emg, data_format=data_format, date_format=date_format,
                  preload=preload, verbose=verbose)


def _get_cnt_info(input_fname, eog, ecg, emg, misc, data_format, date_format):
    """Read the cnt header."""
    data_offset = 900  # Size of the 'SETUP' header.
    cnt_info = dict()
    # Reading only the fields of interest. Structure of the whole header at
    # http://paulbourke.net/dataformats/eeg/
    with open(input_fname, 'rb', buffering=0) as fid:
        fid.seek(21)
        patient_id = read_str(fid, 20)
        patient_id = int(patient_id) if patient_id.isdigit() else 0
        fid.seek(121)
        patient_name = read_str(fid, 20).split()
        last_name = patient_name[0] if len(patient_name) > 0 else ''
        first_name = patient_name[-1] if len(patient_name) > 0 else ''
        fid.seek(2, 1)
        sex = read_str(fid, 1)
        if sex == 'M':
            sex = FIFF.FIFFV_SUBJ_SEX_MALE
        elif sex == 'F':
            sex = FIFF.FIFFV_SUBJ_SEX_FEMALE
        else:  # can be 'U'
            sex = FIFF.FIFFV_SUBJ_SEX_UNKNOWN
        hand = read_str(fid, 1)
        if hand == 'R':
            hand = FIFF.FIFFV_SUBJ_HAND_RIGHT
        elif hand == 'L':
            hand = FIFF.FIFFV_SUBJ_HAND_LEFT
        else:  # can be 'M' for mixed or 'U'
            hand = None
        fid.seek(205)
        session_label = read_str(fid, 20)
        session_date = read_str(fid, 10)
        time = read_str(fid, 12)
        date = session_date.split('/')
        if len(date) != 3:
            meas_date = -1
        else:
            if date[2].startswith('9'):
                date[2] = '19' + date[2]
            elif len(date[2]) == 2:
                date[2] = '20' + date[2]
            time = time.split(':')
            if len(time) == 3:
                if date_format == 'mm/dd/yy':
                    pass
                elif date_format == 'dd/mm/yy':
                    date[0], date[1] = date[1], date[0]
                else:
                    raise ValueError("Only date formats 'mm/dd/yy' and "
                                     "'dd/mm/yy' supported. "
                                     "Got '%s'." % date_format)
                # Assuming mm/dd/yy
                date = datetime.datetime(int(date[2]), int(date[0]),
                                         int(date[1]), int(time[0]),
                                         int(time[1]), int(time[2]))
                meas_date = calendar.timegm(date.utctimetuple())
            else:
                meas_date = -1
        if meas_date < 0:
            warn('  Could not parse meas date from the header. Setting to '
                 '[0, 0]...')
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

        # Header has a field for number of samples, but it does not seem to be
        # too reliable. That's why we have option for setting n_bytes manually.
        fid.seek(864)
        n_samples = np.fromfile(fid, dtype='<i4', count=1)[0]
        fid.seek(869)
        lowcutoff = np.fromfile(fid, dtype='f4', count=1)[0]
        fid.seek(2, 1)
        highcutoff = np.fromfile(fid, dtype='f4', count=1)[0]

        fid.seek(886)
        event_offset = np.fromfile(fid, dtype='<i4', count=1)[0]
        cnt_info['continuous_seconds'] = np.fromfile(fid, dtype='<f4',
                                                     count=1)[0]

        data_size = event_offset - (900 + 75 * n_channels)
        if data_format == 'auto':
            if (n_samples == 0 or
                    data_size // (n_samples * n_channels) not in [2, 4]):
                warn('Could not define the number of bytes automatically. '
                     'Defaulting to 2.')
                n_bytes = 2
                n_samples = data_size // (n_bytes * n_channels)
            else:
                n_bytes = data_size // (n_samples * n_channels)
        else:
            if data_format not in ['int16', 'int32']:
                raise ValueError("data_format should be 'auto', 'int16' or "
                                 "'int32'. Got %s." % data_format)
            n_bytes = 2 if data_format == 'int16' else 4
            n_samples = data_size // (n_bytes * n_channels)
        # Channel offset refers to the size of blocks per channel in the file.
        cnt_info['channel_offset'] = np.fromfile(fid, dtype='<i4', count=1)[0]
        if cnt_info['channel_offset'] > 1:
            cnt_info['channel_offset'] //= n_bytes
        else:
            cnt_info['channel_offset'] = 1
        ch_names, cals, baselines, chs, pos = (list(), list(), list(), list(),
                                               list())
        bads = list()
        for ch_idx in range(n_channels):  # ELECTLOC fields
            fid.seek(data_offset + 75 * ch_idx)
            ch_name = read_str(fid, 10)
            ch_names.append(ch_name)
            fid.seek(data_offset + 75 * ch_idx + 4)
            if np.fromfile(fid, dtype='u1', count=1)[0]:
                bads.append(ch_name)
            fid.seek(data_offset + 75 * ch_idx + 19)
            xy = np.fromfile(fid, dtype='f4', count=2)
            xy[1] *= -1  # invert y-axis
            pos.append(xy)
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
        elif event_type in (2, 3):
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
            if event_type == 3:
                offset *= n_bytes * n_channels
            event_time = offset - 900 - 75 * n_channels
            event_time //= n_channels * n_bytes
            stim_channel[event_time - 1] = event_id

    info = _empty_info(sfreq)
    if lowpass_toggle == 1:
        info['lowpass'] = highcutoff
    if highpass_toggle == 1:
        info['highpass'] = lowcutoff
    subject_info = {'hand': hand, 'id': patient_id, 'sex': sex,
                    'first_name': first_name, 'last_name': last_name}

    if eog == 'auto':
        eog = _find_channels(ch_names, 'EOG')
    if ecg == 'auto':
        ecg = _find_channels(ch_names, 'ECG')
    if emg == 'auto':
        emg = _find_channels(ch_names, 'EMG')

    chs = _create_chs(ch_names, cals, FIFF.FIFFV_COIL_EEG,
                      FIFF.FIFFV_EEG_CH, eog, ecg, emg, misc)
    eegs = [idx for idx, ch in enumerate(chs) if
            ch['coil_type'] == FIFF.FIFFV_COIL_EEG]
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
                    stim_channel=stim_channel, n_bytes=n_bytes)
    info.update(meas_date=np.array([meas_date, 0]),
                description=str(session_label), buffer_size_sec=10., bads=bads,
                subject_info=subject_info, chs=chs)
    info._update_redundant()
    return info, cnt_info


class RawCNT(BaseRaw):
    """Raw object from Neuroscan CNT file.

    .. Note::
        If montage is not provided, the x and y coordinates are read from the
        file header. Channels that are not assigned with keywords ``eog``,
        ``ecg``, ``emg`` and ``misc`` are assigned as eeg channels. All the eeg
        channel locations are fit to a sphere when computing the z-coordinates
        for the channels. If channels assigned as eeg channels have locations
        far away from the head (i.e. x and y coordinates don't fit to a
        sphere), all the channel locations will be distorted. If you are not
        sure that the channel locations in the header are correct, it is
        probably safer to use a (standard) montage. See
        :func:`mne.channels.read_montage`

    Parameters
    ----------
    input_fname : str
        Path to the CNT file.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions. If None,
        xy sensor locations are read from the header (``x_coord`` and
        ``y_coord`` in ``ELECTLOC``) and fit to a sphere. See the documentation
        of :func:`mne.channels.read_montage` for more information.
    eog : list | tuple
        Names of channels or list of indices that should be designated
        EOG channels. If 'auto', the channel names beginning with
        ``EOG`` are used. Defaults to empty tuple.
    misc : list | tuple
        Names of channels or list of indices that should be designated
        MISC channels. Defaults to empty tuple.
    ecg : list | tuple
        Names of channels or list of indices that should be designated
        ECG channels. If 'auto', the channel names beginning with
        ``ECG`` are used. Defaults to empty tuple.
    emg : list | tuple
        Names of channels or list of indices that should be designated
        EMG channels. If 'auto', the channel names beginning with
        ``EMG`` are used. Defaults to empty tuple.
    data_format : 'auto' | 'int16' | 'int32'
        Defines the data format the data is read in. If 'auto', it is
        determined from the file header using ``numsamples`` field.
        Defaults to 'auto'.
    date_format : str
        Format of date in the header. Currently supports 'mm/dd/yy' (default)
        and 'dd/mm/yy'.
    preload : bool | str (default False)
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

    def __init__(self, input_fname, montage, eog=(), misc=(), ecg=(), emg=(),
                 data_format='auto', date_format='mm/dd/yy', preload=False,
                 verbose=None):  # noqa: D102
        input_fname = path.abspath(input_fname)
        info, cnt_info = _get_cnt_info(input_fname, eog, ecg, emg, misc,
                                       data_format, date_format)
        last_samps = [cnt_info['n_samples'] - 1]
        _check_update_montage(info, montage)
        super(RawCNT, self).__init__(
            info, preload, filenames=[input_fname], raw_extras=[cnt_info],
            last_samps=last_samps, orig_format='int',
            verbose=verbose)

    @verbose
    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Take a chunk of raw data, multiply by mult or cals, and store."""
        n_channels = self.info['nchan'] - 1  # Stim channel already read.
        channel_offset = self._raw_extras[0]['channel_offset']
        baselines = self._raw_extras[0]['baselines']
        stim_ch = self._raw_extras[0]['stim_channel']
        n_bytes = self._raw_extras[0]['n_bytes']
        dtype = '<i4' if n_bytes == 4 else '<i2'
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
                if s_offset >= (channel_offset / 2):  # Extend at the end.
                    extra_samps += chunk_size
                count = n_samps // channel_offset * chunk_size + extra_samps
                n_chunks = count // chunk_size
                samps = np.fromfile(fid, dtype=dtype, count=count)
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
                data_[sel] -= baselines[sel][:, None]
                _mult_cal_one(data[:, sample_start:sample_stop], data_, idx,
                              cals, mult=None)
