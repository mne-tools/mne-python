# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#          simplified BSD-3 license

import datetime
import os
import os.path as op
import time

import numpy as np

from .. import Raw
from ..meas_info import Info
from ..constants import FIFF
from ...utils import verbose, logger

_other_fields = [
    'orig_fid_str', 'lowpass', 'buffer_size_sec', 'dev_ctf_t',
    'meas_id', 'subject_info',
    'dev_head_t', 'line_freq', 'acq_stim', 'proj_id', 'description',
    'highpass', 'experimenter', 'file_id', 'filenames', 'proj_name',
    'dig', 'ctf_head_t', 'orig_blocks', 'acq_pars'
]


def _read_header(fid):
    """Read EGI binary header"""

    version = np.fromfile(fid, np.int32, 1)[0]

    if version > 6 & ~np.bitwise_and(version, 6):
        version = version.byteswap().astype(np.uint32)
    else:
        ValueError('Watchout. This does not seem to be a simple '
                   'binary EGI file.')
    my_fread = lambda *x, **y: np.fromfile(*x, **y)[0]
    info = dict(
        version=version,
        year=my_fread(fid, '>i2', 1),
        month=my_fread(fid, '>i2', 1),
        day=my_fread(fid, '>i2', 1),
        hour=my_fread(fid, '>i2', 1),
        minute=my_fread(fid, '>i2', 1),
        second=my_fread(fid, '>i2', 1),
        millisecond=my_fread(fid, '>i4', 1),
        samp_rate=my_fread(fid, '>i2', 1),
        n_channels=my_fread(fid, '>i2', 1),
        gain=my_fread(fid, '>i2', 1),
        bits=my_fread(fid, '>i2', 1),
        value_range=my_fread(fid, '>i2', 1)
    )

    unsegmented = 1 if np.bitwise_and(version, 1) == 0 else 0
    precision = np.bitwise_and(version, 6)
    if precision == 0:
        RuntimeError('Floating point precision is undefined.')

    if unsegmented:
        info.update(dict(n_categories=0,
                         n_segments=1,
                         n_samples=np.fromfile(fid, '>i4', 1)[0],
                         n_events=np.fromfile(fid, '>i2', 1)[0],
                         event_codes=[],
                         category_names=[],
                         category_lengths=[],
                         pre_baseline=0))
        for event in range(info['n_events']):
            info['event_codes'].append(''.join(np.fromfile(fid, '>c2', 4)))
        info['event_codes'] = np.array(info['event_codes'])
    else:
        raise NotImplementedError('Only continous files are supported')

    info.update(dict(precision=precision, unsegmented=unsegmented))

    return info


def _read_events(fid, info):
    """Read events"""
    unpack = [info[k] for k in ['n_events', 'n_segments', 'n_channels']]
    n_events, n_segments, n_channels = unpack
    n_samples = 1 if info['unsegmented'] else info['n_samples']
    events = np.zeros([n_events, n_segments * info['n_samples']])
    dtype, bytesize = {2: ('>i2', 2), 4: ('>f4', 4),
                       6: ('>f8', 8)}[info['precision']]

    info.update({'dtype': dtype, 'bytesize': bytesize})
    beg_dat = fid.tell()

    for ii in range(info['n_events']):
        fid.seek(beg_dat + (int(n_channels) + ii) * bytesize, 0)
        events[ii] = np.fromfile(fid, dtype, n_samples)
        fid.seek(int((n_channels + n_events) * bytesize), 1)
    return events


def _read_data(fid, info):
    """Aux function"""
    if not info['unsegmented']:
        raise NotImplementedError('Only continous files are supported')

    fid.seek(36 + info['n_events'] * 4, 0)  # skip header
    readsize = (info['n_channels'] + info['n_events']) * info['n_samples']
    final_shape = (info['n_samples'], info['n_channels'] + info['n_events'])
    data = np.fromfile(fid, info['dtype'], readsize).reshape(final_shape).T
    return data


def _combine_triggers(info, data, remapping=None):
    """Combine binary triggers"""
    new_trigger = np.zeros(data[0].shape)
    first = np.nonzero(data[0])[0]
    for d in data[1:]:
        if np.any(d.nonzero()[0] == first):
            raise RuntimeError('Events must be mutually exclusive')

    if remapping is None:
        remapping = np.arange(data) + 1

    for d, event_id in zip(data, remapping):
        idx = d.nonzero()
        if np.any(idx):
            new_trigger[idx] += event_id

    return new_trigger[None]


def read_raw_egi(input_fname, event_ids=None):
    """Read EGI simple binary as raw object

    Parameters
    ----------
    input_fname : str
        Path to the raw file.
    event_ids : list of int | None
        The integer values that will be assigned to the synthetized trigger
        channel based on the event codes found. If None, equals
        np.arange(n_events) + 1.
        Note. The trigger channel is artifically constructed based on
        timestamps received by the Netstation. As a consequence triggers only
        last ofr one sample. You therefore need to use `shortest_event=1`
        when using `mne.find_events`.

    Returns
    -------
    raw : instance of mne.fiff.Raw
        A raw object containing EGI data.
    """
    return _RawEGI(input_fname, event_ids)


class _RawEGI(Raw):
    """Raw object from EGI simple binary file
    """
    def __init__(self, input_fname, event_ids=None):
        """docstring for __init__"""
        with open(input_fname, 'r') as fid:
            logger.info('Reading EGI header from %s...' % input_fname)
            egi_info = _read_header(fid)
            logger.info('Reading events ...')
            events = _read_events(fid, egi_info)
            logger.info('Reading data ...')
            data = _read_data(fid, egi_info)

        logger.info('Assembling measurement info ...')
        if event_ids is None:
            event_ids = np.arange(egi_info['n_events']) + 1

        new_trigger = _combine_triggers(egi_info,
                                        data[-egi_info['n_events']:],
                                        remapping=event_ids)
        self._data = data = np.concatenate([data, new_trigger])
        self.info = info = Info(dict((k, None) for k in _other_fields))
        info['sfreq'] = egi_info['samp_rate']
        info['filename'] = input_fname
        info['filenames'] = []
        my_time = datetime.datetime(
            egi_info['year'],
            egi_info['month'],
            egi_info['day'],
            egi_info['hour'],
            egi_info['minute'],
            egi_info['second']
        )
        my_timestamp = time.mktime(my_time.timetuple())
        info['meas_date'] = np.array([my_timestamp], dtype=np.float32)
        info['projs'] = []
        ch_names = ['EEG %i' % (i + 1) for i in range(egi_info['n_channels'])]
        ch_names.extend(list(egi_info['event_codes']))
        ch_names.append('STI 014')  # our new_trigger
        info['nchan'] = len(data)
        info['chs'] = []
        info['ch_names'] = ch_names
        info['bads'] = []
        info['comps'] = []
        for ii, ch_name in enumerate(ch_names):
            ch_info = {'cal': 1.0,
                       'logno': ii + 1,
                       'scanno': ii + 1,
                       'range': 1.0,
                       'unit_mul': 0,
                       'ch_name': ch_name,
                       'unit': FIFF.FIFF_UNIT_V,
                       'coord_frame': FIFF.FIFFV_COORD_HEAD,
                       'coil_type': FIFF.FIFFV_COIL_EEG,
                       'kind': FIFF.FIFFV_EEG_CH,
                       'eeg_loc': None,
                       'loc': np.array([0, 0, 0, 1] * 3, dtype='f4')}

            if len(ch_name) == 4 or ch_name.startswith('STI'):
                u = {'unit_mul': 0,
                     'coil_type': FIFF.FIFFV_COIL_NONE,
                     'unit': FIFF.FIFF_UNIT_NONE,
                     'kind': FIFF.FIFFV_STIM_CH}
                ch_info.update(u)
            info['chs'].append(ch_info)

        self._preloaded = True
        self.first_samp, self.last_samp = 0, self._data.shape[1] - 1
        self._times = np.arange(self.first_samp, self.last_samp + 1,
                                dtype=np.float64)
        self._times /= self.info['sfreq']
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs'
                    % (self.first_samp, self.last_samp,
                       float(self.first_samp) / self.info['sfreq'],
                       float(self.last_samp) / self.info['sfreq']))

        # Raw attributes
        self.fids = list()
        self._projector = None
        self.first_samp = 0
        self.last_samp = egi_info['n_samples'] - 1
        self.comp = None  # no compensation for EDF
        self.proj = False
        self._first_samps = np.array([self.first_samp])
        self._last_samps = np.array([self.last_samp])
        self._raw_lengths = np.array([egi_info['n_samples']])
        self.rawdirs = np.array([])
        self.cals = np.array([])
        # use information from egi
        self.orig_format = {'>f4': 'single', '>f4': 'double',
                            '>i2': 'int'}[egi_info['dtype']]

        logger.info('Ready.')

    def __repr__(self):
        n_chan = self.info['nchan']
        data_range = self.last_samp - self.first_samp + 1
        s = ('%r' % os.path.basename(self.info['filename']),
             "n_channels x n_times : %s x %s" % (n_chan, data_range))
        return "<RawEGI  |  %s>" % ', '.join(s)
