# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#          simplified BSD-3 license

import numpy as np
import os.path as op
from . import Raw
from .meas_info import Info
from .constants import FIFF


def _read_header(fname):
    """Read EGI binary header"""

    fid = open(fname, 'r')
    version = np.fromfile(fid, np.int32, 1)[0]

    if version > 6 & ~np.bitwise_and(version, 6):
        version = version.byteswap().astype(np.uint32)
    else:
        ValueError('This is not a simple binary file.')

    info = dict(
        version=version,
        year=np.fromfile(fid, '>i2', 1),
        month=np.fromfile(fid, '>i2', 1),
        day=np.fromfile(fid, '>i2', 1),
        hour=np.fromfile(fid, '>i2', 1),
        minute=np.fromfile(fid, '>i2', 1),
        second=np.fromfile(fid, '>i2', 1),
        millisecond=np.fromfile(fid, '>i4', 1),
        samp_rate=np.fromfile(fid, '>i2', 1),
        n_channels=np.fromfile(fid, '>i2', 1),
        gain=np.fromfile(fid, '>i2', 1),
        bits=np.fromfile(fid, '>i2', 1),
        value_range=np.fromfile(fid, '>i2', 1)
    )

    unsegmented = 1 if np.bitwise_and(version, 1) == 0 else 1
    precision = np.bitwise_and(version, 6)
    if precision == 0:
        RuntimeError('File precision is not defined.')

    if unsegmented:
        info.update(dict(n_categories=0,
                         n_segments=1,
                         n_samples=np.fromfile(fid, '>i4', 1)[0],
                         n_events=np.fromfile(fid, '>i2', 1)[0],
                         event_codes=[],
                         category_names=[],
                         category_lengths =[],
                         pre_baseline=0))
        for event in range(info['n_events']):
            info['event_codes'].append(''.join(np.fromfile(fid, '>c2', 4)))
        info['event_codes'] = np.array(info['event_codes'])
    else:
        raise NotImplementedError('Only continous files are supported')
    info.update(dict(precision=precision, unsegmented=unsegmented))
    events = _read_events(fid, info)
    fid.close()
    return info, events


def _read_events(fid, info):
    """Read events"""
    unpack = [info[k] for k in ['n_events', 'n_segments', 'n_channels']]
    n_events, n_segments, n_channels = unpack
    n_samples = 1 if info['unsegmented'] else info['n_samples']
    events = np.zeros([n_events, n_segments * info['n_samples']])
    dtype, bytesize = {2: ('>i2', 2), 4: ('>f4', 4),
                       6: ('>f8', 8)}[info['precision']]

    beg_dat = fid.tell()

    for ii in range(info['n_events']):
        fid.seek(beg_dat + (int(n_channels) + ii) * bytesize, 0)
        events[ii] = np.fromfile(fid, dtype, n_samples)
        fid.seek(int((n_channels + n_events) * bytesize), 1)
    return events


def _read_data():
    pass


def read_raw_egi(fname):
    pass


class _RawEDF(Raw):
    """Raw object from EGI simple binary file

    Parameters
    ----------
    input_fname : str
        Path to the raw file.

    See Also
    --------
    mne.fiff.Raw : Documentation of attribute and methods.
    """
    def __init__(self, input_fname):
        """docstring for __init__"""
        logger.info('Reading EGI header from %s...' % input_fname)
        pass