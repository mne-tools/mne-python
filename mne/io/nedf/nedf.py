# -*- coding: utf-8 -*-
"""Import NeuroElectrics DataFormat (NEDF) files."""

from copy import deepcopy
from datetime import datetime, timezone
from xml.etree import ElementTree

import numpy as np

from ..base import BaseRaw
from ..meas_info import create_info
from ..utils import _mult_cal_one
from ...utils import warn, verbose, _check_fname


def _getsubnodetext(node, name):
    """Get an element from an XML node, raise an error otherwise.

    Parameters
    ----------
    node: Element
        XML Element
    name: str
        Child element name

    Returns
    -------
    test: str
        Text contents of the child nodes
    """
    subnode = node.findtext(name)
    if not subnode:
        raise RuntimeError('NEDF header ' + name + ' not found')
    return subnode


def _parse_nedf_header(header):
    """Read header information from the first 10kB of an .nedf file.

    Parameters
    ----------
    header : bytes
        Null-terminated header data, mostly the file's first 10240 bytes.

    Returns
    -------
    info : dict
        A dictionary with header information.
    dt : numpy.dtype
        Structure of the binary EEG/accelerometer/trigger data in the file.
    n_samples : int
        The number of data samples.
    """
    info = {}
    # nedf files have three accelerometer channels sampled at 100Hz followed
    # by five EEG samples + TTL trigger sampled at 500Hz
    # For 32 EEG channels and no stim channels, the data layout may look like
    # [ ('acc', '>u2', (3,)),
    #   ('data', dtype([
    #       ('eeg', 'u1', (32, 3)),
    #       ('trig', '>i4', (1,))
    #   ]), (5,))
    # ]

    dt = []  # dtype for the binary data block
    datadt = []  # dtype for a single EEG sample

    headerend = header.find(b'\0')
    if headerend == -1:
        raise RuntimeError('End of header null not found')
    headerxml = ElementTree.fromstring(header[:headerend])
    nedfversion = headerxml.findtext('NEDFversion', '')
    if nedfversion not in ['1.3', '1.4']:
        warn('NEDFversion unsupported, use with caution')

    if headerxml.findtext('stepDetails/DeviceClass', '') == 'STARSTIM':
        warn('Found Starstim, this hasn\'t been tested extensively!')

    if headerxml.findtext('AdditionalChannelStatus', 'OFF') != 'OFF':
        raise RuntimeError('Unknown additional channel, aborting.')

    n_acc = int(headerxml.findtext('NumberOfChannelsOfAccelerometer', 0))
    if n_acc:
        # expect one sample of u16 accelerometer data per block
        dt.append(('acc', '>u2', (n_acc,)))

    eegset = headerxml.find('EEGSettings')
    if eegset is None:
        raise RuntimeError('No EEG channels found')
    nchantotal = int(_getsubnodetext(eegset, 'TotalNumberOfChannels'))
    info['nchan'] = nchantotal

    info['sfreq'] = int(_getsubnodetext(eegset, 'EEGSamplingRate'))
    info['ch_names'] = [e.text for e in eegset.find('EEGMontage')]
    if nchantotal != len(info['ch_names']):
        raise RuntimeError(
            f"TotalNumberOfChannels ({nchantotal}) != "
            f"channel count ({len(info['ch_names'])})")
    # expect nchantotal uint24s
    datadt.append(('eeg', 'B', (nchantotal, 3)))

    if headerxml.find('STIMSettings') is not None:
        # 2* -> two stim samples per eeg sample
        datadt.append(('stim', 'B', (2, nchantotal, 3)))
        warn('stim channels are currently ignored')

    # Trigger data: 4 bytes in newer versions, 1 byte in older versions
    trigger_type = '>i4' if headerxml.findtext('NEDFversion') else 'B'
    datadt.append(('trig', trigger_type))
    # 5 data samples per block
    dt.append(('data', np.dtype(datadt), (5,)))

    date = headerxml.findtext('StepDetails/StartDate_firstEEGTimestamp', 0)
    info['meas_date'] = datetime.fromtimestamp(int(date) / 1000, timezone.utc)

    n_samples = int(_getsubnodetext(eegset, 'NumberOfRecordsOfEEG'))
    n_full, n_last = divmod(n_samples, 5)
    dt_last = deepcopy(dt)
    assert dt_last[-1][-1] == (5,)
    dt_last[-1] = list(dt_last[-1])
    dt_last[-1][-1] = (n_last,)
    dt_last[-1] = tuple(dt_last[-1])
    return info, np.dtype(dt), np.dtype(dt_last), n_samples, n_full


# the first 10240 bytes are header in XML format, padded with NULL bytes
_HDRLEN = 10240


class RawNedf(BaseRaw):
    """Raw object from NeuroElectrics nedf file."""

    def __init__(self, filename, preload=False, verbose=None):
        filename = _check_fname(filename, 'read', True, 'filename')
        with open(filename, mode='rb') as fid:
            header = fid.read(_HDRLEN)
        header, dt, dt_last, n_samp, n_full = _parse_nedf_header(header)
        ch_names = header['ch_names'] + ['STI 014']
        ch_types = ['eeg'] * len(ch_names)
        ch_types[-1] = 'stim'
        info = create_info(ch_names, header['sfreq'], ch_types)
        # scaling factor ADC-values -> volts
        # taken from the NEDF EEGLAB plugin
        # (https://www.neuroelectrics.com/resources/software/):
        for ch in info['chs'][:-1]:
            ch['cal'] = 2.4 / (6.0 * 8388607)
        with info._unlock():
            info['meas_date'] = header['meas_date']
        raw_extra = dict(dt=dt, dt_last=dt_last, n_full=n_full)
        super().__init__(
            info, preload=preload, filenames=[filename], verbose=verbose,
            raw_extras=[raw_extra], last_samps=[n_samp - 1])

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        dt = self._raw_extras[fi]['dt']
        dt_last = self._raw_extras[fi]['dt_last']
        n_full = self._raw_extras[fi]['n_full']
        n_eeg = dt[1].subdtype[0][0].shape[0]
        # data is stored in 5-sample chunks (except maybe the last one!)
        # so we have to do some gymnastics to pick the correct parts to
        # read
        offset = start // 5 * dt.itemsize + _HDRLEN
        start_sl = start % 5
        n_samples = stop - start
        n_samples_full = min(stop, n_full * 5) - start
        last = None
        n_chunks = (n_samples_full - 1) // 5 + 1
        n_tot = n_chunks * 5
        with open(self._filenames[fi], 'rb') as fid:
            fid.seek(offset, 0)
            chunks = np.fromfile(fid, dtype=dt, count=n_chunks)
            assert len(chunks) == n_chunks
            if n_samples != n_samples_full:
                last = np.fromfile(fid, dtype=dt_last, count=1)
        eeg = _convert_eeg(chunks, n_eeg, n_tot)
        trig = chunks['data']['trig'].reshape(1, n_tot)
        if last is not None:
            n_last = dt_last['data'].shape[0]
            eeg = np.concatenate(
                (eeg, _convert_eeg(last, n_eeg, n_last)), axis=-1)
            trig = np.concatenate(
                (trig, last['data']['trig'].reshape(1, n_last)), axis=-1)
        one_ = np.concatenate((eeg, trig))
        one = one_[:, start_sl:n_samples + start_sl]
        _mult_cal_one(data, one, idx, cals, mult)


def _convert_eeg(chunks, n_eeg, n_tot):
    # convert uint8-triplet -> int32
    eeg = chunks['data']['eeg'] @ np.array([1 << 16, 1 << 8, 1])
    # convert sign if necessary
    eeg[eeg > (1 << 23)] -= 1 << 24
    eeg = eeg.reshape((n_tot, n_eeg)).T
    return eeg


@verbose
def read_raw_nedf(filename, preload=False, verbose=None):
    """Read NeuroElectrics .nedf files.

    NEDF file versions starting from 1.3 are supported.

    Parameters
    ----------
    filename : str
        Path to the .nedf file.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawNedf
        A Raw object containing NEDF data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawNedf(filename, preload, verbose)
