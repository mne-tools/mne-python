# -*- coding: utf-8 -*-
"""Import NeuroElectrics DataFormat (NEDF) files."""

from datetime import datetime, timezone
from xml.etree import ElementTree

import numpy as np

from .. import BaseRaw
from ... import create_info, Annotations
from ...utils import warn


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
        warn('Unexpected NEDFversion, hope this works anyway')

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
        raise RuntimeError("TotalNumberOfChannels != channel count")
    # expect nchantotal uint24s
    datadt.append(('eeg', 'B', (nchantotal, 3)))

    if headerxml.find('STIMSettings') is not None:
        # 2* -> two stim samples per eeg sample
        datadt.append(('stim', 'B', (2, nchantotal, 3)))

    # Trigger data: 4 bytes in newer versions, 1 byte in older versions
    trigger_type = '>i4' if headerxml.findtext('NEDFversion') else 'B'
    datadt.append(('trig', trigger_type))
    # 5 data samples per block
    dt.append(('data', np.dtype(datadt), (5,)))

    date = headerxml.findtext('StepDetails/StartDate_firstEEGTimestamp', 0)
    info['meas_date'] = datetime.fromtimestamp(int(date) / 1000, timezone.utc)
    return info, np.dtype(dt)


def _read_nedf_eeg(filename: str):
    """Read header info and EEG data from an .nedf file.

    Parameters
    ----------
    filename : str
        Path to the .nedf file.

    Returns
    -------
    eeg : array, shape (n_samples, n_channels)
        Unscaled EEG data.
    info : dict
        Information from the file header.
    triggers : array, shape (n_annots, 2)
        Start samples and values of each trigger.
    scale : float
        Scaling factor for the EEG data.
    """
    # the first 10240 bytes are header in XML format, padded with NULL bytes
    hdrlen = 10240
    with open(filename, mode='rb') as f:
        info, dt = _parse_nedf_header(f.read(hdrlen))
        data = np.fromfile(f, dtype=dt)

    # convert uint8-triplet -> int32
    eeg = data['data']['eeg'] @ np.array([1 << 16, 1 << 8, 1], dtype='i4')
    # convert sign if necessary
    eeg[eeg > (1 << 23)] -= 1 << 24
    eeg = eeg.reshape((-1, info['nchan']))

    triggers = data['data']['trig'].flatten()
    triggerind = triggers.nonzero()[0]
    triggers = np.stack((triggerind, triggers[triggerind])).T

    # scaling factor ADC-values -> volts
    # taken from the NEDF EEGLAB plugin
    # (https://www.neuroelectrics.com/resources/software/):
    scale = 2.4 / (6.0 * 8388607)

    return eeg, info, triggers, scale


class RawNedf(BaseRaw):
    """Raw object from NeuroElectrics nedf file."""

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        pass

    def __init__(self, nedffile):
        eeg, header, triggers, scale = _read_nedf_eeg(nedffile)
        info = create_info(ch_names=header['ch_names'],
                           sfreq=header['sfreq'],
                           ch_types='eeg')

        info['bads'] = []
        info['meas_date'] = header['meas_date']

        super().__init__(info, preload=(eeg * scale).T, filenames=nedffile)
        self.set_montage('standard_1020')

        onsets = triggers[:, 0] / info['sfreq']
        self.set_annotations(Annotations(onsets, 0, triggers[:, 1]))


def read_raw_nedf(filename):
    """Read NeuroElectrics .nedf files.

    NEDF file versions starting from 1.3 are supported.

    Parameters
    ----------
    filename : str
        Path to the .nedf file.

    Returns
    -------
    raw : instance of RawNedf
        A Raw object containing NEDF data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawNedf(filename)
