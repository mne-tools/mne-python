# Authors: Federico Raimondo <federaimondo@gmail.com>
#
# License: BSD (3-clause)

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ...utils import fill_doc, logger, verbose, warn
from ..base import BaseRaw
from ..meas_info import create_info
from ...annotations import Annotations


@fill_doc
def read_raw_nihon(fname, preload=False, verbose=None):
    """Reader for an Nihon Kohden EEG file.

    Parameters
    ----------
    fname : str
        Path to the Nihon Kohden data file (.eeg).
    preload : bool
        If True, all data are loaded at initialization.
    %(verbose)s

    Returns
    -------
    raw : instance of RawNihon
        A Raw object containing Nihon Kohden data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawNihon(fname, preload, verbose)


_valid_headers = [
    'EEG-1100A V01.00',
    'EEG-1100B V01.00',
    'EEG-1100C V01.00',
    'QI-403A   V01.00',
    'QI-403A   V02.00',
    'EEG-2100  V01.00',
    'EEG-2100  V02.00',
    'DAE-2100D V01.30',
    'DAE-2100D V02.00',
]


def _read_nihon_metadata(fname):
    metadata = {}
    if not isinstance(fname, Path):
        fname = Path(fname)
    pnt_fname = fname.with_suffix('.PNT')
    if not pnt_fname.exists():
        warn('No PNT file exists. Metadata will be blank')
        return metadata
    logger.info('Found PNT file, reading metadata.')
    with open(pnt_fname, 'r') as fid:
        version = np.fromfile(fid, '|S16', 1).astype('U16')[0]
        if version not in _valid_headers:
            raise ValueError(
                'Not a valid Nihon Kohden PNT file ({})'.format(version))
        metadata['version'] = version

        # Read timestamp
        fid.seek(0x40)
        meas_str = np.fromfile(fid, '|S14', 1).astype('U14')[0]
        meas_date = datetime.strptime(meas_str, '%Y%m%d%H%M%S')
        meas_date = meas_date.replace(tzinfo=timezone.utc)
        metadata['meas_date'] = meas_date

    return metadata


_default_chan_labels = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
    'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ', 'E', 'PG1', 'PG2', 'A1', 'A2',
    'T1', 'T2'
]
_default_chan_labels += ['X{}'.format(i) for i in range(1, 12)]
_default_chan_labels += ['NA{}'.format(i) for i in range(1, 6)]
_default_chan_labels += ['DC{:02}'.format(i) for i in range(1, 33)]
_default_chan_labels += ['BN1', 'BN2', 'Mark1', 'Mark2']
_default_chan_labels += ['NA{}'.format(i) for i in range(6, 28)]
_default_chan_labels += ['X12/BP1', 'X13/BP2', 'X14/BP3', 'X15/BP4']
_default_chan_labels += ['X{}'.format(i) for i in range(16, 166)]
_default_chan_labels += ['NA28', 'Z']


def _read_21e_file(fname):
    if not isinstance(fname, Path):
        fname = Path(fname)
    e_fname = fname.with_suffix('.21E')
    _chan_labels = [x for x in _default_chan_labels]
    if e_fname.exists():
        # Read the 21E file and update the labels accordingly.
        logger.info('Found 21E file, reading channel names.')
        with open(e_fname, 'r') as fid:
            keep_parsing = False
            for line in fid:
                if line.startswith('['):
                    if 'ELECTRODE' in line or 'REFERENCE' in line:
                        keep_parsing = True
                    else:
                        keep_parsing = False
                elif keep_parsing is True:
                    idx, name = line.split('=')
                    idx = int(idx)
                    _chan_labels[idx] = name.strip()
    return _chan_labels


def _read_nihon_header(fname):
    # Read the Nihon Kohden EEG file header
    if not isinstance(fname, Path):
        fname = Path(fname)
    _chan_labels = _read_21e_file(fname)
    header = {}
    logger.info(f'Reading header from {fname}')
    with open(fname, 'r') as fid:
        version = np.fromfile(fid, '|S16', 1).astype('U16')[0]
        if version not in _valid_headers:
            raise ValueError(
                'Not a valid Nihon Kohden EEG file ({})'.format(version))

        fid.seek(0x0081)
        control_block = np.fromfile(fid, '|S16', 1).astype('U16')[0]
        if control_block not in _valid_headers:
            raise ValueError('Not a valid Nihon Kohden EEG file '
                             '(control block {})'.format(version))

        fid.seek(0x17fe)
        waveform_sign = np.fromfile(fid, np.uint8, 1)[0]
        if waveform_sign != 1:
            raise ValueError('Not a valid Nihon Kohden EEG file '
                             '(waveform block)')
        header['version'] = version

        fid.seek(0x0091)
        n_ctlblocks = np.fromfile(fid, np.uint8, 1)[0]
        header['n_ctlblocks'] = n_ctlblocks
        controlblocks = []
        for i_ctl_block in range(n_ctlblocks):
            t_controlblock = {}
            fid.seek(0x0092 + i_ctl_block * 20)
            t_ctl_address = np.fromfile(fid, np.uint32, 1)[0]
            t_controlblock['address'] = t_ctl_address
            fid.seek(t_ctl_address + 17)
            n_datablocks = np.fromfile(fid, np.uint8, 1)[0]
            t_controlblock['n_datablocks'] = n_datablocks
            t_controlblock['datablocks'] = []
            for i_data_block in range(n_datablocks):
                t_datablock = {}
                fid.seek(t_ctl_address + i_data_block * 20 + 18)
                t_data_address = np.fromfile(fid, np.uint32, 1)[0]
                t_datablock['address'] = t_data_address

                fid.seek(t_data_address + 0x26)
                t_n_channels = np.fromfile(fid, np.uint8, 1)[0]
                t_datablock['n_channels'] = t_n_channels

                t_channels = []
                for i_ch in range(t_n_channels):
                    fid.seek(t_data_address + 0x27 + (i_ch * 10))
                    t_idx = np.fromfile(fid, np.uint8, 1)[0]
                    t_channels.append(_chan_labels[t_idx])

                t_datablock['channels'] = t_channels

                fid.seek(t_data_address + 0x1C)
                t_record_duration = np.fromfile(fid, np.uint32, 1)[0]
                t_datablock['duration'] = t_record_duration

                fid.seek(t_data_address + 0x1a)
                sfreq = np.fromfile(fid, np.uint16, 1)[0] & 0x3FFF
                t_datablock['sfreq'] = sfreq

                t_datablock['n_samples'] = int(t_record_duration * sfreq / 10)
                t_controlblock['datablocks'].append(t_datablock)
            controlblocks.append(t_controlblock)
        header['controlblocks'] = controlblocks

    # Now check that every data block has the same channels and sfreq
    chans = []
    sfreqs = []
    nsamples = []
    for t_ctl in header['controlblocks']:
        for t_dtb in t_ctl['datablocks']:
            chans.append(t_dtb['channels'])
            sfreqs.append(t_dtb['sfreq'])
            nsamples.append(t_dtb['n_samples'])
    for i_elem in range(1, len(chans)):
        if chans[0] != chans[i_elem]:
            raise ValueError('Channel names in datablocks do not match')
        if sfreqs[0] != sfreqs[i_elem]:
            raise ValueError('Sample frequency in datablocks do not match')
    header['ch_names'] = chans[0]
    header['sfreq'] = sfreqs[0]
    header['n_samples'] = np.sum(nsamples)

    # TODO: Support more than one controlblock and more than one datablock
    if header['n_ctlblocks'] != 1:
        raise NotImplementedError('I dont know how to read more than one '
                                  'control block for this type of file :(')
    if header['controlblocks'][0]['n_datablocks'] != 1:
        raise NotImplementedError('I dont know how to read more than one '
                                  'data block for this type of file :(')

    return header


def _read_nihon_events(fname, orig_time):
    if not isinstance(fname, Path):
        fname = Path(fname)
    annotations = None
    log_fname = fname.with_suffix('.LOG')
    if not log_fname.exists():
        warn('No LOG file exists. Annotations will not be read')
        return annotations
    logger.info('Found LOG file, reading events.')
    with open(log_fname, 'r') as fid:
        version = np.fromfile(fid, '|S16', 1).astype('U16')[0]
        if version not in _valid_headers:
            raise ValueError(
                'Not a valid Nihon Kohden LOG file ({})'.format(version))

        fid.seek(0x91)
        n_logblocks = np.fromfile(fid, np.uint8, 1)[0]
        all_onsets = []
        all_descriptions = []
        for t_block in range(n_logblocks):
            fid.seek(0x92 + t_block * 20)
            t_blk_address = np.fromfile(fid, np.uint32, 1)[0]
            fid.seek(t_blk_address + 0x12)
            n_logs = np.fromfile(fid, np.uint8, 1)[0]
            fid.seek(t_blk_address + 0x14)
            t_logs = np.fromfile(fid, '|S45', n_logs).astype('U45')

            for t_log in t_logs:
                t_desc =  t_log[:20].strip('\x00')
                t_onset = datetime.strptime(t_log[20:26], '%H%M%S')
                t_onset = (t_onset.hour * 3600 + t_onset.minute * 60 +
                           t_onset.second)
                all_onsets.append(t_onset)
                all_descriptions.append(t_desc)

        annotations = Annotations(all_onsets, 0.0, all_descriptions, orig_time)
    return annotations


def _map_ch_to_type(ch_name):
    ch_type = 'eeg'
    if 'Mark' in ch_name:
        ch_type = 'stim'
    elif 'DC' in ch_name:
        ch_type = 'misc'
    elif 'NA' in ch_name:
        ch_type = 'misc'
    elif 'X' in ch_name:
        ch_type = 'bio'
    elif 'Z' in ch_name:
        ch_type = 'misc'
    elif 'STI 014' in ch_name:
        ch_type = 'stim'
    return ch_type


def _map_ch_to_specs(ch_name):
    if ch_name == 'STI 014':
        unit_mult = 1
        phys_min = 0
        phys_max = 1
    else:
        unit_mult = 1e-3
        phys_min = -12002.9
        phys_max = 12002.56
        if ch_name.upper() in _default_chan_labels:
            idx = _default_chan_labels.index(ch_name.upper())
            if (idx < 42 or idx > 73) and idx not in [76, 77]:
                unit_mult = 1e-6
                phys_min = -3200
                phys_max = 3199.902
    return dict(unit=unit_mult, phys_min=phys_min, phys_max=phys_max)


@fill_doc
class RawNihon(BaseRaw):
    """Raw object from a Nihon Kohden EEG file.

    Parameters
    ----------
    fname : str
        Path to the Nihon Kohden data file (.eeg).
    preload : bool
        If True, all data are loaded at initialization.
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None):
        if not isinstance(fname, Path):
            fname = Path(fname)
        data_name = fname.name
        logger.info('Loading %s' % data_name)

        header = _read_nihon_header(fname)
        metadata = _read_nihon_metadata(fname)

        # n_chan = len(header['ch_names']) + 1
        sfreq = header['sfreq']
        # data are multiplexed int16
        ch_names = header['ch_names']
        ch_types = [_map_ch_to_type(x) for x in ch_names]

        info = create_info(ch_names, sfreq, ch_types)
        n_samples = header['n_samples']

        if 'meas_date' in metadata:
            info['meas_date'] = metadata['meas_date']
        ch_extras = {x: _map_ch_to_specs(x) for x in ch_names}

        raw_extras = {
            'chs': ch_extras, 'header': header}
        self._header = header

        for i_ch, ch_name in enumerate(info['ch_names']):
            if ch_name == 'STI 014':
                continue
            t_range = (ch_extras[ch_name]['phys_max'] -
                       ch_extras[ch_name]['phys_min'])
            info['chs'][i_ch]['range'] = t_range
            info['chs'][i_ch]['cal'] = 1 / 65535.0

        super(RawNihon, self).__init__(
            info, preload=preload, last_samps=(n_samples - 1,),
            filenames=[fname.as_posix()], orig_format='short',
            raw_extras=[raw_extras])

        # Get annotations from LOG file
        annots = _read_nihon_events(fname, orig_time=info['meas_date'])
        self.set_annotations(annots)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        return _read_segment_file(data, idx, fi, start, stop,
                                  self._raw_extras[fi], cals,
                                  self._filenames[fi])


def _read_segment_file(data, idx, fi, start, stop, raw_extras, cals, filename):
    # For now we assume one control block and one data block.
    header = raw_extras['header']
    orig_ch_names = header['ch_names']
    chs = raw_extras['chs']

    gains = np.atleast_2d([chs[x]['unit'] for x in orig_ch_names])[:, idx].T

    physical_min = np.array([chs[x]['phys_min'] for x in orig_ch_names])[idx]
    digital_min = np.array([-32768 for x in orig_ch_names])[idx]

    offsets = np.atleast_2d(physical_min - (digital_min * cals[:, 0])).T

    datablock = header['controlblocks'][0]['datablocks'][0]
    n_channels = datablock['n_channels'] + 1
    datastart = datablock['address'] + 0x27 + (datablock['n_channels'] * 10)

    with open(filename, 'rb') as fid:
        start_offset = datastart + start * n_channels * 2
        to_read = (stop - start) * n_channels
        fid.seek(start_offset)
        block_data = np.fromfile(fid, '<u2', to_read) + 0x8000
        block_data = block_data.astype(np.int16)
        block_data = block_data.reshape(n_channels, -1, order='F')
        block_data = ((block_data[idx, :] * cals + offsets) * gains)
        data[:, :] = block_data[:, :]

    return None
