# Authors: Federico Raimondo <federaimondo@gmail.com>
#
# License: BSD-3-Clause

from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ...utils import fill_doc, logger, verbose, warn, _check_fname
from ..base import BaseRaw
from ..meas_info import create_info
from ...annotations import Annotations
from ..utils import _mult_cal_one


def _ensure_path(fname):
    out = fname
    if not isinstance(out, Path):
        out = Path(out)
    return out


@fill_doc
def read_raw_nihon(fname, preload=False, verbose=None):
    """Reader for an Nihon Kohden EEG file.

    Parameters
    ----------
    fname : str
        Path to the Nihon Kohden data file (``.EEG``).
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
    # 'EEG-1200A V01.00',  # Not working for the moment.
]


def _read_nihon_metadata(fname):
    metadata = {}
    fname = _ensure_path(fname)
    pnt_fname = fname.with_suffix('.PNT')
    if not pnt_fname.exists():
        warn('No PNT file exists. Metadata will be blank')
        return metadata
    logger.info('Found PNT file, reading metadata.')
    with open(pnt_fname, 'r') as fid:
        version = np.fromfile(fid, '|S16', 1).astype('U16')[0]
        if version not in _valid_headers:
            raise ValueError(f'Not a valid Nihon Kohden PNT file ({version})')
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
_default_chan_labels += [f'X{i}' for i in range(1, 12)]
_default_chan_labels += [f'NA{i}' for i in range(1, 6)]
_default_chan_labels += [f'DC{i:02}' for i in range(1, 33)]
_default_chan_labels += ['BN1', 'BN2', 'Mark1', 'Mark2']
_default_chan_labels += [f'NA{i}' for i in range(6, 28)]
_default_chan_labels += ['X12/BP1', 'X13/BP2', 'X14/BP3', 'X15/BP4']
_default_chan_labels += [f'X{i}' for i in range(16, 166)]
_default_chan_labels += ['NA28', 'Z']


def _read_21e_file(fname):
    fname = _ensure_path(fname)
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
                    if idx >= len(_chan_labels):
                        n = idx - len(_chan_labels) + 1
                        _chan_labels.extend(['UNK'] * n)
                    _chan_labels[idx] = name.strip()
    return _chan_labels


def _read_nihon_header(fname):
    # Read the Nihon Kohden EEG file header
    fname = _ensure_path(fname)
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
    if header['controlblocks'][0]['n_datablocks'] > 1:
        # Multiple blocks, check that they all have the same kind of data
        datablocks = header['controlblocks'][0]['datablocks']
        block_0 = datablocks[0]
        for t_block in datablocks[1:]:
            if block_0['n_channels'] != t_block['n_channels']:
                raise ValueError(
                    'Cannot read NK file with different number of channels '
                    'in each datablock')
            if block_0['channels'] != t_block['channels']:
                raise ValueError(
                    'Cannot read NK file with different channels in each '
                    'datablock')
            if block_0['sfreq'] != t_block['sfreq']:
                raise ValueError(
                    'Cannot read NK file with different sfreq in each '
                    'datablock')

    return header


def _read_nihon_annotations(fname):
    fname = _ensure_path(fname)
    log_fname = fname.with_suffix('.LOG')
    if not log_fname.exists():
        warn('No LOG file exists. Annotations will not be read')
        return dict(onset=[], duration=[], description=[])
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
        encodings = ('utf-8', 'latin1')
        for t_block in range(n_logblocks):
            fid.seek(0x92 + t_block * 20)
            t_blk_address = np.fromfile(fid, np.uint32, 1)[0]
            fid.seek(t_blk_address + 0x12)
            n_logs = np.fromfile(fid, np.uint8, 1)[0]
            fid.seek(t_blk_address + 0x14)
            t_logs = np.fromfile(fid, '|S45', n_logs)
            for t_log in t_logs:
                for enc in encodings:
                    try:
                        t_log = t_log.decode(enc)
                    except UnicodeDecodeError:
                        pass
                    else:
                        break
                else:
                    warn(f'Could not decode log as one of {encodings}')
                    continue
                t_desc = t_log[:20].strip('\x00')
                t_onset = datetime.strptime(t_log[20:26], '%H%M%S')
                t_onset = (t_onset.hour * 3600 + t_onset.minute * 60 +
                           t_onset.second)
                all_onsets.append(t_onset)
                all_descriptions.append(t_desc)

        annots = dict(
            onset=all_onsets,
            duration=[0] * len(all_onsets),
            description=all_descriptions)
    return annots


def _map_ch_to_type(ch_name):
    ch_type_pattern = OrderedDict([
        ('stim', ('Mark',)), ('misc', ('DC', 'NA', 'Z', '$')),
        ('bio', ('X',))])
    for key, kinds in ch_type_pattern.items():
        if any(kind in ch_name for kind in kinds):
            return key
    return 'eeg'


def _map_ch_to_specs(ch_name):
    unit_mult = 1e-3
    phys_min = -12002.9
    phys_max = 12002.56
    dig_min = -32768
    if ch_name.upper() in _default_chan_labels:
        idx = _default_chan_labels.index(ch_name.upper())
        if (idx < 42 or idx > 73) and idx not in [76, 77]:
            unit_mult = 1e-6
            phys_min = -3200
            phys_max = 3199.902
    t_range = phys_max - phys_min
    cal = t_range / 65535
    offset = phys_min - (dig_min * cal)

    out = dict(unit=unit_mult, phys_min=phys_min, phys_max=phys_max,
               dig_min=dig_min, cal=cal, offset=offset)
    return out


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
        fname = _check_fname(fname, 'read', True, 'fname')
        fname = _ensure_path(fname)
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
            with info._unlock():
                info['meas_date'] = metadata['meas_date']
        chs = {x: _map_ch_to_specs(x) for x in ch_names}

        orig_ch_names = header['ch_names']
        cal = np.array(
            [chs[x]['cal'] for x in orig_ch_names], float)[:, np.newaxis]
        offsets = np.array(
            [chs[x]['offset'] for x in orig_ch_names], float)[:, np.newaxis]
        gains = np.array(
            [chs[x]['unit'] for x in orig_ch_names], float)[:, np.newaxis]

        raw_extras = dict(
            cal=cal, offsets=offsets, gains=gains, header=header)
        self._header = header

        for i_ch, ch_name in enumerate(info['ch_names']):
            t_range = (chs[ch_name]['phys_max'] - chs[ch_name]['phys_min'])
            info['chs'][i_ch]['range'] = t_range
            info['chs'][i_ch]['cal'] = 1 / t_range

        super(RawNihon, self).__init__(
            info, preload=preload, last_samps=(n_samples - 1,),
            filenames=[fname.as_posix()], orig_format='short',
            raw_extras=[raw_extras])

        # Get annotations from LOG file
        annots = _read_nihon_annotations(fname)

        # Annotate acqusition skips
        controlblock = self._header['controlblocks'][0]
        cur_sample = 0
        if controlblock['n_datablocks'] > 1:
            for i_block in range(controlblock['n_datablocks'] - 1):
                t_block = controlblock['datablocks'][i_block]
                cur_sample = cur_sample + t_block['n_samples']
                cur_tpoint = (cur_sample - 0.5) / t_block['sfreq']
                # Add annotations as in append raw
                annots['onset'].append(cur_tpoint)
                annots['duration'].append(0.0)
                annots['description'].append('BAD boundary')
                annots['onset'].append(cur_tpoint)
                annots['duration'].append(0.0)
                annots['description'].append('EDGE boundary')

        annotations = Annotations(**annots, orig_time=info['meas_date'])
        self.set_annotations(annotations)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        # For now we assume one control block
        header = self._raw_extras[fi]['header']

        # Get the original cal, offsets and gains
        cal = self._raw_extras[fi]['cal']
        offsets = self._raw_extras[fi]['offsets']
        gains = self._raw_extras[fi]['gains']

        # get the right datablock
        datablocks = header['controlblocks'][0]['datablocks']
        ends = np.cumsum([t['n_samples'] for t in datablocks])

        start_block = np.where(start < ends)[0][0]
        stop_block = np.where(stop <= ends)[0][0]

        if start_block != stop_block:
            # Recursive call for each block independently
            new_start = start
            sample_start = 0
            for t_block_idx in range(start_block, stop_block + 1):
                t_block = datablocks[t_block_idx]
                if t_block == stop_block:
                    # If its the last block, we stop on the last sample to read
                    new_stop = stop
                else:
                    # Otherwise, stop on the last sample of the block
                    new_stop = t_block['n_samples'] + new_start
                samples_to_read = new_stop - new_start
                sample_stop = sample_start + samples_to_read

                self._read_segment_file(
                    data[:, sample_start:sample_stop], idx, fi,
                    new_start, new_stop, cals, mult
                )

                # Update variables for next loop
                sample_start = sample_stop
                new_start = new_stop
        else:
            datablock = datablocks[start_block]

            n_channels = datablock['n_channels'] + 1
            datastart = (datablock['address'] + 0x27 +
                         (datablock['n_channels'] * 10))

            # Compute start offset based on the beginning of the block
            rel_start = start
            if start_block != 0:
                rel_start = start - ends[start_block - 1]
            start_offset = datastart + rel_start * n_channels * 2

            with open(self._filenames[fi], 'rb') as fid:
                to_read = (stop - start) * n_channels
                fid.seek(start_offset)
                block_data = np.fromfile(fid, '<u2', to_read) + 0x8000
                block_data = block_data.astype(np.int16)
                block_data = block_data.reshape(n_channels, -1, order='F')
                block_data = block_data[:-1] * cal  # cast to float64
                block_data += offsets
                block_data *= gains
                _mult_cal_one(data, block_data, idx, cals, mult)
