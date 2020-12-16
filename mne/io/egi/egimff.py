"""EGI NetStation Load Function."""

import datetime
import math
import os.path as op
import re
import time
from xml.dom.minidom import parse

import numpy as np

from .events import _read_events, _combine_triggers
from .general import (_get_signalfname, _get_ep_info, _extract, _get_blocks,
                      _get_gains, _block_r)
from ..base import BaseRaw
from ..constants import FIFF
from ..meas_info import _empty_info, create_info
from ..proj import setup_proj
from ..utils import _create_chs, _mult_cal_one
from ...annotations import Annotations
from ...utils import verbose, logger, warn, _check_option
from ...evoked import EvokedArray


def _read_mff_header(filepath):
    """Read mff header."""
    all_files = _get_signalfname(filepath)
    eeg_file = all_files['EEG']['signal']
    eeg_info_file = all_files['EEG']['info']

    info_filepath = op.join(filepath, 'info.xml')  # add with filepath
    tags = ['mffVersion', 'recordTime']
    version_and_date = _extract(tags, filepath=info_filepath)
    version = ""
    if len(version_and_date['mffVersion']):
        version = version_and_date['mffVersion'][0]

    fname = op.join(filepath, eeg_file)
    signal_blocks = _get_blocks(fname)
    epochs = _get_ep_info(filepath)
    summaryinfo = dict(eeg_fname=eeg_file,
                       info_fname=eeg_info_file)
    summaryinfo.update(signal_blocks)
    # sanity check and update relevant values
    record_time = version_and_date['recordTime'][0]
    # e.g.,
    # 2018-07-30T10:47:01.021673-04:00
    # 2017-09-20T09:55:44.072000000+01:00
    g = re.match(
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.(\d{6}(?:\d{3})?)[+-]\d{2}:\d{2}',  # noqa: E501
        record_time)
    if g is None:
        raise RuntimeError('Could not parse recordTime %r' % (record_time,))
    frac = g.groups()[0]
    assert len(frac) in (6, 9) and all(f.isnumeric() for f in frac)  # regex
    div = 1000 if len(frac) == 6 else 1000000
    for key in ('last_samps', 'first_samps'):
        # convert from times in µS to samples
        for ei, e in enumerate(epochs[key]):
            if e % div != 0:
                raise RuntimeError('Could not parse epoch time %s' % (e,))
            epochs[key][ei] = e // div
        epochs[key] = np.array(epochs[key], np.uint64)
        # I guess they refer to times in milliseconds?
        # What we really need to do here is:
        # epochs[key] *= signal_blocks['sfreq']
        # epochs[key] //= 1000
        # But that multiplication risks an overflow, so let's only multiply
        # by what we need to (e.g., a sample rate of 500 means we can multiply
        # by 1 and divide by 2 rather than multiplying by 500 and dividing by
        # 1000)
        numerator = signal_blocks['sfreq']
        denominator = 1000
        this_gcd = math.gcd(numerator, denominator)
        numerator = numerator // this_gcd
        denominator = denominator // this_gcd
        with np.errstate(over='raise'):
            epochs[key] *= numerator
        epochs[key] //= denominator
        # Should be safe to cast to int now, which makes things later not
        # upbroadcast to float
        epochs[key] = epochs[key].astype(np.int64)
    n_samps_block = signal_blocks['samples_block'].sum()
    n_samps_epochs = (epochs['last_samps'] - epochs['first_samps']).sum()
    bad = (n_samps_epochs != n_samps_block or
           not (epochs['first_samps'] < epochs['last_samps']).all() or
           not (epochs['first_samps'][1:] >= epochs['last_samps'][:-1]).all())
    if bad:
        raise RuntimeError('EGI epoch first/last samps could not be parsed:\n'
                           '%s\n%s' % (list(epochs['first_samps']),
                                       list(epochs['last_samps'])))
    summaryinfo.update(epochs)
    # index which samples in raw are actually readable from disk (i.e., not
    # in a skip)
    disk_samps = np.full(epochs['last_samps'][-1], -1)
    offset = 0
    for first, last in zip(epochs['first_samps'], epochs['last_samps']):
        n_this = last - first
        disk_samps[first:last] = np.arange(offset, offset + n_this)
        offset += n_this
    summaryinfo['disk_samps'] = disk_samps

    # Add the sensor info.
    sensor_layout_file = op.join(filepath, 'sensorLayout.xml')
    sensor_layout_obj = parse(sensor_layout_file)
    sensors = sensor_layout_obj.getElementsByTagName('sensor')
    chan_type = list()
    chan_unit = list()
    n_chans = 0
    numbers = list()  # used for identification
    for sensor in sensors:
        sensortype = int(sensor.getElementsByTagName('type')[0]
                         .firstChild.data)
        if sensortype in [0, 1]:
            sn = sensor.getElementsByTagName('number')[0].firstChild.data
            sn = sn.encode()
            numbers.append(sn)
            chan_type.append('eeg')
            chan_unit.append('uV')
            n_chans = n_chans + 1
    if n_chans != summaryinfo['n_channels']:
        raise RuntimeError('Number of defined channels (%d) did not match the '
                           'expected channels (%d)'
                           % (n_chans, summaryinfo['n_channels']))

    # Check presence of PNS data
    pns_names = []
    if 'PNS' in all_files:
        pns_fpath = op.join(filepath, all_files['PNS']['signal'])
        pns_blocks = _get_blocks(pns_fpath)
        pns_samples = pns_blocks['samples_block']
        signal_samples = signal_blocks['samples_block']
        same_blocks = (np.array_equal(pns_samples[:-1],
                                      signal_samples[:-1]) and
                       pns_samples[-1] in (signal_samples[-1] - np.arange(2)))
        if not same_blocks:
            raise RuntimeError('PNS and signals samples did not match:\n'
                               '%s\nvs\n%s'
                               % (list(pns_samples), list(signal_samples)))

        pns_file = op.join(filepath, 'pnsSet.xml')
        pns_obj = parse(pns_file)
        sensors = pns_obj.getElementsByTagName('sensor')
        pns_types = []
        pns_units = []
        for sensor in sensors:
            # sensor number:
            # sensor.getElementsByTagName('number')[0].firstChild.data
            name = sensor.getElementsByTagName('name')[0].firstChild.data
            unit_elem = sensor.getElementsByTagName('unit')[0].firstChild
            unit = ''
            if unit_elem is not None:
                unit = unit_elem.data

            if name == 'ECG':
                ch_type = 'ecg'
            elif 'EMG' in name:
                ch_type = 'emg'
            else:
                ch_type = 'bio'
            pns_types.append(ch_type)
            pns_units.append(unit)
            pns_names.append(name)

        summaryinfo.update(pns_types=pns_types, pns_units=pns_units,
                           pns_fname=all_files['PNS']['signal'],
                           pns_sample_blocks=pns_blocks)
    summaryinfo.update(pns_names=pns_names, version=version,
                       date=version_and_date['recordTime'][0],
                       chan_type=chan_type, chan_unit=chan_unit,
                       numbers=numbers)

    return summaryinfo


class _FixedOffset(datetime.tzinfo):
    """Fixed offset in minutes east from UTC.

    Adapted from the official Python documentation.
    """

    def __init__(self, offset):
        self._offset = datetime.timedelta(minutes=offset)

    def utcoffset(self, dt):
        return self._offset

    def tzname(self, dt):
        return 'MFF'

    def dst(self, dt):
        return datetime.timedelta(0)


def _read_header(input_fname):
    """Obtain the headers from the file package mff.

    Parameters
    ----------
    input_fname : str
        Path for the file

    Returns
    -------
    info : dict
        Main headers set.
    """
    mff_hdr = _read_mff_header(input_fname)
    with open(input_fname + '/signal1.bin', 'rb') as fid:
        version = np.fromfile(fid, np.int32, 1)[0]
    # This should be equivalent to the following, but no need for external dep:
    # import dateutil.parser
    # time_n = dateutil.parser.parse(mff_hdr['date'])
    dt = mff_hdr['date'][:26]
    assert mff_hdr['date'][-6] in ('+', '-')
    sn = -1 if mff_hdr['date'][-6] == '-' else 1  # +
    tz = [sn * int(t) for t in (mff_hdr['date'][-5:-3], mff_hdr['date'][-2:])]
    time_n = datetime.datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S.%f')
    time_n = time_n.replace(tzinfo=_FixedOffset(60 * tz[0] + tz[1]))
    info = dict(
        version=version,
        year=int(time_n.strftime('%Y')),
        month=int(time_n.strftime('%m')),
        day=int(time_n.strftime('%d')),
        hour=int(time_n.strftime('%H')),
        minute=int(time_n.strftime('%M')),
        second=int(time_n.strftime('%S')),
        millisecond=int(time_n.strftime('%f')),
        gain=0,
        bits=0,
        value_range=0)
    info.update(n_categories=0, n_segments=1, n_events=0, event_codes=[],
                category_names=[], category_lengths=[], pre_baseline=0)
    info.update(mff_hdr)
    return info


def _get_eeg_calibration_info(filepath, egi_info):
    """Calculate calibration info for EEG channels."""
    gains = _get_gains(op.join(filepath, egi_info['info_fname']))
    if egi_info['value_range'] != 0 and egi_info['bits'] != 0:
        cals = [egi_info['value_range'] / 2 ** egi_info['bits']] * \
            len(egi_info['chan_type'])
    else:
        cal_scales = {'uV': 1e-6, 'V': 1}
        cals = [cal_scales[t] for t in egi_info['chan_unit']]
    if 'gcal' in gains:
        cals *= gains['gcal']
    return cals


def _read_locs(filepath, chs, egi_info):
    """Read channel locations."""
    fname = op.join(filepath, 'coordinates.xml')
    if not op.exists(fname):
        return chs
    numbers = np.array(egi_info['numbers'])
    coordinates = parse(fname)
    sensors = coordinates.getElementsByTagName('sensor')
    for sensor in sensors:
        nr = sensor.getElementsByTagName('number')[0].firstChild.data.encode()
        id = np.where(numbers == nr)[0]
        if len(id) == 0:
            continue
        loc = chs[id[0]]['loc']
        loc[0] = sensor.getElementsByTagName('x')[0].firstChild.data
        loc[1] = sensor.getElementsByTagName('y')[0].firstChild.data
        loc[2] = sensor.getElementsByTagName('z')[0].firstChild.data
        loc /= 100.  # cm -> m
    return chs


def _add_pns_channel_info(chs, egi_info, ch_names):
    """Add info for PNS channels to channel info dict."""
    for i_ch, ch_name in enumerate(egi_info['pns_names']):
        idx = ch_names.index(ch_name)
        ch_type = egi_info['pns_types'][i_ch]
        type_to_kind_map = {'ecg': FIFF.FIFFV_ECG_CH,
                            'emg': FIFF.FIFFV_EMG_CH
                            }
        ch_kind = type_to_kind_map.get(ch_type, FIFF.FIFFV_BIO_CH)
        ch_unit = FIFF.FIFF_UNIT_V
        ch_cal = 1e-6
        if egi_info['pns_units'][i_ch] != 'uV':
            ch_unit = FIFF.FIFF_UNIT_NONE
            ch_cal = 1.0
        chs[idx].update(
            cal=ch_cal, kind=ch_kind, coil_type=FIFF.FIFFV_COIL_NONE,
            unit=ch_unit)
    return chs


@verbose
def _read_raw_egi_mff(input_fname, eog=None, misc=None,
                      include=None, exclude=None, preload=False,
                      channel_naming='E%d', verbose=None):
    """Read EGI mff binary as raw object.

    .. note:: This function attempts to create a synthetic trigger channel.
              See notes below.

    Parameters
    ----------
    input_fname : str
        Path to the raw file.
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. Default is None.
    misc : list or tuple
        Names of channels or list of indices that should be designated
        MISC channels. Default is None.
    include : None | list
       The event channels to be ignored when creating the synthetic
       trigger. Defaults to None.
       Note. Overrides `exclude` parameter.
    exclude : None | list
       The event channels to be ignored when creating the synthetic
       trigger. Defaults to None. If None, channels that have more than
       one event and the ``sync`` and ``TREV`` channels will be
       ignored.
    %(preload)s
    channel_naming : str
        Channel naming convention for the data channels. Defaults to 'E%%d'
        (resulting in channel names 'E1', 'E2', 'E3'...). The effective default
        prior to 0.14.0 was 'EEG %%03d'.
    %(verbose)s

    Returns
    -------
    raw : instance of RawMff
        A Raw object containing EGI mff data.

    Notes
    -----
    The trigger channel names are based on the arbitrary user dependent event
    codes used. However this function will attempt to generate a synthetic
    trigger channel named ``STI 014`` in accordance with the general
    Neuromag / MNE naming pattern.

    The event_id assignment equals ``np.arange(n_events) + 1``. The resulting
    ``event_id`` mapping is stored as attribute to the resulting raw object but
    will be ignored when saving to a fiff. Note. The trigger channel is
    artificially constructed based on timestamps received by the Netstation.
    As a consequence, triggers have only short durations.

    This step will fail if events are not mutually exclusive.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    .. versionadded:: 0.15.0
    """
    return RawMff(input_fname, eog, misc, include, exclude,
                  preload, channel_naming, verbose)


class RawMff(BaseRaw):
    """RawMff class."""

    @verbose
    def __init__(self, input_fname, eog=None, misc=None,
                 include=None, exclude=None, preload=False,
                 channel_naming='E%d', verbose=None):
        """Init the RawMff class."""
        logger.info('Reading EGI MFF Header from %s...' % input_fname)
        egi_info = _read_header(input_fname)
        if eog is None:
            eog = []
        if misc is None:
            misc = np.where(np.array(
                egi_info['chan_type']) != 'eeg')[0].tolist()

        logger.info('    Reading events ...')
        egi_events, egi_info = _read_events(input_fname, egi_info)
        cals = _get_eeg_calibration_info(input_fname, egi_info)
        logger.info('    Assembling measurement info ...')
        if egi_info['n_events'] > 0:
            event_codes = list(egi_info['event_codes'])
            if include is None:
                exclude_list = ['sync', 'TREV'] if exclude is None else exclude
                exclude_inds = [i for i, k in enumerate(event_codes) if k in
                                exclude_list]
                more_excludes = []
                if exclude is None:
                    for ii, event in enumerate(egi_events):
                        if event.sum() <= 1 and event_codes[ii]:
                            more_excludes.append(ii)
                if len(exclude_inds) + len(more_excludes) == len(event_codes):
                    warn('Did not find any event code with more than one '
                         'event.', RuntimeWarning)
                else:
                    exclude_inds.extend(more_excludes)

                exclude_inds.sort()
                include_ = [i for i in np.arange(egi_info['n_events']) if
                            i not in exclude_inds]
                include_names = [k for i, k in enumerate(event_codes)
                                 if i in include_]
            else:
                include_ = [i for i, k in enumerate(event_codes)
                            if k in include]
                include_names = include

            for kk, v in [('include', include_names), ('exclude', exclude)]:
                if isinstance(v, list):
                    for k in v:
                        if k not in event_codes:
                            raise ValueError('Could find event named "%s"' % k)
                elif v is not None:
                    raise ValueError('`%s` must be None or of type list' % kk)
            logger.info('    Synthesizing trigger channel "STI 014" ...')
            logger.info('    Excluding events {%s} ...' %
                        ", ".join([k for i, k in enumerate(event_codes)
                                   if i not in include_]))
            events_ids = np.arange(len(include_)) + 1
            egi_info['new_trigger'] = _combine_triggers(
                egi_events[include_], remapping=events_ids)
            self.event_id = dict(zip([e for e in event_codes if e in
                                      include_names], events_ids))
            if egi_info['new_trigger'] is not None:
                egi_events = np.vstack([egi_events, egi_info['new_trigger']])
            assert egi_events.shape[1] == egi_info['last_samps'][-1]
        else:
            # No events
            self.event_id = None
            egi_info['new_trigger'] = None
            event_codes = []
        info = _empty_info(egi_info['sfreq'])
        my_time = datetime.datetime(
            egi_info['year'], egi_info['month'], egi_info['day'],
            egi_info['hour'], egi_info['minute'], egi_info['second'])
        my_timestamp = time.mktime(my_time.timetuple())
        info['meas_date'] = (my_timestamp, 0)

        # First: EEG
        ch_names = [channel_naming % (i + 1) for i in
                    range(egi_info['n_channels'])]

        # Second: Stim
        ch_names.extend(list(egi_info['event_codes']))
        if egi_info['new_trigger'] is not None:
            ch_names.append('STI 014')  # channel for combined events
        cals = np.concatenate(
            [cals, np.repeat(1, len(event_codes) + 1 + len(misc) + len(eog))])

        # Third: PNS
        ch_names.extend(egi_info['pns_names'])
        cals = np.concatenate(
            [cals, np.repeat(1, len(egi_info['pns_names']))])

        # Actually create channels as EEG, then update stim and PNS
        ch_coil = FIFF.FIFFV_COIL_EEG
        ch_kind = FIFF.FIFFV_EEG_CH
        chs = _create_chs(ch_names, cals, ch_coil, ch_kind, eog, (), (), misc)
        chs = _read_locs(input_fname, chs, egi_info)
        sti_ch_idx = [i for i, name in enumerate(ch_names) if
                      name.startswith('STI') or name in event_codes]
        for idx in sti_ch_idx:
            chs[idx].update({'unit_mul': FIFF.FIFF_UNITM_NONE,
                             'cal': cals[idx],
                             'kind': FIFF.FIFFV_STIM_CH,
                             'coil_type': FIFF.FIFFV_COIL_NONE,
                             'unit': FIFF.FIFF_UNIT_NONE})
        chs = _add_pns_channel_info(chs, egi_info, ch_names)

        info['chs'] = chs
        info._update_redundant()
        file_bin = op.join(input_fname, egi_info['eeg_fname'])
        egi_info['egi_events'] = egi_events

        # Check how many channels to read are from EEG
        keys = ('eeg', 'sti', 'pns')
        idx = dict()
        idx['eeg'] = np.where(
            [ch['kind'] == FIFF.FIFFV_EEG_CH for ch in chs])[0]
        idx['sti'] = np.where(
            [ch['kind'] == FIFF.FIFFV_STIM_CH for ch in chs])[0]
        idx['pns'] = np.where(
            [ch['kind'] in (FIFF.FIFFV_ECG_CH, FIFF.FIFFV_EMG_CH,
                            FIFF.FIFFV_BIO_CH) for ch in chs])[0]
        # By construction this should always be true, but check anyway
        if not np.array_equal(
                np.concatenate([idx[key] for key in keys]),
                np.arange(len(chs))):
            raise ValueError('Currently interlacing EEG and PNS channels'
                             'is not supported')
        egi_info['kind_bounds'] = [0]
        for key in keys:
            egi_info['kind_bounds'].append(len(idx[key]))
        egi_info['kind_bounds'] = np.cumsum(egi_info['kind_bounds'])
        assert egi_info['kind_bounds'][0] == 0
        assert egi_info['kind_bounds'][-1] == info['nchan']
        first_samps = [0]
        last_samps = [egi_info['last_samps'][-1] - 1]

        annot = dict(onset=list(), duration=list(), description=list())
        if len(idx['pns']):
            # PNS Data is present and should be read:
            egi_info['pns_filepath'] = op.join(
                input_fname, egi_info['pns_fname'])
            # Check for PNS bug immediately
            pns_samples = np.sum(
                egi_info['pns_sample_blocks']['samples_block'])
            eeg_samples = np.sum(egi_info['samples_block'])
            if pns_samples == eeg_samples - 1:
                warn('This file has the EGI PSG sample bug')
                annot['onset'].append(last_samps[-1] / egi_info['sfreq'])
                annot['duration'].append(1 / egi_info['sfreq'])
                annot['description'].append('BAD_EGI_PSG')
            elif pns_samples != eeg_samples:
                raise RuntimeError(
                    'PNS samples (%d) did not match EEG samples (%d)'
                    % (pns_samples, eeg_samples))

        self._filenames = [file_bin]
        self._raw_extras = [egi_info]

        super(RawMff, self).__init__(
            info, preload=preload, orig_format='float', filenames=[file_bin],
            first_samps=first_samps, last_samps=last_samps,
            raw_extras=[egi_info], verbose=verbose)

        # Annotate acquisition skips
        for first, prev_last in zip(egi_info['first_samps'][1:],
                                    egi_info['last_samps'][:-1]):
            gap = first - prev_last
            assert gap >= 0
            if gap:
                annot['onset'].append((prev_last - 0.5) / egi_info['sfreq'])
                annot['duration'].append(gap / egi_info['sfreq'])
                annot['description'].append('BAD_ACQ_SKIP')

        if len(annot['onset']):
            self.set_annotations(Annotations(**annot, orig_time=None))

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of data."""
        dtype = '<f4'  # Data read in four byte floats.

        egi_info = self._raw_extras[fi]
        one = np.zeros((egi_info['kind_bounds'][-1], stop - start))

        # info about the binary file structure
        n_channels = egi_info['n_channels']
        samples_block = egi_info['samples_block']

        # Check how many channels to read are from each type
        bounds = egi_info['kind_bounds']
        if isinstance(idx, slice):
            idx = np.arange(idx.start, idx.stop)
        eeg_out = np.where(idx < bounds[1])[0]
        eeg_one = idx[eeg_out, np.newaxis]
        eeg_in = idx[eeg_out]
        stim_out = np.where((idx >= bounds[1]) & (idx < bounds[2]))[0]
        stim_one = idx[stim_out]
        stim_in = idx[stim_out] - bounds[1]
        pns_out = np.where((idx >= bounds[2]) & (idx < bounds[3]))[0]
        pns_in = idx[pns_out] - bounds[2]
        pns_one = idx[pns_out, np.newaxis]
        del eeg_out, stim_out, pns_out

        # take into account events (already extended to correct size)
        one[stim_one, :] = egi_info['egi_events'][stim_in, start:stop]

        # Convert start and stop to limits in terms of the data
        # actually on disk, plus an indexer (disk_use_idx) that populates
        # the potentially larger `data` with it, taking skips into account
        disk_samps = egi_info['disk_samps'][start:stop]
        disk_use_idx = np.where(disk_samps > -1)[0]
        if len(disk_use_idx):
            start = disk_samps[disk_use_idx[0]]
            stop = disk_samps[disk_use_idx[-1]] + 1
            assert len(disk_use_idx) == stop - start

        # Get starting/stopping block/samples
        block_samples_offset = np.cumsum(samples_block)
        offset_blocks = np.sum(block_samples_offset < start)
        offset_samples = start - (block_samples_offset[offset_blocks - 1]
                                  if offset_blocks > 0 else 0)

        samples_to_read = stop - start
        with open(self._filenames[fi], 'rb', buffering=0) as fid:
            # Go to starting block
            current_block = 0
            current_block_info = None
            current_data_sample = 0
            while current_block < offset_blocks:
                this_block_info = _block_r(fid)
                if this_block_info is not None:
                    current_block_info = this_block_info
                fid.seek(current_block_info['block_size'], 1)
                current_block = current_block + 1

            # Start reading samples
            while samples_to_read > 0:
                this_block_info = _block_r(fid)
                if this_block_info is not None:
                    current_block_info = this_block_info

                to_read = (current_block_info['nsamples'] *
                           current_block_info['nc'])
                block_data = np.fromfile(fid, dtype, to_read)
                block_data = block_data.reshape(n_channels, -1, order='C')

                # Compute indexes
                samples_read = block_data.shape[1]
                if offset_samples > 0:
                    # First block read, skip to the offset:
                    block_data = block_data[:, offset_samples:]
                    samples_read = samples_read - offset_samples
                    offset_samples = 0
                if samples_to_read < samples_read:
                    # Last block to read, skip the last samples
                    block_data = block_data[:, :samples_to_read]
                    samples_read = samples_to_read

                s_start = current_data_sample
                s_end = s_start + samples_read

                one[eeg_one, disk_use_idx[s_start:s_end]] = block_data[eeg_in]
                samples_to_read = samples_to_read - samples_read
                current_data_sample = current_data_sample + samples_read

        if len(pns_one) > 0:
            # PNS Data is present and should be read:
            pns_filepath = egi_info['pns_filepath']
            pns_info = egi_info['pns_sample_blocks']
            n_channels = pns_info['n_channels']
            samples_block = pns_info['samples_block']

            # Get starting/stopping block/samples
            block_samples_offset = np.cumsum(samples_block)
            offset_blocks = np.sum(block_samples_offset < start)
            offset_samples = start - (block_samples_offset[offset_blocks - 1]
                                      if offset_blocks > 0 else 0)

            samples_to_read = stop - start
            with open(pns_filepath, 'rb', buffering=0) as fid:
                # Check file size
                fid.seek(0, 2)
                file_size = fid.tell()
                fid.seek(0)
                # Go to starting block
                current_block = 0
                current_block_info = None
                current_data_sample = 0
                while current_block < offset_blocks:
                    this_block_info = _block_r(fid)
                    if this_block_info is not None:
                        current_block_info = this_block_info
                    fid.seek(current_block_info['block_size'], 1)
                    current_block = current_block + 1

                # Start reading samples
                while samples_to_read > 0:
                    if samples_to_read == 1 and fid.tell() == file_size:
                        # We are in the presence of the EEG bug
                        # fill with zeros and break the loop
                        one[pns_one, -1] = 0
                        break

                    this_block_info = _block_r(fid)
                    if this_block_info is not None:
                        current_block_info = this_block_info

                    to_read = (current_block_info['nsamples'] *
                               current_block_info['nc'])
                    block_data = np.fromfile(fid, dtype, to_read)
                    block_data = block_data.reshape(n_channels, -1, order='C')

                    # Compute indexes
                    samples_read = block_data.shape[1]
                    if offset_samples > 0:
                        # First block read, skip to the offset:
                        block_data = block_data[:, offset_samples:]
                        samples_read = samples_read - offset_samples
                        offset_samples = 0

                    if samples_to_read < samples_read:
                        # Last block to read, skip the last samples
                        block_data = block_data[:, :samples_to_read]
                        samples_read = samples_to_read

                    s_start = current_data_sample
                    s_end = s_start + samples_read

                    one[pns_one, disk_use_idx[s_start:s_end]] = \
                        block_data[pns_in]
                    samples_to_read = samples_to_read - samples_read
                    current_data_sample = current_data_sample + samples_read

        # do the calibration
        _mult_cal_one(data, one, idx, cals, mult)


@verbose
def read_evokeds_mff(fname, condition=None, channel_naming='E%d',
                     baseline=None, verbose=None):
    """Read averaged MFF file as EvokedArray or list of EvokedArray.

    Parameters
    ----------
    fname : str
        File path to averaged MFF file. Should end in .mff.
    condition : int or str | list of int or str | None
        The index (indices) or category (categories) from which to read in
        data. Averaged MFF files can contain separate averages for different
        categories. These can be indexed by the block number or the category
        name. If ``condition`` is a list or None, a list of EvokedArray objects
        is returned.
    channel_naming : str
        Channel naming convention for EEG channels. Defaults to 'E%%d'
        (resulting in channel names 'E1', 'E2', 'E3'...).
    baseline : None (default) or tuple of length 2
        The time interval to apply baseline correction. If None do not apply
        it. If baseline is (a, b) the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used and if b is None then b
        is set to the end of the interval. If baseline is equal to (None, None)
        all the time interval is used. Correction is applied by computing mean
        of the baseline period and subtracting it from the data. The baseline
        (a, b) includes both endpoints, i.e. all timepoints t such that
        a <= t <= b.
    %(verbose)s

    Returns
    -------
    evoked : EvokedArray or list of EvokedArray
        The evoked dataset(s); one EvokedArray if condition is int or str,
        or list of EvokedArray if condition is None or list.

    Raises
    ------
    ValueError
        If ``fname`` has file extension other than '.mff'.
    ValueError
        If the MFF file specified by ``fname`` is not averaged.
    ValueError
        If no categories.xml file in MFF directory specified by ``fname``.

    See Also
    --------
    Evoked, EvokedArray, create_info

    Notes
    -----
    .. versionadded:: 0.22
    """
    mffpy = _import_mffpy()
    # Confirm `fname` is a path to an MFF file
    if not fname.endswith('.mff'):
        raise ValueError('fname must be an MFF file with extension ".mff".')
    # Confirm the input MFF is averaged
    mff = mffpy.Reader(fname)
    if mff.flavor != 'averaged':
        raise ValueError(f'{fname} is a {mff.flavor} MFF file. '
                         'fname must be the path to an averaged MFF file.')
    # Check for categories.xml file
    if 'categories.xml' not in mff.directory.listdir():
        raise ValueError('categories.xml not found in MFF directory. '
                         f'{fname} may not be an averaged MFF file.')
    return_list = True
    if condition is None:
        categories = mff.categories.categories
        condition = list(categories.keys())
    elif not isinstance(condition, list):
        condition = [condition]
        return_list = False
    logger.info(f'Reading {len(condition)} evoked datasets from {fname} ...')
    output = [_read_evoked_mff(fname, c, channel_naming=channel_naming,
                               verbose=verbose).apply_baseline(baseline)
              for c in condition]
    return output if return_list else output[0]


def _read_evoked_mff(fname, condition, channel_naming='E%d', verbose=None):
    """Read evoked data from MFF file."""
    import mffpy
    egi_info = _read_header(fname)
    mff = mffpy.Reader(fname)
    categories = mff.categories.categories

    if isinstance(condition, str):
        # Condition is interpreted as category name
        category = _check_option('condition', condition, categories,
                                 extra='provided as category name')
        epoch = mff.epochs[category]
    elif isinstance(condition, int):
        # Condition is interpreted as epoch index
        try:
            epoch = mff.epochs[condition]
        except IndexError:
            raise ValueError(f'"condition" parameter ({condition}), provided '
                             'as epoch index, is out of range for available '
                             f'epochs ({len(mff.epochs)}).')
        category = epoch.name
    else:
        raise TypeError('"condition" parameter must be either int or str.')

    # Read in signals from the target epoch
    data = mff.get_physical_samples_from_epoch(epoch)
    eeg_data, t0 = data['EEG']
    if 'PNSData' in data:
        pns_data, t0 = data['PNSData']
        all_data = np.vstack((eeg_data, pns_data))
        ch_types = egi_info['chan_type'] + egi_info['pns_types']
    else:
        all_data = eeg_data
        ch_types = egi_info['chan_type']
    all_data *= 1e-6  # convert to volts

    # Load metadata into info object
    # Exclude info['meas_date'] because record time info in
    # averaged MFF is the time of the averaging, not true record time.
    ch_names = [channel_naming % (i + 1) for i in
                range(mff.num_channels['EEG'])]
    ch_names.extend(egi_info['pns_names'])
    info = create_info(ch_names, mff.sampling_rates['EEG'], ch_types)
    info['nchan'] = sum(mff.num_channels.values())

    # Add individual channel info
    # Get calibration info for EEG channels
    cals = _get_eeg_calibration_info(fname, egi_info)
    # Initialize calibration for PNS channels, will be updated later
    cals = np.concatenate([cals, np.repeat(1, len(egi_info['pns_names']))])
    ch_coil = FIFF.FIFFV_COIL_EEG
    ch_kind = FIFF.FIFFV_EEG_CH
    chs = _create_chs(ch_names, cals, ch_coil, ch_kind, (), (), (), ())
    chs = _read_locs(fname, chs, egi_info)
    # Update PNS channel info
    chs = _add_pns_channel_info(chs, egi_info, ch_names)
    info['chs'] = chs

    # Add bad channels to info
    info['description'] = category
    try:
        channel_status = categories[category][0]['channelStatus']
    except KeyError:
        warn(f'Channel status data not found for condition {category}. '
             'No channels will be marked as bad.', category=UserWarning)
        channel_status = None
    bads = []
    if channel_status:
        for entry in channel_status:
            if entry['exclusion'] == 'badChannels':
                if entry['signalBin'] == 1:
                    # Add bad EEG channels
                    for ch in entry['channels']:
                        bads.append(channel_naming % ch)
                elif entry['signalBin'] == 2:
                    # Add bad PNS channels
                    for ch in entry['channels']:
                        bads.append(egi_info['pns_names'][ch - 1])
    info['bads'] = bads

    # Add EEG reference to info
    # Initialize 'custom_ref_applied' to False
    info['custom_ref_applied'] = False
    with mff.directory.filepointer('history') as fp:
        history = mffpy.XML.from_file(fp)
    for entry in history.entries:
        if entry['method'] == 'Montage Operations Tool':
            if 'Average Reference' in entry['settings']:
                # Average reference has been applied
                projector, info = setup_proj(info)
            else:
                # Custom reference has been applied that is not an average
                info['custom_ref_applied'] = True

    # Get nave from categories.xml
    try:
        nave = categories[category][0]['keys']['#seg']['data']
    except KeyError:
        warn(f'Number of averaged epochs not found for condition {category}. '
             'nave will default to 1.', category=UserWarning)
        nave = 1

    # Let tmin default to 0
    return EvokedArray(all_data, info, tmin=0., comment=category,
                       nave=nave, verbose=verbose)


def _import_mffpy(why='read averaged .mff files'):
    """Import and return module mffpy."""
    try:
        import mffpy
    except ImportError as exp:
        msg = f'mffpy is required to {why}, got:\n{exp}'
        raise ImportError(msg)

    return mffpy
