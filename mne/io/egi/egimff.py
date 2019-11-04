"""EGI NetStation Load Function."""

import datetime
import os.path as op
import time
from xml.dom.minidom import parse

import numpy as np

from .events import _read_events, _combine_triggers
from .general import (_get_signalfname, _get_ep_info, _extract, _get_blocks,
                      _get_gains, _block_r)
from ..base import BaseRaw
from ..constants import FIFF
from ..meas_info import _empty_info
from ..utils import _create_chs
from ...utils import verbose, logger, warn
from ...annotations import _sync_onset


def _read_mff_header(filepath):
    """Read mff header.

    Parameters
    ----------
    filepath : str
        Path to the file.
    """
    all_files = _get_signalfname(filepath)
    eeg_file = all_files['EEG']['signal']
    eeg_info_file = all_files['EEG']['info']

    fname = op.join(filepath, eeg_file)
    signal_blocks = _get_blocks(fname)
    samples_block = np.sum(signal_blocks['samples_block'])

    epoch_info = _get_ep_info(filepath)
    summaryinfo = dict(eeg_fname=eeg_file,
                       info_fname=eeg_info_file,
                       samples_block=samples_block)
    summaryinfo.update(signal_blocks)

    # Pull header info from the summary info.
    categfile = op.join(filepath, 'categories.xml')
    if op.isfile(categfile):  # epochtype = 'seg'
        n_samples = epoch_info[0]['last_samp'] - epoch_info['first_samp']
        n_trials = len(epoch_info)
    else:  # 'cnt'
        n_samples = np.sum(summaryinfo['samples_block'])
        n_trials = 1

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
        print("Error. Should never occur.")

    # Check presence of PNS data
    if 'PNS' in all_files:
        pns_fpath = op.join(filepath, all_files['PNS']['signal'])
        pns_blocks = _get_blocks(pns_fpath)

        pns_file = op.join(filepath, 'pnsSet.xml')
        pns_obj = parse(pns_file)
        sensors = pns_obj.getElementsByTagName('sensor')
        pns_names = []
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
                           pns_names=pns_names, n_pns_channels=len(pns_names),
                           pns_fname=all_files['PNS']['signal'],
                           pns_sample_blocks=pns_blocks)

    info_filepath = op.join(filepath, 'info.xml')  # add with filepath
    tags = ['mffVersion', 'recordTime']
    version_and_date = _extract(tags, filepath=info_filepath)
    version = ""
    if len(version_and_date['mffVersion']):
        version = version_and_date['mffVersion'][0]
    summaryinfo.update(version=version,
                       date=version_and_date['recordTime'][0],
                       n_samples=n_samples, n_trials=n_trials,
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
    unsegmented = 1 if mff_hdr['n_trials'] == 1 else 0
    if unsegmented:
        info.update(dict(n_categories=0,
                         n_segments=1,
                         n_events=0,
                         event_codes=[],
                         category_names=[],
                         category_lengths=[],
                         pre_baseline=0))
    else:
        raise NotImplementedError('Only continuous files are supported')
    info['unsegmented'] = unsegmented
    info.update(mff_hdr)
    return info


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
        Channel naming convention for the data channels. Defaults to 'E%d'
        (resulting in channel names 'E1', 'E2', 'E3'...). The effective default
        prior to 0.14.0 was 'EEG %03d'.
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
        gains = _get_gains(op.join(input_fname, egi_info['info_fname']))
        if egi_info['value_range'] != 0 and egi_info['bits'] != 0:
            cals = [egi_info['value_range'] / 2 ** egi_info['bits'] for i
                    in range(len(egi_info['chan_type']))]
        else:
            cal_scales = {'uV': 1e-6, 'V': 1}
            cals = [cal_scales[t] for t in egi_info['chan_unit']]
        if 'gcal' in gains:
            cals *= gains['gcal']
        if 'ical' in gains:
            pass  # XXX: currently not used
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
            self._new_trigger = _combine_triggers(egi_events[include_],
                                                  remapping=events_ids)
            self.event_id = dict(zip([e for e in event_codes if e in
                                      include_names], events_ids))
            if self._new_trigger is not None:
                egi_events = np.vstack([egi_events, self._new_trigger])
        else:
            # No events
            self.event_id = None
            event_codes = []
        info = _empty_info(egi_info['sfreq'])
        my_time = datetime.datetime(
            egi_info['year'], egi_info['month'], egi_info['day'],
            egi_info['hour'], egi_info['minute'], egi_info['second'])
        my_timestamp = time.mktime(my_time.timetuple())
        info['meas_date'] = (my_timestamp, 0)
        ch_names = [channel_naming % (i + 1) for i in
                    range(egi_info['n_channels'])]
        ch_names.extend(list(egi_info['event_codes']))
        if hasattr(self, '_new_trigger') and self._new_trigger is not None:
            ch_names.append('STI 014')  # channel for combined events
        ch_coil = FIFF.FIFFV_COIL_EEG
        ch_kind = FIFF.FIFFV_EEG_CH
        cals = np.concatenate(
            [cals, np.repeat(1, len(event_codes) + 1 + len(misc) + len(eog))])
        if 'pns_names' in egi_info:
            ch_names.extend(egi_info['pns_names'])
            cals = np.concatenate(
                [cals, np.repeat(1, len(egi_info['pns_names']))])
        chs = _create_chs(ch_names, cals, ch_coil, ch_kind, eog, (), (), misc)
        chs = _read_locs(input_fname, chs, egi_info)
        sti_ch_idx = [i for i, name in enumerate(ch_names) if
                      name.startswith('STI') or name in event_codes]
        for idx in sti_ch_idx:
            chs[idx].update({'unit_mul': 0, 'cal': cals[idx],
                             'kind': FIFF.FIFFV_STIM_CH,
                             'coil_type': FIFF.FIFFV_COIL_NONE,
                             'unit': FIFF.FIFF_UNIT_NONE})
        if 'pns_names' in egi_info:
            for i_ch, ch_name in enumerate(egi_info['pns_names']):
                idx = ch_names.index(ch_name)
                ch_type = egi_info['pns_types'][i_ch]
                ch_kind = FIFF.FIFFV_BIO_CH
                if ch_type == 'ecg':
                    ch_kind = FIFF.FIFFV_ECG_CH
                elif ch_type == 'emg':
                    ch_kind = FIFF.FIFFV_EMG_CH
                ch_unit = FIFF.FIFF_UNIT_V
                ch_cal = 1e-6
                if egi_info['pns_units'][i_ch] != 'uV':
                    ch_unit = FIFF.FIFF_UNIT_NONE
                    ch_cal = 1.0

                chs[idx].update({'cal': ch_cal, 'kind': ch_kind,
                                 'coil_type': FIFF.FIFFV_COIL_NONE,
                                 'unit': ch_unit})

        info['chs'] = chs
        info._update_redundant()
        file_bin = op.join(input_fname, egi_info['eeg_fname'])
        egi_info['egi_events'] = egi_events

        if 'pns_names' in egi_info:
            egi_info['pns_filepath'] = op.join(
                input_fname, egi_info['pns_fname'])

        self._filenames = [file_bin]
        self._raw_extras = [egi_info]

        super(RawMff, self).__init__(
            info, preload=preload, orig_format='float', filenames=[file_bin],
            last_samps=[egi_info['n_samples'] - 1], raw_extras=[egi_info],
            verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of data."""
        from ..utils import _mult_cal_one
        dtype = '<f4'  # Data read in four byte floats.

        egi_info = self._raw_extras[fi]

        # info about the binary file structure
        n_channels = egi_info['n_channels']
        samples_block = egi_info['samples_block']

        # Check how many channels to read are from EEG
        if isinstance(idx, slice):
            chs_to_read = self.info['chs'][idx]
        else:
            chs_to_read = [self.info['chs'][x] for x in idx]
        eeg_chans = [i for i, x in enumerate(chs_to_read) if x['kind'] in
                     (FIFF.FIFFV_EEG_CH, FIFF.FIFFV_STIM_CH)]
        pns_chans = [i for i, x in enumerate(chs_to_read) if x['kind'] in
                     (FIFF.FIFFV_ECG_CH, FIFF.FIFFV_EMG_CH, FIFF.FIFFV_BIO_CH)]

        eeg_chans = np.array(eeg_chans)
        pns_chans = np.array(pns_chans)

        if len(pns_chans):
            if not np.max(eeg_chans) < np.max(pns_chans):
                raise ValueError('Currently interlacing EEG and PNS channels'
                                 'is not supported')
        # Number of channels to be read from EEG
        n_data1_channels = len(eeg_chans)

        # Number of channels expected in the EEG binary file
        n_eeg_channels = n_channels

        # Get starting/stopping block/samples
        block_samples_offset = np.cumsum(samples_block)
        offset_blocks = np.sum(block_samples_offset < start)
        offset_samples = start - (block_samples_offset[offset_blocks - 1]
                                  if offset_blocks > 0 else 0)

        samples_to_read = stop - start

        # Now account for events
        egi_events = egi_info['egi_events']
        if len(egi_events) > 0:
            n_eeg_channels += egi_events.shape[0]

        if len(pns_chans):
            # Split idx slice into EEG and PNS
            if isinstance(idx, slice):
                if idx.start is not None or idx.stop is not None:
                    eeg_idx = slice(idx.start, n_data1_channels)
                    pns_idx = slice(0, idx.stop - n_eeg_channels)
                else:
                    eeg_idx = idx
                    pns_idx = idx
            else:
                eeg_idx = idx[eeg_chans]
                pns_idx = idx[pns_chans] - n_eeg_channels
        else:
            eeg_idx = idx
            pns_idx = []

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

                # take into account events
                if len(egi_events) > 0:
                    e_chs = egi_events[:, start + s_start:start + s_end]
                    block_data = np.vstack([block_data, e_chs])

                data_view = data[:n_data1_channels, s_start:s_end]

                _mult_cal_one(data_view, block_data, eeg_idx,
                              cals[:n_data1_channels], mult)
                samples_to_read = samples_to_read - samples_read
                current_data_sample = current_data_sample + samples_read

        if 'pns_names' in egi_info and len(pns_chans) > 0:
            # PNS Data is present and should be read:
            pns_filepath = egi_info['pns_filepath']
            n_pns_channels = egi_info['n_pns_channels']
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
                        data[n_data1_channels:, -1] = 0
                        warn('This file has the EGI PSG sample bug')
                        an_start = current_data_sample
                        # XXX : use of _sync_onset should live in annotations
                        self.annotations.append(
                            _sync_onset(self, an_start / self.info['sfreq']),
                            1 / self.info['sfreq'], 'BAD_EGI_PSG')
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

                    data_view = data[n_data1_channels:, s_start:s_end]
                    _mult_cal_one(data_view, block_data[:n_pns_channels],
                                  pns_idx,
                                  cals[n_data1_channels:], mult)
                    del data_view
                    samples_to_read = samples_to_read - samples_read
                    current_data_sample = current_data_sample + samples_read
