# -*- coding: utf-8 -*-
#
# License: BSD (3-clause)

import datetime
import os
import time

from xml.dom.minidom import parse
import dateutil.parser
import numpy as np

from .events import _read_events
from .general import (_get_signalfname, _get_ep_info, _extract, _get_blocks,
                      _get_gains)
from ..base import BaseRaw, _check_update_montage
from ..constants import FIFF
from ..meas_info import _empty_info
from ..utils import _create_chs
from ...utils import verbose, logger, warn


def _read_mff_header(filepath):
    """Read mff header.

    Parameters
    ----------
    filepath : str
        Path to the file.
    """
    eeg_files, info_files = _get_signalfname(filepath, 'PNSData')
    eeg_file = eeg_files[0]
    fname = os.path.join(filepath, eeg_file)
    signal_blocks = _get_blocks(fname)
    blocknumsamps = np.sum(signal_blocks['blockNumSamps'])

    pibhasref = False
    pibnchans = 0

    epoch_info = _get_ep_info(filepath)

    summaryinfo = dict(eeg_fname=eeg_file,
                       info_fname=info_files[0],
                       pibNChans=pibnchans,
                       pibHasRef=pibhasref,
                       blockNumSamps=blocknumsamps)
    summaryinfo.update(signal_blocks)

    # Pull header info from the summary info.
    categfile = os.path.join(filepath, 'categories.xml')
    if os.path.isfile(categfile):  # epochtype = 'seg'
        n_samples = epoch_info[0]['last_samp'] - epoch_info['first_samp']
        ntrials = len(epoch_info)
    else:  # 'cnt'
        n_samples = np.sum(summaryinfo['blockNumSamps'])
        ntrials = 1

    # Add the sensor info.
    sensor_layout_file = filepath + '/sensorLayout.xml'
    sensor_layout_obj = parse(sensor_layout_file)
    sensors = sensor_layout_obj.getElementsByTagName('sensor')
    label = []
    chan_type = []
    chan_unit = []
    n_chans = 0
    for sensor in sensors:
        sensortype = int(sensor.getElementsByTagName('type')[0]
                         .firstChild.data)
        if sensortype == 0 or sensortype == 1:
            if sensor.getElementsByTagName('name')[0].firstChild is None:
                sn = sensor.getElementsByTagName('number')[0].firstChild.data
                sn = sn.encode()
                tmp_label = 'E' + sn.decode()
            else:
                sn = sensor.getElementsByTagName('name')[0].firstChild.data
                sn = sn.encode()
                tmp_label = sn.decode()
            label.append(tmp_label)
            chan_type.append('eeg')
            chan_unit.append('uV')
            n_chans = n_chans + 1
    if n_chans != summaryinfo['n_channels']:
        print("Error. Should never occur.")

    if summaryinfo['pibNChans'] > 0:
        pns_set_file = filepath + '/pnsSet.xml'
        pns_set_obj = parse(pns_set_file)
        pns_sensors = pns_set_obj.getElementsByTagName('sensor')
        for p in range(summaryinfo['pibNChans']):
            tmp_label = 'pib' + str(p + 1)
            label.append(tmp_label)
            pns_sensor_obj = pns_sensors[p]
            chan_type.append(pns_sensor_obj.getElementsByTagName('name')[0]
                             .firstChild.data.encode())
            chan_unit.append(pns_sensor_obj.getElementsByTagName('unit')[0]
                             .firstChild.data.encode())

    info_filepath = filepath + "/" + "info.xml"  # add with filepath
    tags = ['mffVersion', 'recordTime']
    version_and_date = _extract(tags, filepath=info_filepath)
    summaryinfo.update({'version': version_and_date['mffVersion'][0],
                        'date': version_and_date['recordTime'][0],
                        'n_samples': n_samples,
                        'nTrials': ntrials,
                        'label': label,
                        'chan_type': chan_type,
                        'chan_unit': chan_unit})
    return summaryinfo


def _read_header(input_fname):
    """Obtain the headers of the file package mff.

    Parameters
    ----------
    input_fname : str
        Path for the file

    Returns
    -------
    info : dict
        Set with main headers.
    mff_hdr : dict
        Headers.
    mff_events : dict
        Events.
    """
    mff_hdr = _read_mff_header(input_fname)
    with open(input_fname + '/signal1.bin', 'rb') as fid:
        version = np.fromfile(fid, np.int32, 1)[0]
    time_n = dateutil.parser.parse(mff_hdr['date'])
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
    unsegmented = 1 if mff_hdr['nTrials'] == 1 else 0
    precision = 4
    if precision == 0:
        RuntimeError('Floating point precision is undefined.')
    if unsegmented:
        info.update(dict(n_categories=0,
                         n_segments=1,
                         n_events=0,
                         event_codes=[],
                         category_names=[],
                         category_lengths=[],
                         pre_baseline=0))
    else:
        raise NotImplementedError('Only continuos files are supported')
    info['unsegmented'] = unsegmented
    info['dtype'], info['orig_format'] = {2: ('>i2', 'short'),
                                          4: ('>f4', 'float'),
                                          6: ('>f8', 'double')}[precision]
    info['dtype'] = np.dtype(info['dtype'])
    info.update(mff_hdr)
    return info


@verbose
def read_raw_egi_mff(input_fname, montage=None, eog=None, misc=None,
                     include=None, exclude=None, preload=False,
                     channel_naming='E%d', verbose=None):
    """Read EGI mff binary as raw object.

    .. note:: This function attempts to create a synthetic trigger channel.
              See notes below.

    Parameters
    ----------
    input_fname : str
        Path to the raw file.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
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
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    channel_naming : str
        Channel naming convention for the data channels. Defaults to 'E%d'
        (resulting in channel names 'E1', 'E2', 'E3'...). The effective default
        prior to 0.14.0 was 'EEG %03d'.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : Instance of RawMff
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

    ..versionadded:: 0.15.0
    """
    return RawMff(input_fname, montage, eog, misc, include, exclude,
                  preload, channel_naming, verbose)


class RawMff(BaseRaw):
    """RAWMff class."""

    @verbose
    def __init__(self, input_fname, montage=None, eog=None, misc=None,
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
        gains = _get_gains(os.path.join(input_fname, egi_info['info_fname']))
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

            event_ids, = np.nonzero(np.unique(egi_events))
            logger.info('    Synthesizing trigger channel "STI 014" ...')
            logger.info('    Excluding events {%s} ...' %
                        ", ".join([k for i, k in enumerate(event_codes)
                                   if i not in include_]))
            self.event_id = dict(zip([e for e in event_codes if e in
                                      include_names], event_ids))
        else:
            # No events
            self.event_id = None

        info = _empty_info(egi_info['sfreq'])
        info['buffer_size_sec'] = 1.  # reasonable default
        my_time = datetime.datetime(
            egi_info['year'], egi_info['month'], egi_info['day'],
            egi_info['hour'], egi_info['minute'], egi_info['second'])
        my_timestamp = time.mktime(my_time.timetuple())
        info['meas_date'] = np.array([my_timestamp], dtype=np.float32)
        ch_names = [channel_naming % (i + 1) for i in
                    range(egi_info['n_channels'])]
        ch_names.extend(list(egi_info['event_codes']))
        if len(egi_events) > 0:
            ch_names.append('STI 014')  # channel for combined events
        ch_coil = FIFF.FIFFV_COIL_EEG
        ch_kind = FIFF.FIFFV_EEG_CH
        cals = np.concatenate(
            [cals, np.repeat(1, len(event_codes) + 1 + len(misc) + len(eog))])
        chs = _create_chs(ch_names, cals, ch_coil, ch_kind, eog, (), (), misc)
        sti_ch_idx = [i for i, name in enumerate(ch_names) if
                      name.startswith('STI') or name in event_codes]
        for idx in sti_ch_idx:
            chs[idx].update({'unit_mul': 0, 'cal': cals[idx],
                             'kind': FIFF.FIFFV_STIM_CH,
                             'coil_type': FIFF.FIFFV_COIL_NONE,
                             'unit': FIFF.FIFF_UNIT_NONE})
        info['chs'] = chs
        info._update_redundant()
        _check_update_montage(info, montage)
        file_bin = os.path.join(input_fname, egi_info['eeg_fname'])
        egi_info['egi_events'] = egi_events

        self._filenames = [file_bin]
        self._raw_extras = [egi_info]
        super(RawMff, self).__init__(
            info, preload=preload, orig_format=egi_info['orig_format'],
            filenames=[file_bin], last_samps=[egi_info['n_samples'] - 1],
            raw_extras=[egi_info], verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of data."""
        from ..utils import _mult_cal_one
        dtype = '<f4'
        n_bytes = np.dtype(dtype).itemsize
        egi_info = self._raw_extras[fi]
        offset = egi_info['header_sizes'][0] - n_bytes
        n_channels = egi_info['n_channels']
        n_samples = egi_info['blockNumSamps'][0]
        block_size = n_samples * n_channels
        data_offset = n_channels * start * n_bytes + offset
        data_left = (stop - start) * n_channels

        egi_events = egi_info['egi_events'][:, start:stop]
        extra_samps = (start // n_samples)
        beginning = extra_samps * block_size
        data_left += (data_offset - offset) // n_bytes - beginning
        # STI 014 is simply the sum of all event channels (powers of 2).
        if len(egi_events) > 0:
            e_start = 0
            egi_events = np.vstack([egi_events, np.sum(egi_events, axis=0)])
        with open(self._filenames[fi], 'rb', buffering=0) as fid:
            fid.seek(int(beginning * n_bytes + offset + extra_samps * n_bytes))
            # extract data in chunks
            sample_start = 0
            # s_offset determines the offset inside the block in samples.
            s_offset = (data_offset // n_bytes - beginning) // n_channels
            while sample_start * n_channels < data_left:
                flag = np.fromfile(fid, dtype=np.dtype('i4'), count=1)[0]
                if flag == 1:  # meta data
                    header_size = np.fromfile(fid, dtype=np.dtype('i4'),
                                              count=1)[0]
                    block_size = np.fromfile(fid, dtype=np.dtype('i4'),
                                             count=1)[0]
                    fid.seek(header_size - 3 * n_bytes, 1)

                block = np.fromfile(fid, dtype, block_size)
                block = block.reshape(n_channels, -1, order='C')

                count = data_left - sample_start * n_channels
                end = count // n_channels + 2
                if sample_start == 0:
                    block = block[:, s_offset:end]
                    sample_sl = slice(sample_start,
                                      sample_start + block.shape[1])
                elif count < block_size:
                    block = block[:, :end]
                if len(egi_events) > 0:
                    e_chs = egi_events[:, e_start:e_start + block.shape[1]]
                    block = np.vstack([block, e_chs])
                    e_start += block.shape[1]
                if sample_start != 0:
                    sample_sl = slice(sample_start - s_offset,
                                      sample_start - s_offset + block.shape[1])
                data_view = data[:, sample_sl]
                sample_start = sample_start + n_samples
                _mult_cal_one(data_view, block, idx, cals, mult)
