# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 11:01:39 2017.

@author: ramonapariciog
"""
import datetime
import os
import time

from xml.dom.minidom import parse
import dateutil.parser
import numpy as np

from .events import _read_events
from .general import (_block_r, _get_signal_bl, _get_ep_inf, _extract)
from ..base import BaseRaw, _check_update_montage
from ..constants import FIFF
from ..meas_info import _empty_info
from ..utils import _create_chs
from ...utils import verbose, logger, warn


def _read_mff_header(filepath):
    """Header reader Function.

    Parameters
    ----------
    filepath : str
        Path to the file.
    """
    signal_blocks = _get_signal_bl(filepath)
    samprate = signal_blocks['sampRate']
    numblocks = signal_blocks['blocks']
    blocknumsamps = np.array(signal_blocks['binObj'])

    pibhasref = False
    pibnchans = 0
    if signal_blocks['pibSignalFile'] != []:
        pnssetfile = filepath + '/pnsSet.xml'
        pnssetobj = parse(pnssetfile)
        pnssensors = pnssetobj.getElementsByTagName('sensor')
        pibnchans = pnssensors.length
        if signal_blocks['npibChan'] - pibnchans == 1:
            pibhasref = True

    epoch_info = _get_ep_inf(filepath, samprate)

    blockbeginsamps = np.zeros((numblocks), dtype='i8')
    for x in range(0, (numblocks - 1)):
        blockbeginsamps[x + 1] = blockbeginsamps[x] + blocknumsamps[x]

    summaryinfo = dict(blocks=signal_blocks['blocks'],
                       eegFilename=signal_blocks['eegFile'],
                       infoFile=signal_blocks['infoFile'],
                       sampRate=signal_blocks['sampRate'],
                       nChans=signal_blocks['nChan'],
                       pibBinObj=signal_blocks['pibBinObj'],
                       pibBlocks=signal_blocks['pibBlocks'],
                       pibFilename=signal_blocks['pibSignalFile'],
                       pibNChans=pibnchans,
                       pibHasRef=pibhasref,
                       epochType=epoch_info['epochType'],
                       epochBeginSamps=epoch_info['epochBeginSamps'],
                       epochNumSamps=epoch_info['epochNumSamps'],
                       epochFirstBlocks=epoch_info['epochFirstBlocks'],
                       epochLastBlocks=epoch_info['epochLastBlocks'],
                       epochLabels=epoch_info['epochLabels'],
                       epochTime0=epoch_info['epochTime0'],
                       multiSubj=epoch_info['multiSubj'],
                       epochSubjects=epoch_info['epochSubjects'],
                       epochFilenames=epoch_info['epochFilenames'],
                       epochSegStatus=epoch_info['epochSegStatus'],
                       blockBeginSamps=blockbeginsamps,
                       blockNumSamps=blocknumsamps)

    # Pull header info from the summary info.
    nsamplespre = 0
    if summaryinfo['epochType'] == 'seg':
        nsamples = summaryinfo['epochNumSamps'][0]
        ntrials = len(summaryinfo['epochNumSamps'])

        # if Time0 is the same for all segments...
        if len(set(summaryinfo['epochTime0'])) == 1:
            nsamplespre = summaryinfo['epochTime0'][0]
    else:
        nsamples = sum(summaryinfo['blockNumSamps'])
        ntrials = 1

    # Add the sensor info.
    sensor_layout_file = filepath + '/sensorLayout.xml'
    sensor_layout_obj = parse(sensor_layout_file)
    sensors = sensor_layout_obj.getElementsByTagName('sensor')
    label = []
    chantype = []
    chanunit = []
    tmp_label = []
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
            chantype.append('eeg')
            chanunit.append('uV')
            n_chans = n_chans + 1
    if n_chans != summaryinfo['nChans']:
        print("Error. Should never occur.")

    if summaryinfo['pibNChans'] > 0:
        pns_set_file = filepath + '/pnsSet.xml'
        pns_set_obj = parse(pns_set_file)
        pns_sensors = pns_set_obj.getElementsByTagName('sensor')
        for p in range(summaryinfo['pibNChans']):
            tmp_label = 'pib' + str(p + 1)
            label.append(tmp_label)
            pns_sensor_obj = pns_sensors[p]
            chantype.append(pns_sensor_obj.getElementsByTagName('name')[0]
                            .firstChild.data.encode())
            chanunit.append(pns_sensor_obj.getElementsByTagName('unit')[0]
                            .firstChild.data.encode())

    n_chans = n_chans + summaryinfo['pibNChans']
    info_filepath = filepath + "/" + "info.xml"  # add with filepath
    tags = ['mffVersion', 'recordTime']
    version_and_date = _extract(tags, filepath=info_filepath)
    header = dict(Fs=summaryinfo['sampRate'],
                  version=version_and_date['mffVersion'][0],
                  date=version_and_date['recordTime'][0],
                  nChans=n_chans,
                  nSamplesPre=nsamplespre,
                  nSamples=nsamples,
                  nTrials=ntrials,
                  label=label,
                  chantype=chantype,
                  chanunit=chanunit,
                  orig=summaryinfo)
    return header


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
        samp_rate=mff_hdr['Fs'],
        n_channels=mff_hdr['nChans'],
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
                         n_samples=mff_hdr['nSamples'],
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
    return info, mff_hdr


@verbose
def read_raw_egi_mff(input_fname, montage=None, eog=None, misc=None,
                     include=None, exclude=None, preload=False,
                     verbose=None):
    """Read EGI mff binary as raw object.

    .. note:: The trigger channel names are based on the
              arbitrary user dependent event codes used. However this
              function will attempt to generate a synthetic trigger channel
              named ``STI 014`` in accordance with the general
              Neuromag / MNE naming pattern.

              The event_id assignment equals
              ``np.arange(n_events - n_excluded) + 1``. The resulting
              `event_id` mapping is stored as attribute to the resulting
              raw object but will be ignored when saving to a fiff.
              Note. The trigger channel is artificially constructed based
              on timestamps received by the Netstation. As a consequence,
              triggers have only short durations.

              This step will fail if events are not mutually exclusive.

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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : Instance of RawEGI
        A Raw object containing EGI data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawMff(input_fname, montage, eog, misc, include, exclude,
                  preload, verbose)


class RawMff(BaseRaw):
    """RAWMff class."""

    @verbose
    def __init__(self, input_fname, montage=None, eog=None, misc=None,
                 include=None, exclude=None, preload=False, verbose=None):
        """Init the RawMff class."""
        if eog is None:
            eog = []
        if misc is None:
            misc = []
        logger.info('Reading EGI MFF Header from %s...' % input_fname)
        egi_info, egi_hdr = _read_header(input_fname)
        logger.info('    Reading events ...')
        egi_events, egi_info = _read_events(input_fname, egi_hdr, egi_info)
        if egi_info['value_range'] != 0 and egi_info['bits'] != 0:
            cal = egi_info['value_range'] / 2 ** egi_info['bits']
        else:
            cal = 1e-6

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

        info = _empty_info(egi_info['samp_rate'])
        info['buffer_size_sec'] = 1.  # reasonable default
        my_time = datetime.datetime(
            egi_info['year'], egi_info['month'], egi_info['day'],
            egi_info['hour'], egi_info['minute'], egi_info['second'])
        my_timestamp = time.mktime(my_time.timetuple())
        info['meas_date'] = np.array([my_timestamp], dtype=np.float32)
        ch_names = ['EEG %03d' % (i + 1) for i in
                    range(egi_info['n_channels'])]
        ch_names.extend(list(egi_info['event_codes']))
        if len(egi_events) > 0:
            ch_names.append('STI 014')  # channel for combined events
        nchan = len(ch_names)
        cals = np.repeat(cal, nchan)
        ch_coil = FIFF.FIFFV_COIL_EEG
        ch_kind = FIFF.FIFFV_EEG_CH
        chs = _create_chs(ch_names, cals, ch_coil, ch_kind, eog, (), (), misc)
        sti_ch_idx = [i for i, name in enumerate(ch_names) if
                      name.startswith('STI') or len(name) == 4]
        for idx in sti_ch_idx:
            chs[idx].update({'unit_mul': 0, 'cal': 1,
                             'kind': FIFF.FIFFV_STIM_CH,
                             'coil_type': FIFF.FIFFV_COIL_NONE,
                             'unit': FIFF.FIFF_UNIT_NONE})
        info['chs'] = chs
        info._update_redundant()
        _check_update_montage(info, montage)
        file_bin = input_fname + '/' + egi_hdr['orig']['eegFilename'][0]

        with open(file_bin, 'rb') as fid:
            block_info = _block_r(fid)
        egi_info['block_info'] = block_info
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
        egi_info = self._raw_extras[fi]
        block_info = egi_info['block_info']
        offset = block_info['position'] - 4
        n_channels = egi_info['n_channels']
        block_size = block_info['hl']
        dtype = '<f4'
        n_bytes = np.dtype(dtype).itemsize

        data_offset = n_channels * start * n_bytes + offset
        data_left = (stop - start) * n_channels

        egi_events = egi_info['egi_events'][:, start:stop]
        extra_samps = (start // block_info['nsamples'])
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
            s_offset = (data_offset // n_bytes - beginning) / n_channels
            while sample_start * n_channels < data_left:
                flag = np.fromfile(fid, dtype=np.dtype('i4'), count=1)[0]
                while flag == 1:  # meta data
                    # Meta data consists of:
                    # * 1 bytes of header size
                    # * 1 bytes of block size
                    # * 1 byte of n_channels
                    # * n_channels bytes of offsets
                    # * n_channels bytes of sigfreqs?
                    # * 1 byte of flag (1 for meta data, 0 for data)
                    headersize = np.fromfile(fid, dtype=np.dtype('i4'),
                                             count=1)[0]
                    blocksize = np.fromfile(fid, dtype=np.dtype('i4'),
                                            count=1)[0]
                    n_channels = np.fromfile(fid, dtype=np.dtype('i4'),
                                             count=1)[0]
                    offsets = np.fromfile(fid, dtype=np.dtype('i4'),
                                          count=n_channels)
                    sigfreqs = np.fromfile(fid, dtype=np.dtype('i4'),
                                           count=n_channels)
                    flag = np.fromfile(fid, dtype=np.dtype('i4'), count=1)[0]

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
                sample_start = sample_start + block_info['nsamples']
                _mult_cal_one(data_view, block, idx, cals, mult)
