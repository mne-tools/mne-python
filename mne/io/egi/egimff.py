# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 11:01:39 2017.

@author: ramonapariciog
"""
import datetime
import dateutil.parser
import time

import numpy as np

from .source.header import read_mff_header
from .source.data import read_mff_data
from .source.events import read_mff_events

from ..base import BaseRaw, _check_update_montage
from ..utils import _create_chs
from ..meas_info import _empty_info
from ..constants import FIFF
from ...utils import verbose, logger, warn


def _read_header(input_fname):
    """Obtain the headers of the file package mff.

    Parameters
    ----------
    input_fname : str
        Path for the file
    Returns
    ------------------------
    info : dictionary
        Set with main headers.
    mff_hdr : dictionary
        Headers.
    mff_events : dictionary.
        Events.
    """
    mff_hdr = read_mff_header(input_fname)
    with open(input_fname + '/signal1.bin', 'rb') as fid:
        version = np.fromfile(fid, np.int32, 1)[0]
    # if version > 6 & ~np.bitwise_and(version, 6):
    #    version = version.byteswap().astype(np.uint32)
    # else:
    #    ValueError('Watchout. This does not seem to be a simple '
    #               'binary EGI file.')

    time_n = dateutil.parser.parse(mff_hdr['date'])
    info = dict(   # leer del info.xml desde read_mff_header
        version=version,  # duda
        year=int(time_n.strftime('%Y')),           # duda
        month=int(time_n.strftime('%m')),          # duda
        day=int(time_n.strftime('%d')),
        hour=int(time_n.strftime('%H')),
        minute=int(time_n.strftime('%M')),
        second=int(time_n.strftime('%S')),
        millisecond=int(time_n.strftime('%f')),
        samp_rate=mff_hdr['Fs'],
        n_channels=mff_hdr['nChans'],
        gain=0,           # duda
        bits=0,           # duda
        value_range=0)    # duda
    unsegmented = 1 if mff_hdr['nTrials'] == 1 else 0
    precision = 4
    # np.bitwise_and(version,6)
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


def _read_events(input_fname, hdr, info):
    """Read events for the record.

    in:
        input_fname : str with the file path
        hdr : dictionary with the headers get
              by read_mff_header
        info : header info array
    """
    mff_events, event_codes = read_mff_events(input_fname, hdr)
    info['n_events'] = len(event_codes)
    info['event_codes'] = np.asarray(event_codes).astype('<U4')
    events = np.zeros([info['n_events'],
                      info['n_segments'] * info['n_samples']])
    for n, event in enumerate(event_codes):
        for i in mff_events[event]:
            events[n][i] = 1.4012984643248171e-45
    return events, info


def _combine_triggers_mff(data, remapping=None):
    """Combine binary triggers."""
    new_trigger = np.zeros(data.shape[1])
    if data.astype(bool).sum(axis=0).max() > 1:  # ensure no overlaps
        logger.info('    Found multiple events at the same time '
                    'sample. Cannot create trigger channel.')
        return
    if remapping is None:
        remapping = np.arange(data) + 1
    for d, event_id in zip(data, remapping):
        idx = d.nonzero()
        if np.any(idx):
            new_trigger[idx] += event_id
    return new_trigger


@verbose
def read_raw_egi_mff(input_fname, montage=None, eog=None, misc=None,
                 include=None, exclude=None, preload=False, verbose=None):
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

        ..versionadded:: 0.11

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
    return RawMff(input_fname, montage, eog, misc, include, exclude, preload,
                  verbose)


class RawMff(BaseRaw):
    """RAWMff class."""

    @verbose
    def __init__(self, input_fname, montage=None, eog=None, misc=None,
                 include=None, exclude=None, preload=False, verbose=None):
        """Init for the RawMff class."""
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

        # duda

        logger.info('    Assembling measurement info ...')

        # from here is the same that the raw egi reader

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

            event_ids = np.arange(len(include_)) + 1
            logger.info('    Synthesizing trigger channel "STI 014" ...')
            logger.info('    Excluding events {%s} ...' %
                        ", ".join([k for i, k in enumerate(event_codes)
                                   if i not in include_]))
            self._new_trigger = _combine_triggers_mff(egi_events[include_],
                                                  remapping=event_ids)
            self.event_id = dict(zip([e for e in event_codes if e in
                                      include_names], event_ids))
        else:
            # No events
            self.event_id = None
            self._new_trigger = None
        info = _empty_info(egi_info['samp_rate'])
        info['buffer_size_sec'] = 1.  # reasonable default
        # info['filename'] = input_fname
        my_time = datetime.datetime(
            egi_info['year'], egi_info['month'], egi_info['day'],
            egi_info['hour'], egi_info['minute'], egi_info['second'])
        my_timestamp = time.mktime(my_time.timetuple())
        info['meas_date'] = np.array([my_timestamp], dtype=np.float32)
        ch_names = ['EEG %03d' % (i + 1) for i in
                    range(egi_info['n_channels'])]
        ch_names.extend(list(egi_info['event_codes']))
        if self._new_trigger is not None:
            ch_names.append('STI 014')  # our new_trigger
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
        data = read_mff_data(input_fname,
                             'epoch', 1, egi_hdr['nTrials'],
                             egi_hdr)
        data *= cal
        if self._new_trigger is not None:
            sti = self._new_trigger.reshape((1, len(self._new_trigger)))
            data = np.concatenate((data, egi_events, sti), axis=0)
        else:
            data = np.concatenate((data, egi_events), axis=0)
        dtype = np.complex128 if np.any(np.iscomplex(data)) else np.float64
        data = np.asanyarray(data, dtype=dtype)
        if len(data) != len(info['ch_names']):
            raise ValueError('len(data) does not match len(info["ch_names"])')
        logger.info('Creating RawArray with %s data, n_channels=%s, n_times=%s'
                    % (dtype.__name__, data.shape[0], data.shape[1]))
        super(RawMff, self).__init__(
            info, preload=data, orig_format=egi_info['orig_format'],
            filenames=[file_bin], last_samps=[egi_info['n_samples'] - 1],
            raw_extras=[egi_info], verbose=verbose)
