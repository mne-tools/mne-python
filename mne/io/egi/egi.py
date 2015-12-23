# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#
#          simplified BSD-3 license

import datetime
import time
import warnings

import numpy as np

from ..base import _BaseRaw, _check_update_montage
from ..utils import _mult_cal_one
from ..meas_info import _empty_info
from ..constants import FIFF
from ...utils import verbose, logger


def _read_header(fid):
    """Read EGI binary header"""

    version = np.fromfile(fid, np.int32, 1)[0]

    if version > 6 & ~np.bitwise_and(version, 6):
        version = version.byteswap().astype(np.uint32)
    else:
        ValueError('Watchout. This does not seem to be a simple '
                   'binary EGI file.')

    def my_fread(*x, **y):
        return np.fromfile(*x, **y)[0]

    info = dict(
        version=version,
        year=my_fread(fid, '>i2', 1),
        month=my_fread(fid, '>i2', 1),
        day=my_fread(fid, '>i2', 1),
        hour=my_fread(fid, '>i2', 1),
        minute=my_fread(fid, '>i2', 1),
        second=my_fread(fid, '>i2', 1),
        millisecond=my_fread(fid, '>i4', 1),
        samp_rate=my_fread(fid, '>i2', 1),
        n_channels=my_fread(fid, '>i2', 1),
        gain=my_fread(fid, '>i2', 1),
        bits=my_fread(fid, '>i2', 1),
        value_range=my_fread(fid, '>i2', 1)
    )

    unsegmented = 1 if np.bitwise_and(version, 1) == 0 else 0
    precision = np.bitwise_and(version, 6)
    if precision == 0:
        RuntimeError('Floating point precision is undefined.')

    if unsegmented:
        info.update(dict(n_categories=0,
                         n_segments=1,
                         n_samples=np.fromfile(fid, '>i4', 1)[0],
                         n_events=np.fromfile(fid, '>i2', 1)[0],
                         event_codes=[],
                         category_names=[],
                         category_lengths=[],
                         pre_baseline=0))
        for event in range(info['n_events']):
            event_codes = ''.join(np.fromfile(fid, 'S1', 4).astype('U1'))
            info['event_codes'].append(event_codes)
        info['event_codes'] = np.array(info['event_codes'])
    else:
        raise NotImplementedError('Only continous files are supported')
    info['unsegmented'] = unsegmented
    info['dtype'], info['orig_format'] = {2: ('>i2', 'short'),
                                          4: ('>f4', 'float'),
                                          6: ('>f8', 'double')}[precision]
    info['dtype'] = np.dtype(info['dtype'])
    return info


def _read_events(fid, info):
    """Read events"""
    events = np.zeros([info['n_events'],
                       info['n_segments'] * info['n_samples']])
    fid.seek(36 + info['n_events'] * 4, 0)  # skip header
    for si in range(info['n_samples']):
        # skip data channels
        fid.seek(info['n_channels'] * info['dtype'].itemsize, 1)
        # read event channels
        events[:, si] = np.fromfile(fid, info['dtype'], info['n_events'])
    return events


def _combine_triggers(data, remapping=None):
    """Combine binary triggers"""
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
def read_raw_egi(input_fname, montage=None, eog=None, misc=None,
                 include=None, exclude=None, preload=None, verbose=None):
    """Read EGI simple binary as raw object

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
    return RawEGI(input_fname, montage, eog, misc, include, exclude, preload,
                  verbose)


class RawEGI(_BaseRaw):
    """Raw object from EGI simple binary file
    """
    @verbose
    def __init__(self, input_fname, montage=None, eog=None, misc=None,
                 include=None, exclude=None, preload=None, verbose=None):
        if preload is None:
            warnings.warn('preload is True by default but will be changed to '
                          'False in v0.12. Please explicitly set preload.',
                          DeprecationWarning)
            preload = True
        if eog is None:
            eog = []
        if misc is None:
            misc = []
        with open(input_fname, 'rb') as fid:  # 'rb' important for py3k
            logger.info('Reading EGI header from %s...' % input_fname)
            egi_info = _read_header(fid)
            logger.info('    Reading events ...')
            egi_events = _read_events(fid, egi_info)  # update info + jump
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
                    warnings.warn('Did not find any event code with more '
                                  'than one event.', RuntimeWarning)
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
            self._new_trigger = _combine_triggers(egi_events[include_],
                                                  remapping=event_ids)
            self.event_id = dict(zip([e for e in event_codes if e in
                                      include_names], event_ids))
        else:
            # No events
            self.event_id = None
            self._new_trigger = None
        info = _empty_info(egi_info['samp_rate'])
        info['buffer_size_sec'] = 1.  # reasonable default
        info['filename'] = input_fname
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
        info['nchan'] = nchan = len(ch_names)
        info['ch_names'] = ch_names
        for ii, ch_name in enumerate(ch_names):
            ch_info = {
                'cal': cal, 'logno': ii + 1, 'scanno': ii + 1, 'range': 1.0,
                'unit_mul': 0, 'ch_name': ch_name, 'unit': FIFF.FIFF_UNIT_V,
                'coord_frame': FIFF.FIFFV_COORD_HEAD,
                'coil_type': FIFF.FIFFV_COIL_EEG, 'kind': FIFF.FIFFV_EEG_CH,
                'loc': np.array([0, 0, 0, 1] * 3, dtype='f4')}
            if ch_name in eog or ii in eog or ii - nchan in eog:
                ch_info.update(coil_type=FIFF.FIFFV_COIL_NONE,
                               kind=FIFF.FIFFV_EOG_CH)
            if ch_name in misc or ii in misc or ii - nchan in misc:
                ch_info.update(coil_type=FIFF.FIFFV_COIL_NONE,
                               kind=FIFF.FIFFV_MISC_CH)
            if len(ch_name) == 4 or ch_name.startswith('STI'):
                ch_info.update(
                    {'unit_mul': 0, 'cal': 1, 'kind': FIFF.FIFFV_STIM_CH,
                     'coil_type': FIFF.FIFFV_COIL_NONE,
                     'unit': FIFF.FIFF_UNIT_NONE})
            info['chs'].append(ch_info)
        _check_update_montage(info, montage)
        super(RawEGI, self).__init__(
            info, preload, orig_format=egi_info['orig_format'],
            filenames=[input_fname], last_samps=[egi_info['n_samples'] - 1],
            raw_extras=[egi_info], verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file"""
        egi_info = self._raw_extras[fi]
        n_chan_read = egi_info['n_channels'] + egi_info['n_events']
        data_start = (36 + egi_info['n_events'] * 4 +
                      start * n_chan_read * egi_info['dtype'].itemsize)
        n_chan_out = n_chan_read + (1 if self._new_trigger is not None else 0)
        one = np.empty((n_chan_out, stop - start))
        with open(self._filenames[fi], 'rb') as fid:
            fid.seek(data_start, 0)  # skip header
            final_shape = (stop - start, n_chan_read)
            one_ = np.fromfile(fid, egi_info['dtype'], np.prod(final_shape))
            one_.shape = final_shape
            one[:n_chan_read] = one_.T
        # reads events as well
        if self._new_trigger is not None:
            one[-1] = self._new_trigger[start:stop]
        _mult_cal_one(data, one, idx, cals, mult)
