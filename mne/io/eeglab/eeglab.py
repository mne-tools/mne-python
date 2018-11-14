# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
from functools import partial

from ..utils import (_read_segments_file, _find_channels,
                     _synthesize_stim_channel, _deprecate_stim_channel)
from ...utils import deprecated
from ..constants import FIFF, Bunch
from ..meas_info import _empty_info, create_info
from ..base import BaseRaw, _check_update_montage
from ...utils import logger, verbose, warn
from ...channels.montage import Montage
from ...epochs import BaseEpochs
from ...event import read_events
from ...externals.six import string_types
from ...annotations import (Annotations, events_from_annotations,
                            read_annotations)

# just fix the scaling for now, EEGLAB doesn't seem to provide this info
CAL = 1e-6


def _check_fname(fname):
    """Check if the file extension is valid."""
    fmt = str(op.splitext(fname)[-1])
    if fmt == '.dat':
        raise NotImplementedError(
            'Old data format .dat detected. Please update your EEGLAB '
            'version and resave the data in .fdt format')
    elif fmt != '.fdt':
        raise IOError('Expected .fdt file format. Found %s format' % fmt)


def _check_load_mat(fname, uint16_codec):
    """Check if the mat struct contains 'EEG'."""
    from ...externals.pymatreader import read_mat
    eeg = read_mat(fname, uint16_codec=uint16_codec)
    if 'ALLEEG' in eeg:
        raise NotImplementedError(
            'Loading an ALLEEG array is not supported. Please contact'
            'mne-python developers for more information.')
    if 'EEG' not in eeg:
        raise ValueError('Could not find EEG array in the .set file.')
    eeg = Bunch(**eeg['EEG'])
    eeg.trials = int(eeg.trials)
    eeg.nbchan = int(eeg.nbchan)
    eeg.pnts = int(eeg.pnts)
    return eeg


def _to_loc(ll):
    """Check if location exists."""
    if isinstance(ll, (int, float)) or len(ll) > 0:
        return ll
    else:
        return np.nan


def _get_info(eeg, montage, eog=()):
    """Get measurement info."""
    from scipy import io
    info = _empty_info(sfreq=eeg.srate)
    update_ch_names = True

    # add the ch_names and info['chs'][idx]['loc']
    path = None
    if not isinstance(eeg.chanlocs, np.ndarray) and eeg.nbchan == 1:
        eeg.chanlocs = [eeg.chanlocs]

    if isinstance(eeg.chanlocs, dict):
        eeg.chanlocs = _dol_to_lod(eeg.chanlocs)

    good = len(eeg.chanlocs) > 0

    if good:
        pos_fields = ['X', 'Y', 'Z']
        if isinstance(eeg.chanlocs[0], io.matlab.mio5_params.mat_struct):
            has_pos = all(hasattr(eeg.chanlocs[0], fld)
                          for fld in pos_fields)
        elif isinstance(eeg.chanlocs[0], np.ndarray):
            # Old files
            has_pos = all(fld in eeg.chanlocs[0].dtype.names
                          for fld in pos_fields)
        elif isinstance(eeg.chanlocs[0], dict):
            # new files
            has_pos = all(fld in eeg.chanlocs[0] for fld in pos_fields)
        else:
            good = False
            has_pos = False  # unknown (sometimes we get [0, 0])

    if good:
        get_pos = has_pos and montage is None
        pos_ch_names, ch_names, pos = list(), list(), list()
        kind = 'user_defined'
        update_ch_names = False
        for chanloc in eeg.chanlocs:
            ch_names.append(chanloc['labels'])
            if get_pos:
                loc_x = _to_loc(chanloc['X'])
                loc_y = _to_loc(chanloc['Y'])
                loc_z = _to_loc(chanloc['Z'])
                locs = np.r_[-loc_y, loc_x, loc_z]
                if not np.any(np.isnan(locs)):
                    pos_ch_names.append(chanloc['labels'])
                    pos.append(locs)
        n_channels_with_pos = len(pos_ch_names)
        info = create_info(ch_names, eeg.srate, ch_types='eeg')
        if n_channels_with_pos > 0:
            selection = np.arange(n_channels_with_pos)
            montage = Montage(np.array(pos), pos_ch_names, kind, selection)
    elif isinstance(montage, string_types):
        path = op.dirname(montage)
    else:  # if eeg.chanlocs is empty, we still need default chan names
        ch_names = ["EEG %03d" % ii for ii in range(eeg.nbchan)]

    if montage is None:
        info = create_info(ch_names, eeg.srate, ch_types='eeg')
    else:
        _check_update_montage(
            info, montage, path=path, update_ch_names=update_ch_names,
            raise_missing=False)

    if eog == 'auto':
        eog = _find_channels(ch_names)

    for idx, ch in enumerate(info['chs']):
        ch['cal'] = CAL
        if ch['ch_name'] in eog or idx in eog:
            ch['coil_type'] = FIFF.FIFFV_COIL_NONE
            ch['kind'] = FIFF.FIFFV_EOG_CH
    return info


def read_raw_eeglab(input_fname, montage=None, eog=(), event_id=None,
                    event_id_func='strip_to_integer', preload=False,
                    uint16_codec=None, stim_channel=None, verbose=None):
    r"""Read an EEGLAB .set file.

    Parameters
    ----------
    input_fname : str
        Path to the .set file. If the data is stored in a separate .fdt file,
        it is expected to be in the same folder as the .set file.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    eog : list | tuple | 'auto'
        Names or indices of channels that should be designated EOG channels.
        If 'auto', the channel names containing ``EOG`` or ``EYE`` are used.
        Defaults to empty tuple.
    event_id : dict | None
        The ids of the events to consider. If None (default), an empty dict is
        used and ``event_id_func`` (see below) is called on every event value.
        If dict, the keys will be mapped to trigger values on the stimulus
        channel and only keys not in ``event_id`` will be handled by
        ``event_id_func``. Keys are case-sensitive.
        Example::

            {'SyncStatus': 1; 'Pulse Artifact': 3}

        This was deprecated in 0.17 and will be removed in 0.18.
    event_id_func : None | str | callable
        What to do for events not found in ``event_id``. Must take one ``str``
        argument and return an ``int``. If string, must be 'strip-to-integer',
        in which case it defaults to stripping event codes such as "D128" or
        "S  1" of their non-integer parts and returns the integer.
        If the event is not in the ``event_id`` and calling ``event_id_func``
        on it results in a ``TypeError`` (e.g. if ``event_id_func`` is
        ``None``) or a ``ValueError``, the event is dropped.
        This was deprecated in 0.17 and will be removed in 0.18.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory). Note that
        preload=False will be effective only if the data is stored in a
        separate binary file.
    uint16_codec : str | None
        If your \*.set file contains non-ascii characters, sometimes reading
        it may fail and give rise to error message stating that "buffer is
        too small". ``uint16_codec`` allows to specify what codec (for example:
        'latin1' or 'utf-8') should be used when reading character arrays and
        can therefore help you solve this problem.
    stim_channel : bool (default True)
        Add a stim channel from the events.

        .. warning:: This defaults to True in 0.17 but will change to False in
                     0.18 (when no stim channel synthesis will be allowed)
                     and be removed in 0.19; migrate code to use
                     :func:`mne.events_from_annotations` instead.

        .. versionadded:: 0.17
    verbose : bool | str | int | None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raw : Instance of RawEEGLAB
        A Raw object containing EEGLAB .set data.

    Notes
    -----
    .. versionadded:: 0.11.0

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawEEGLAB(input_fname=input_fname, montage=montage, preload=preload,
                     eog=eog, event_id=event_id, event_id_func=event_id_func,
                     verbose=verbose, uint16_codec=uint16_codec,
                     stim_channel=stim_channel)


def read_epochs_eeglab(input_fname, events=None, event_id=None, montage=None,
                       eog=(), verbose=None, uint16_codec=None):
    r"""Reader function for EEGLAB epochs files.

    Parameters
    ----------
    input_fname : str
        Path to the .set file. If the data is stored in a separate .fdt file,
        it is expected to be in the same folder as the .set file.
    events : str | array, shape (n_events, 3) | None
        Path to events file. If array, it is the events typically returned
        by the read_events function. If some events don't match the events
        of interest as specified by event_id, they will be marked as 'IGNORED'
        in the drop log. If None, it is constructed from the EEGLAB (.set) file
        with each unique event encoded with a different integer.
    event_id : int | list of int | dict | None
        The id of the event to consider. If dict, the keys can later be used
        to access associated events.
        Example::

            {"auditory":1, "visual":3}

        If int, a dict will be created with
        the id as string. If a list, all events with the IDs specified
        in the list are used. If None, the event_id is constructed from the
        EEGLAB (.set) file with each descriptions copied from `eventtype`.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    eog : list | tuple | 'auto'
        Names or indices of channels that should be designated EOG channels.
        If 'auto', the channel names containing ``EOG`` or ``EYE`` are used.
        Defaults to empty tuple.
    verbose : bool | str | int | None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    uint16_codec : str | None
        If your \*.set file contains non-ascii characters, sometimes reading
        it may fail and give rise to error message stating that "buffer is
        too small". ``uint16_codec`` allows to specify what codec (for example:
        'latin1' or 'utf-8') should be used when reading character arrays and
        can therefore help you solve this problem.

    Returns
    -------
    epochs : instance of Epochs
        The epochs.

    Notes
    -----
    .. versionadded:: 0.11.0


    See Also
    --------
    mne.Epochs : Documentation of attribute and methods.
    """
    epochs = EpochsEEGLAB(input_fname=input_fname, events=events, eog=eog,
                          event_id=event_id, montage=montage, verbose=verbose,
                          uint16_codec=uint16_codec)
    return epochs


class RawEEGLAB(BaseRaw):
    r"""Raw object from EEGLAB .set file.

    Parameters
    ----------
    input_fname : str
        Path to the .set file. If the data is stored in a separate .fdt file,
        it is expected to be in the same folder as the .set file.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions. If None,
        sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    eog : list | tuple | 'auto'
        Names or indices of channels that should be designated EOG channels.
        If 'auto', the channel names containing ``EOG`` or ``EYE`` are used.
        Defaults to empty tuple.
    event_id : dict | None
        The ids of the events to consider. If None (default), an empty dict is
        used and ``event_id_func`` (see below) is called on every event value.
        If dict, the keys will be mapped to trigger values on the stimulus
        channel and only keys not in ``event_id`` will be handled by
        ``event_id_func``. Keys are case-sensitive.
        Example::

            {'SyncStatus': 1; 'Pulse Artifact': 3}

        This was deprecated in 0.17 and will be removed in 0.18.
    event_id_func : None | str | callable
        What to do for events not found in ``event_id``. Must take one ``str``
        argument and return an ``int``. If string, must be 'strip-to-integer',
        in which case it defaults to stripping event codes such as "D128" or
        "S  1" of their non-integer parts and returns the integer.
        If the event is not in the ``event_id`` and calling ``event_id_func``
        on it results in a ``TypeError`` (e.g. if ``event_id_func`` is
        ``None``) or a ``ValueError``, the event is dropped.
        This was deprecated in 0.17 and will be removed in 0.18.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires large
        amount of memory). If preload is a string, preload is the file name of
        a memory-mapped file which is used to store the data on the hard
        drive (slower, requires less memory).
    uint16_codec : str | None
        If your \*.set file contains non-ascii characters, sometimes reading
        it may fail and give rise to error message stating that "buffer is
        too small". ``uint16_codec`` allows to specify what codec (for example:
        'latin1' or 'utf-8') should be used when reading character arrays and
        can therefore help you solve this problem.
    stim_channel : bool (default True)
        Add a stim channel from the events.

        .. warning:: This defaults to True in 0.17 but will change to False in
                     0.18 (when no stim channel synthesis will be allowed)
                     and be removed in 0.19; migrate code to use
                     :func:`mne.events_from_annotations` instead.

        .. versionadded:: 0.17
    verbose : bool | str | int | None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raw : Instance of RawEEGLAB
        A Raw object containing EEGLAB .set data.

    Notes
    -----
    .. versionadded:: 0.11.0

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, input_fname, montage, eog=(), event_id=None,
                 event_id_func='strip_to_integer', preload=False,
                 uint16_codec=None, stim_channel=None,
                 verbose=None):  # noqa: D102
        stim_channel = _deprecate_stim_channel(stim_channel)
        basedir = op.dirname(input_fname)
        eeg = _check_load_mat(input_fname, uint16_codec)
        if eeg.trials != 1:
            raise TypeError('The number of trials is %d. It must be 1 for raw'
                            ' files. Please use `mne.io.read_epochs_eeglab` if'
                            ' the .set file contains epochs.' % eeg.trials)

        last_samps = [eeg.pnts - 1]
        info = _get_info(eeg, montage, eog=eog)

        if stim_channel:
            stim_chan = dict(ch_name='STI 014', coil_type=FIFF.FIFFV_COIL_NONE,
                             kind=FIFF.FIFFV_STIM_CH,
                             logno=len(info["chs"]) + 1,
                             scanno=len(info["chs"]) + 1, cal=1., range=1.,
                             loc=np.full(12, np.nan), unit=FIFF.FIFF_UNIT_NONE,
                             unit_mul=0., coord_frame=FIFF.FIFFV_COORD_UNKNOWN)
            info['chs'].append(stim_chan)
            info._update_redundant()

            # dummy event channel to be populated from annotations later on
            self._create_event_ch(np.empty((0, 3)), n_samples=eeg.pnts)
        else:
            self._event_ch = None

        # read the data
        if isinstance(eeg.data, string_types):
            data_fname = op.join(basedir, eeg.data)
            _check_fname(data_fname)
            logger.info('Reading %s' % data_fname)

            super(RawEEGLAB, self).__init__(
                info, preload, filenames=[data_fname], last_samps=last_samps,
                orig_format='double', verbose=verbose)
        else:
            if preload is False or isinstance(preload, string_types):
                warn('Data will be preloaded. preload=False or a string '
                     'preload is not supported when the data is stored in '
                     'the .set file')
            # can't be done in standard way with preload=True because of
            # different reading path (.set file)
            if eeg.nbchan == 1 and len(eeg.data.shape) == 1:
                n_chan, n_times = [1, eeg.data.shape[0]]
            else:
                n_chan, n_times = eeg.data.shape
            data = np.empty((n_chan + int(stim_channel), n_times), dtype=float)
            data[:n_chan] = eeg.data
            data *= CAL
            if stim_channel:
                data[-1] = self._event_ch
            super(RawEEGLAB, self).__init__(
                info, data, filenames=[input_fname], last_samps=last_samps,
                orig_format='double', verbose=verbose)

        # create event_ch from annotations
        annot = read_annotations(input_fname)
        self.set_annotations(annot)

        _check_boundary(annot, event_id)

        latencies = np.round(annot.onset * self.info['sfreq'])
        _check_latencies(latencies)

        if stim_channel:
            dropped_desc = []  # use to collect dropped descriptions
            event_id_ = partial(_event_id_func,
                                event_id=event_id,
                                event_id_func=event_id_func,
                                dropped=dropped_desc)
            events, _ = events_from_annotations(self, event_id=event_id_)
            annot_length = self.annotations.onset.size
            if events.shape[0] < annot_length:
                msg = (
                    "{0}/{1} event codes could not be mapped to integers. Use "
                    "the 'event_id' parameter to map such events manually.")
                warn(msg.format(annot_length - events.shape[0], annot_length))
            if not events.size and len(annot):  # only if some evs were in file
                logger.info(
                    'Returning empty stim channel. Some annotations were'
                    'found but dropped during build of the raw.'
                    'Please use `event_id` and `event_id_func` to drive'
                    'the selection/rejection of events')
            self._create_event_ch(events, n_samples=eeg.pnts)
            if getattr(self, 'preload', False):
                self._data[-1] = self._event_ch

            if len(dropped_desc) > 0:
                dropped = list(set(dropped_desc))
                logger.info("{0} annotation(s) will be dropped, such as {1}. "
                            .format(len(dropped), dropped[:5]))
                warn('Events like the following will be dropped entirely: {1},'
                     ' {0} in total'.format(len(dropped), dropped[:5]),
                     RuntimeWarning)

    def _create_event_ch(self, events, n_samples=None):
        """Create the event channel."""
        n_dropped = len(events[:, 0]) - len(set(events[:, 0]))
        if n_dropped > 0:
            warn(str(n_dropped) + " events will be dropped because they "
                 "occur on the same time sample as another event. "
                 "`mne.io.Raw` objects store events on an event channel, "
                 "which cannot represent two events on the same sample. "
                 "Please use `read_annotations_eeglab` and create events "
                 "using `events_from_annotations` to extract the original "
                 "event structure. Then, you can e.g. "
                 "subset the extracted events for constructing epochs.")
        if n_samples is None:
            n_samples = self.last_samp - self.first_samp + 1
        events = np.array(events, int)
        if events.ndim != 2 or events.shape[1] != 3:
            raise ValueError("[n_events x 3] shaped array required")
        # update events
        self._event_ch = _synthesize_stim_channel(events, n_samples)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        n_channels = self.info['nchan'] - (self._event_ch is not None)
        _read_segments_file(self, data, idx, fi, start, stop, cals, mult,
                            dtype=np.float32, trigger_ch=self._event_ch,
                            n_channels=n_channels)


class EpochsEEGLAB(BaseEpochs):
    r"""Epochs from EEGLAB .set file.

    Parameters
    ----------
    input_fname : str
        Path to the .set file. If the data is stored in a separate .fdt file,
        it is expected to be in the same folder as the .set file.
    events : str | array, shape (n_events, 3) | None
        Path to events file. If array, it is the events typically returned
        by the read_events function. If some events don't match the events
        of interest as specified by event_id, they will be marked as 'IGNORED'
        in the drop log. If None, it is constructed from the EEGLAB (.set) file
        with each unique event encoded with a different integer.
    event_id : int | list of int | dict | None
        The id of the event to consider. If dict,
        the keys can later be used to access associated events. Example:
        dict(auditory=1, visual=3). If int, a dict will be created with
        the id as string. If a list, all events with the IDs specified
        in the list are used. If None, the event_id is constructed from the
        EEGLAB (.set) file with each descriptions copied from ``eventtype``.
    tmin : float
        Start time before event.
    baseline : None or tuple of length 2 (default (None, 0))
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used.
        The baseline (a, b) includes both endpoints, i.e. all
        timepoints t such that a <= t <= b.
    reject : dict | None
        Rejection parameters based on peak-to-peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # V (EEG channels)
                          eog=250e-6 # V (EOG channels)
                          )
    flat : dict | None
        Rejection parameters based on flatness of signal.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg', and values
        are floats that set the minimum acceptable peak-to-peak amplitude.
        If flat is None then no rejection is done.
    reject_tmin : scalar | None
        Start of the time window used to reject epochs (with the default None,
        the window will start with tmin).
    reject_tmax : scalar | None
        End of the time window used to reject epochs (with the default None,
        the window will end with tmax).
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    eog : list | tuple | 'auto'
        Names or indices of channels that should be designated EOG channels.
        If 'auto', the channel names containing ``EOG`` or ``EYE`` are used.
        Defaults to empty tuple.
    verbose : bool | str | int | None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    uint16_codec : str | None
        If your \*.set file contains non-ascii characters, sometimes reading
        it may fail and give rise to error message stating that "buffer is
        too small". ``uint16_codec`` allows to specify what codec (for example:
        'latin1' or 'utf-8') should be used when reading character arrays and
        can therefore help you solve this problem.

    Notes
    -----
    .. versionadded:: 0.11.0

    See Also
    --------
    mne.Epochs : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, input_fname, events=None, event_id=None, tmin=0,
                 baseline=None, reject=None, flat=None, reject_tmin=None,
                 reject_tmax=None, montage=None, eog=(), verbose=None,
                 uint16_codec=None):  # noqa: D102
        eeg = _check_load_mat(input_fname, uint16_codec)

        if not ((events is None and event_id is None) or
                (events is not None and event_id is not None)):
            raise ValueError('Both `events` and `event_id` must be '
                             'None or not None')

        if events is None and eeg.trials > 1:
            # first extract the events and construct an event_id dict
            event_name, event_latencies, unique_ev = list(), list(), list()
            ev_idx = 0
            warn_multiple_events = False
            epochs = _bunchify(eeg.epoch)
            events = _bunchify(eeg.event)
            for ep in epochs:
                if isinstance(ep.eventtype, int):
                    ep.eventtype = str(ep.eventtype)
                if not isinstance(ep.eventtype, string_types):
                    event_type = '/'.join([str(et) for et in ep.eventtype])
                    event_name.append(event_type)
                    # store latency of only first event
                    event_latencies.append(events[ev_idx].latency)
                    ev_idx += len(ep.eventtype)
                    warn_multiple_events = True
                else:
                    event_type = ep.eventtype
                    event_name.append(ep.eventtype)
                    event_latencies.append(events[ev_idx].latency)
                    ev_idx += 1

                if event_type not in unique_ev:
                    unique_ev.append(event_type)

                # invent event dict but use id > 0 so you know its a trigger
                event_id = dict((ev, idx + 1) for idx, ev
                                in enumerate(unique_ev))

            # warn about multiple events in epoch if necessary
            if warn_multiple_events:
                warn('At least one epoch has multiple events. Only the latency'
                     ' of the first event will be retained.')

            # now fill up the event array
            events = np.zeros((eeg.trials, 3), dtype=int)
            for idx in range(0, eeg.trials):
                if idx == 0:
                    prev_stim = 0
                elif (idx > 0 and
                        event_latencies[idx] - event_latencies[idx - 1] == 1):
                    prev_stim = event_id[event_name[idx - 1]]
                events[idx, 0] = event_latencies[idx]
                events[idx, 1] = prev_stim
                events[idx, 2] = event_id[event_name[idx]]
        elif isinstance(events, string_types):
            events = read_events(events)

        logger.info('Extracting parameters from %s...' % input_fname)
        input_fname = op.abspath(input_fname)
        info = _get_info(eeg, montage, eog=eog)

        for key, val in event_id.items():
            if val not in events[:, 2]:
                raise ValueError('No matching events found for %s '
                                 '(event id %i)' % (key, val))

        if isinstance(eeg.data, string_types):
            basedir = op.dirname(input_fname)
            data_fname = op.join(basedir, eeg.data)
            _check_fname(data_fname)
            with open(data_fname, 'rb') as data_fid:
                data = np.fromfile(data_fid, dtype=np.float32)
                data = data.reshape((eeg.nbchan, eeg.pnts, eeg.trials),
                                    order="F")
        else:
            data = eeg.data

        if eeg.nbchan == 1 and len(data.shape) == 2:
            data = data[np.newaxis, :]
        data = data.transpose((2, 0, 1)).astype('double')
        data *= CAL
        assert data.shape == (eeg.trials, eeg.nbchan, eeg.pnts)
        tmin, tmax = eeg.xmin, eeg.xmax

        super(EpochsEEGLAB, self).__init__(
            info, data, events, event_id, tmin, tmax, baseline,
            reject=reject, flat=flat, reject_tmin=reject_tmin,
            reject_tmax=reject_tmax, filename=input_fname, verbose=verbose)

        # data are preloaded but _bad_dropped is not set so we do it here:
        self._bad_dropped = True
        logger.info('Ready.')


def _check_boundary(annot, event_id):
    if event_id is None:
        event_id = dict()
    if "boundary" in annot.description and "boundary" not in event_id:
        warn("The data contains 'boundary' events, indicating data "
             "discontinuities. Be cautious of filtering and epoching around "
             "these events.")


def _check_latencies(latencies):
    if (latencies < -1).any():
        raise ValueError('At least one event sample index is negative. Please'
                         ' check if EEG.event.sample values are correct.')
    if (latencies == -1).any():
        warn("At least one event has a sample index of -1. This usually is "
             "a consequence of how eeglab handles event latency after "
             "resampling - especially when you had a boundary event at the "
             "beginning of the file. Please make sure that the events at "
             "the very beginning of your EEGLAB file can be safely dropped "
             "(e.g., because they are boundary events).")


@deprecated('read_events_eeglab is deprecated from 0.17 and will be removed'
            ' in 0.18. Please use read_annotations and create events using'
            ' events_from_annotations.')
def read_events_eeglab(eeg, event_id=None, event_id_func='strip_to_integer',
                       uint16_codec=None):
    r"""Create events array from EEGLAB structure.

    An event array is constructed by looking up events in the
    event_id, trying to reduce them to their integer part otherwise, and
    entirely dropping them (with a warning) if this is impossible.
    Returns a 1x3 array of zeros if no events are found.

    Usually, the EEGLAB readers will automatically construct event information
    for you. However, the reader for continuous data stores event information
    in the stimulus channel, which can only code one event per time sample.
    Use this function if your EEGLAB file has events happening at the
    same time (sample) point to manually create an events array.

    Parameters
    ----------
    eeg : str | object
        The EEGLAB object from which events are read in.
        If str, path to the (EEGLAB) .set file.
        Else, the "EEG" field of a MATLAB EEGLAB structure as read in by
        scipy.io.loadmat.
    event_id : dict | None
        The ids of the events to consider. If None (default), an empty dict is
        used and ``event_id_func`` (see below) is called on every event value.
        If dict, the keys will be mapped to trigger values on the stimulus
        channel and only keys not in ``event_id`` will be handled by
        ``event_id_func``. Keys are case-sensitive.
        Example::

            {'SyncStatus': 1; 'Pulse Artifact': 3}

    event_id_func : None | str | callable
        What to do for events not found in ``event_id``. Must take one ``str``
        argument and return an ``int``. If string, must be 'strip-to-integer',
        in which case it defaults to stripping event codes such as "D128" or
        "S  1" of their non-integer parts and returns the integer.
        If the event is not in the ``event_id`` and calling ``event_id_func``
        on it results in a ``TypeError`` (e.g. if ``event_id_func`` is
        ``None``) or a ``ValueError``, the event is dropped.
    uint16_codec : str | None
        If your \*.set file contains non-ascii characters, sometimes reading
        it may fail and give rise to error message stating that "buffer is
        too small". ``uint16_codec`` allows to specify what codec (for example:
        'latin1' or 'utf-8') should be used when reading character arrays and
        can therefore help you solve this problem.

    Returns
    -------
    events : array, shape = (n_events, 3)
        All events that were found. The first column contains the event time
        in samples and the third column contains the event id. The center
        column is zero.

    See Also
    --------
    mne.find_events : Extract events from a stim channel. Note that stim
        channels can only code for one event per time point.
    """
    if event_id_func == 'strip_to_integer':
        event_id_func = _strip_to_integer
    if event_id is None:
        event_id = dict()

    if isinstance(eeg, string_types):
        from scipy import io
        eeg = io.loadmat(eeg, struct_as_record=False, squeeze_me=True,
                         uint16_codec=uint16_codec)['EEG']

    annotations = _read_annotations_eeglab(eeg)
    types = annotations.description
    latencies = annotations.onset * eeg.srate

    _check_boundary(annotations, event_id)

    if len(types) < 1:  # if there are 0 events, we can exit here
        logger.info('No events found, returning empty stim channel ...')
        return np.zeros((0, 3), dtype=int)

    _check_latencies(latencies)

    not_in_event_id = set(x for x in types if x not in event_id)
    not_purely_numeric = set(x for x in not_in_event_id if not x.isdigit())
    no_numbers = set([x for x in not_purely_numeric
                      if not any([d.isdigit() for d in x])])
    have_integers = set([x for x in not_purely_numeric
                         if x not in no_numbers])
    if len(not_purely_numeric) > 0:
        basewarn = "Events like the following will be dropped"
        n_no_numbers, n_have_integers = len(no_numbers), len(have_integers)
        if n_no_numbers > 0:
            no_num_warm = " entirely: {0}, {1} in total"
            warn(basewarn + no_num_warm.format(list(no_numbers)[:5],
                                               n_no_numbers))
        if n_have_integers > 0 and event_id_func is None:
            intwarn = (", but could be reduced to their integer part "
                       "instead with the default `event_id_func`: "
                       "{0}, {1} in total")
            warn(basewarn + intwarn.format(list(have_integers)[:5],
                                           n_have_integers))

    events = list()
    for tt, latency in zip(types, latencies):
        try:  # look up the event in event_id and if not, try event_id_func
            event_code = event_id[tt] if tt in event_id else event_id_func(tt)
            events.append([int(round(latency)), 0, event_code])
        except (ValueError, TypeError):  # if event_id_func fails
            pass  # We're already raising warnings above, so we just drop

    if len(events) < len(types):
        missing = len(types) - len(events)
        msg = ("{0}/{1} event codes could not be mapped to integers. Use "
               "the 'event_id' parameter to map such events manually.")
        warn(msg.format(missing, len(types)))
        if len(events) < 1:
            warn("As is, the trigger channel will consist entirely of zeros.")
            return np.zeros((0, 3), dtype=int)

    return np.asarray(events)


def _bunchify(items):
    if isinstance(items, dict):
        items = _dol_to_lod(items)
    if len(items) > 0 and isinstance(items[0], dict):
        items = [Bunch(**item) for item in items]
    return items


def _read_annotations_eeglab(eeg, uint16_codec=None):
    r"""Create Annotations from EEGLAB file.

    This function reads the event attribute from the EEGLAB
    structure and makes an :class:`mne.Annotations` object.

    Parameters
    ----------
    eeg : object | str
        'EEG' struct or the path to the (EEGLAB) .set file.
    uint16_codec : str | None
        If your \*.set file contains non-ascii characters, sometimes reading
        it may fail and give rise to error message stating that "buffer is
        too small". ``uint16_codec`` allows to specify what codec (for example:
        'latin1' or 'utf-8') should be used when reading character arrays and
        can therefore help you solve this problem.

    Returns
    -------
    annotations : instance of Annotations
        The annotations present in the file.
    """
    if isinstance(eeg, string_types):
        eeg = _check_load_mat(eeg, uint16_codec=uint16_codec)

    if not hasattr(eeg, 'event'):
        events = []
    elif isinstance(eeg.event, dict) and \
            np.array(eeg.event['latency']).ndim > 0:
        events = _dol_to_lod(eeg.event)
    elif not isinstance(eeg.event, (np.ndarray, list)):
        events = [eeg.event]
    else:
        events = eeg.event
    events = _bunchify(events)
    description = [str(event.type) for event in events]
    onset = [event.latency - 1 for event in events]
    duration = np.zeros(len(onset))
    if len(events) > 0 and hasattr(events[0], 'duration'):
        duration[:] = [event.duration for event in events]

    return Annotations(onset=np.array(onset) / eeg.srate,
                       duration=duration,
                       description=description,
                       orig_time=None)


def _dol_to_lod(dol):
    """Convert a dict of lists to a list of dicts."""
    return [dict((key, dol[key][ii]) for key in dol.keys())
            for ii in range(len(dol[list(dol.keys())[0]]))]


@deprecated('read_annotations_eeglab is deprecated from 0.17 and will be'
            ' removed in 0.18. Please use mne.read_annotations')
def read_annotations_eeglab(fname, uint16_codec=None):
    r"""Create Annotations from EEGLAB file.

    This function reads the event attribute from the EEGLAB
    structure and makes an :class:`mne.Annotations` object.

    Parameters
    ----------
    fname : str | object
        The path to the (EEGLAB) .set file.
    uint16_codec : str | None
        If your \*.set file contains non-ascii characters, sometimes reading
        it may fail and give rise to error message stating that "buffer is
        too small". ``uint16_codec`` allows to specify what codec (for example:
        'latin1' or 'utf-8') should be used when reading character arrays and
        can therefore help you solve this problem.

    Returns
    -------
    annotations : instance of Annotations
        The annotations present in the file.
    """
    eeg = _check_load_mat(fname, uint16_codec=uint16_codec)
    return _read_annotations_eeglab(eeg)


def _strip_to_integer(trigger):
    """Return only the integer part of a string."""
    return int("".join([x for x in trigger if x.isdigit()]))


def _event_id_func(trigger, event_id, event_id_func, dropped):
    """Mimic old behavior to be used with events_from_annotations."""
    if event_id is not None and trigger in event_id:
        return event_id[trigger]
    if event_id_func == 'strip_to_integer':
        trigger_new = "".join([x for x in trigger if x.isdigit()])
        if trigger_new.isdigit():
            return int(trigger_new)
        else:
            dropped.append(trigger)
            return None
    elif event_id_func is not None:
        return event_id_func(trigger)
