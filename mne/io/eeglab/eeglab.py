# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause

import os.path as op

import numpy as np

from ..pick import _PICK_TYPES_KEYS
from ..utils import _read_segments_file, _find_channels
from ..constants import FIFF
from ..meas_info import create_info
from ..base import BaseRaw
from ...utils import logger, verbose, warn, fill_doc, Bunch, _check_fname
from ...channels import make_dig_montage
from ...epochs import BaseEpochs
from ...event import read_events
from ...annotations import Annotations, read_annotations

# just fix the scaling for now, EEGLAB doesn't seem to provide this info
CAL = 1e-6


def _check_eeglab_fname(fname, dataname):
    """Check whether the filename is valid.

    Check if the file extension is ``.fdt`` (older ``.dat`` being invalid) or
    whether the ``EEG.data`` filename exists. If ``EEG.data`` file is absent
    the set file name with .set changed to .fdt is checked.
    """
    fmt = str(op.splitext(dataname)[-1])
    if fmt == '.dat':
        raise NotImplementedError(
            'Old data format .dat detected. Please update your EEGLAB '
            'version and resave the data in .fdt format')
    elif fmt != '.fdt':
        raise IOError('Expected .fdt file format. Found %s format' % fmt)

    basedir = op.dirname(fname)
    data_fname = op.join(basedir, dataname)
    if not op.exists(data_fname):
        fdt_from_set_fname = op.splitext(fname)[0] + '.fdt'
        if op.exists(fdt_from_set_fname):
            data_fname = fdt_from_set_fname
            msg = ('Data file name in EEG.data ({}) is incorrect, the file '
                   'name must have changed on disk, using the correct file '
                   'name ({}).')
            warn(msg.format(dataname, op.basename(fdt_from_set_fname)))
        elif not data_fname == fdt_from_set_fname:
            msg = 'Could not find the .fdt data file, tried {} and {}.'
            raise FileNotFoundError(msg.format(data_fname, fdt_from_set_fname))
    return data_fname


def _check_load_mat(fname, uint16_codec):
    """Check if the mat struct contains 'EEG'."""
    from ...externals.pymatreader import read_mat
    eeg = read_mat(fname, uint16_codec=uint16_codec)
    if 'ALLEEG' in eeg:
        raise NotImplementedError(
            'Loading an ALLEEG array is not supported. Please contact'
            'mne-python developers for more information.')
    if 'EEG' in eeg:  # fields are contained in EEG structure
        eeg = eeg['EEG']
    eeg = eeg.get('EEG', eeg)  # handle nested EEG structure
    eeg = Bunch(**eeg)
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


def _eeg_has_montage_information(eeg):
    try:
        from scipy.io.matlab import mat_struct
    except ImportError:  # SciPy < 1.8
        from scipy.io.matlab.mio5_params import mat_struct
    if not len(eeg.chanlocs):
        has_pos = False
    else:
        pos_fields = ['X', 'Y', 'Z']
        if isinstance(eeg.chanlocs[0], mat_struct):
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
            has_pos = False  # unknown (sometimes we get [0, 0])

    return has_pos


def _get_montage_information(eeg, get_pos):
    """Get channel name, type and montage information from ['chanlocs']."""
    ch_names, ch_types, pos_ch_names, pos = list(), list(), list(), list()
    unknown_types = dict()
    for chanloc in eeg.chanlocs:
        # channel name
        ch_names.append(chanloc['labels'])

        # channel type
        ch_type = 'eeg'
        try_type = chanloc.get('type', None)
        if isinstance(try_type, str):
            try_type = try_type.strip().lower()
            if try_type in _PICK_TYPES_KEYS:
                ch_type = try_type
            else:
                if try_type in unknown_types:
                    unknown_types[try_type].append(chanloc['labels'])
                else:
                    unknown_types[try_type] = [chanloc['labels']]
        ch_types.append(ch_type)

        # channel loc
        if get_pos:
            loc_x = _to_loc(chanloc['X'])
            loc_y = _to_loc(chanloc['Y'])
            loc_z = _to_loc(chanloc['Z'])
            locs = np.r_[-loc_y, loc_x, loc_z]
            if not np.any(np.isnan(locs)):
                pos_ch_names.append(chanloc['labels'])
                pos.append(locs)

    # warn if unknown types were provided
    if len(unknown_types):
        warn('Unknown types found, setting as type EEG:\n' +
             '\n'.join([f'{key}: {sorted(unknown_types[key])}'
                        for key in sorted(unknown_types)]))

    if pos_ch_names:
        montage = make_dig_montage(
            ch_pos=dict(zip(ch_names, np.array(pos))),
            coord_frame='head')
    else:
        montage = None

    return ch_names, ch_types, montage


def _get_info(eeg, eog=()):
    """Get measurement info."""
    # add the ch_names and info['chs'][idx]['loc']
    if not isinstance(eeg.chanlocs, np.ndarray) and eeg.nbchan == 1:
        eeg.chanlocs = [eeg.chanlocs]

    if isinstance(eeg.chanlocs, dict):
        eeg.chanlocs = _dol_to_lod(eeg.chanlocs)

    eeg_has_ch_names_info = len(eeg.chanlocs) > 0

    if eeg_has_ch_names_info:
        has_pos = _eeg_has_montage_information(eeg)
        ch_names, ch_types, eeg_montage = \
            _get_montage_information(eeg, has_pos)
        update_ch_names = False
    else:  # if eeg.chanlocs is empty, we still need default chan names
        ch_names = ["EEG %03d" % ii for ii in range(eeg.nbchan)]
        ch_types = 'eeg'
        eeg_montage = None
        update_ch_names = True

    info = create_info(ch_names, sfreq=eeg.srate, ch_types=ch_types)

    eog = _find_channels(ch_names, ch_type='EOG') if eog == 'auto' else eog
    for idx, ch in enumerate(info['chs']):
        ch['cal'] = CAL
        if ch['ch_name'] in eog or idx in eog:
            ch['coil_type'] = FIFF.FIFFV_COIL_NONE
            ch['kind'] = FIFF.FIFFV_EOG_CH

    return info, eeg_montage, update_ch_names


def _set_dig_montage_in_init(self, montage):
    """Set EEG sensor configuration and head digitization from when init.

    This is done from the information within fname when
    read_raw_eeglab(fname) or read_epochs_eeglab(fname).
    """
    if montage is None:
        self.set_montage(None)
    else:
        missing_channels = set(self.ch_names) - set(montage.ch_names)
        ch_pos = dict(zip(
            list(missing_channels),
            np.full((len(missing_channels), 3), np.nan)
        ))
        self.set_montage(
            montage + make_dig_montage(ch_pos=ch_pos, coord_frame='head')
        )


@fill_doc
def read_raw_eeglab(input_fname, eog=(), preload=False,
                    uint16_codec=None, verbose=None):
    r"""Read an EEGLAB .set file.

    Parameters
    ----------
    input_fname : str
        Path to the .set file. If the data is stored in a separate .fdt file,
        it is expected to be in the same folder as the .set file.
    eog : list | tuple | 'auto'
        Names or indices of channels that should be designated EOG channels.
        If 'auto', the channel names containing ``EOG`` or ``EYE`` are used.
        Defaults to empty tuple.
    %(preload)s
        Note that preload=False will be effective only if the data is stored
        in a separate binary file.
    uint16_codec : str | None
        If your \*.set file contains non-ascii characters, sometimes reading
        it may fail and give rise to error message stating that "buffer is
        too small". ``uint16_codec`` allows to specify what codec (for example:
        'latin1' or 'utf-8') should be used when reading character arrays and
        can therefore help you solve this problem.
    %(verbose)s

    Returns
    -------
    raw : instance of RawEEGLAB
        A Raw object containing EEGLAB .set data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    .. versionadded:: 0.11.0
    """
    return RawEEGLAB(input_fname=input_fname, preload=preload,
                     eog=eog, verbose=verbose, uint16_codec=uint16_codec)


@fill_doc
def read_epochs_eeglab(input_fname, events=None, event_id=None,
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
        EEGLAB (.set) file with each descriptions copied from ``eventtype``.
    eog : list | tuple | 'auto'
        Names or indices of channels that should be designated EOG channels.
        If 'auto', the channel names containing ``EOG`` or ``EYE`` are used.
        Defaults to empty tuple.
    %(verbose)s
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

    See Also
    --------
    mne.Epochs : Documentation of attribute and methods.

    Notes
    -----
    .. versionadded:: 0.11.0
    """
    epochs = EpochsEEGLAB(input_fname=input_fname, events=events, eog=eog,
                          event_id=event_id, verbose=verbose,
                          uint16_codec=uint16_codec)
    return epochs


@fill_doc
class RawEEGLAB(BaseRaw):
    r"""Raw object from EEGLAB .set file.

    Parameters
    ----------
    input_fname : str
        Path to the .set file. If the data is stored in a separate .fdt file,
        it is expected to be in the same folder as the .set file.
    eog : list | tuple | 'auto'
        Names or indices of channels that should be designated EOG channels.
        If 'auto', the channel names containing ``EOG`` or ``EYE`` are used.
        Defaults to empty tuple.
    %(preload)s
        Note that preload=False will be effective only if the data is stored
        in a separate binary file.
    uint16_codec : str | None
        If your \*.set file contains non-ascii characters, sometimes reading
        it may fail and give rise to error message stating that "buffer is
        too small". ``uint16_codec`` allows to specify what codec (for example:
        'latin1' or 'utf-8') should be used when reading character arrays and
        can therefore help you solve this problem.
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    .. versionadded:: 0.11.0
    """

    @verbose
    def __init__(self, input_fname, eog=(),
                 preload=False, uint16_codec=None, verbose=None):  # noqa: D102
        input_fname = _check_fname(input_fname, 'read', True, 'input_fname')
        eeg = _check_load_mat(input_fname, uint16_codec)
        if eeg.trials != 1:
            raise TypeError('The number of trials is %d. It must be 1 for raw'
                            ' files. Please use `mne.io.read_epochs_eeglab` if'
                            ' the .set file contains epochs.' % eeg.trials)

        last_samps = [eeg.pnts - 1]
        info, eeg_montage, _ = _get_info(eeg, eog=eog)

        # read the data
        if isinstance(eeg.data, str):
            data_fname = _check_eeglab_fname(input_fname, eeg.data)
            logger.info('Reading %s' % data_fname)

            super(RawEEGLAB, self).__init__(
                info, preload, filenames=[data_fname], last_samps=last_samps,
                orig_format='double', verbose=verbose)
        else:
            if preload is False or isinstance(preload, str):
                warn('Data will be preloaded. preload=False or a string '
                     'preload is not supported when the data is stored in '
                     'the .set file')
            # can't be done in standard way with preload=True because of
            # different reading path (.set file)
            if eeg.nbchan == 1 and len(eeg.data.shape) == 1:
                n_chan, n_times = [1, eeg.data.shape[0]]
            else:
                n_chan, n_times = eeg.data.shape
            data = np.empty((n_chan, n_times), dtype=float)
            data[:n_chan] = eeg.data
            data *= CAL
            super(RawEEGLAB, self).__init__(
                info, data, filenames=[input_fname], last_samps=last_samps,
                orig_format='double', verbose=verbose)

        # create event_ch from annotations
        annot = read_annotations(input_fname)
        self.set_annotations(annot)
        _check_boundary(annot, None)

        _set_dig_montage_in_init(self, eeg_montage)

        latencies = np.round(annot.onset * self.info['sfreq'])
        _check_latencies(latencies)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        _read_segments_file(
            self, data, idx, fi, start, stop, cals, mult, dtype='<f4')


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
    eog : list | tuple | 'auto'
        Names or indices of channels that should be designated EOG channels.
        If 'auto', the channel names containing ``EOG`` or ``EYE`` are used.
        Defaults to empty tuple.
    %(verbose)s
    uint16_codec : str | None
        If your \*.set file contains non-ascii characters, sometimes reading
        it may fail and give rise to error message stating that "buffer is
        too small". ``uint16_codec`` allows to specify what codec (for example:
        'latin1' or 'utf-8') should be used when reading character arrays and
        can therefore help you solve this problem.

    See Also
    --------
    mne.Epochs : Documentation of attribute and methods.

    Notes
    -----
    .. versionadded:: 0.11.0
    """

    @verbose
    def __init__(self, input_fname, events=None, event_id=None, tmin=0,
                 baseline=None, reject=None, flat=None, reject_tmin=None,
                 reject_tmax=None, eog=(), verbose=None,
                 uint16_codec=None):  # noqa: D102
        input_fname = _check_fname(fname=input_fname, must_exist=True,
                                   overwrite='read')
        eeg = _check_load_mat(input_fname, uint16_codec)

        if not ((events is None and event_id is None) or
                (events is not None and event_id is not None)):
            raise ValueError('Both `events` and `event_id` must be '
                             'None or not None')

        if eeg.trials <= 1:
            raise ValueError("The file does not seem to contain epochs "
                             "(trials less than 2). "
                             "You should try using read_raw_eeglab function.")

        if events is None and eeg.trials > 1:
            # first extract the events and construct an event_id dict
            event_name, event_latencies, unique_ev = list(), list(), list()
            ev_idx = 0
            warn_multiple_events = False
            epochs = _bunchify(eeg.epoch)
            events = _bunchify(eeg.event)
            for ep in epochs:
                if isinstance(ep.eventtype, (int, float)):
                    ep.eventtype = str(ep.eventtype)
                if not isinstance(ep.eventtype, str):
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
                event_id = {ev: idx + 1 for idx, ev in enumerate(unique_ev)}

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
        elif isinstance(events, str):
            events = read_events(events)

        logger.info('Extracting parameters from %s...' % input_fname)
        info, eeg_montage, _ = _get_info(eeg, eog=eog)

        for key, val in event_id.items():
            if val not in events[:, 2]:
                raise ValueError('No matching events found for %s '
                                 '(event id %i)' % (key, val))

        if isinstance(eeg.data, str):
            data_fname = _check_eeglab_fname(input_fname, eeg.data)
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

        _set_dig_montage_in_init(self, eeg_montage)

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
    if isinstance(eeg, str):
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
        for idx, event in enumerate(events):
            # empty duration fields are read as empty arrays
            is_empty_array = (isinstance(event.duration, np.ndarray)
                              and len(event.duration) == 0)
            duration[idx] = np.nan if is_empty_array else event.duration

    return Annotations(onset=np.array(onset) / eeg.srate,
                       duration=duration / eeg.srate,
                       description=description,
                       orig_time=None)


def _dol_to_lod(dol):
    """Convert a dict of lists to a list of dicts."""
    return [{key: dol[key][ii] for key in dol.keys()}
            for ii in range(len(dol[list(dol.keys())[0]]))]
