# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
import warnings

from ..utils import _read_segments_file, _find_channels
from ..constants import FIFF
from ..meas_info import _empty_info, create_info
from ..base import _BaseRaw, _check_update_montage
from ...utils import logger, verbose, check_version
from ...channels.montage import Montage
from ...epochs import _BaseEpochs
from ...event import read_events
from ...externals.six import string_types

# just fix the scaling for now, EEGLAB doesn't seem to provide this info
CAL = 1e-6


def _check_fname(fname):
    """Check if the file extension is valid.
    """
    fmt = str(op.splitext(fname)[-1])
    if fmt == '.dat':
        raise NotImplementedError(
            'Old data format .dat detected. Please update your EEGLAB '
            'version and resave the data in .fdt format')
    elif fmt != '.fdt':
        raise IOError('Expected .fdt file format. Found %s format' % fmt)


def _check_mat_struct(fname):
    """Check if the mat struct contains 'EEG'.
    """
    if not check_version('scipy', '0.12'):
        raise RuntimeError('scipy >= 0.12 must be installed for reading EEGLAB'
                           ' files.')
    from scipy import io
    mat = io.whosmat(fname, struct_as_record=False,
                     squeeze_me=True)
    if 'ALLEEG' in mat[0]:
        raise NotImplementedError(
            'Loading an ALLEEG array is not supported. Please contact'
            'mne-python developers for more information.')
    elif 'EEG' not in mat[0]:
        msg = ('Unknown array in the .set file.')
        raise ValueError(msg)


def _to_loc(ll):
    """Check if location exists.
    """
    if isinstance(ll, (int, float)) or len(ll) > 0:
        return ll
    else:
        return 0.


def _get_info(eeg, montage, eog=()):
    """Get measurement info.
    """
    info = _empty_info(sfreq=eeg.srate)
    info['nchan'] = eeg.nbchan

    # add the ch_names and info['chs'][idx]['loc']
    path = None
    if len(eeg.chanlocs) > 0:
        ch_names, pos = list(), list()
        kind = 'user_defined'
        selection = np.arange(len(eeg.chanlocs))
        locs_available = True
        for chanloc in eeg.chanlocs:
            ch_names.append(chanloc.labels)
            loc_x = _to_loc(chanloc.X)
            loc_y = _to_loc(chanloc.Y)
            loc_z = _to_loc(chanloc.Z)
            locs = np.r_[-loc_y, loc_x, loc_z]
            if np.unique(locs).size == 1:
                locs_available = False
            pos.append(locs)
        if locs_available:
            montage = Montage(np.array(pos), ch_names, kind, selection)
    elif isinstance(montage, string_types):
        path = op.dirname(montage)

    if montage is None:
        info = create_info(ch_names, eeg.srate, ch_types='eeg')
    else:
        _check_update_montage(info, montage, path=path,
                              update_ch_names=True)

    info['buffer_size_sec'] = 1.  # reasonable default
    # update the info dict

    if eog == 'auto':
        eog = _find_channels(ch_names)

    for idx, ch in enumerate(info['chs']):
        ch['cal'] = CAL
        if ch['ch_name'] in eog or idx in eog:
            ch['coil_type'] = FIFF.FIFFV_COIL_NONE
            ch['kind'] = FIFF.FIFFV_EOG_CH

    return info


def read_raw_eeglab(input_fname, montage=None, preload=False, eog=(),
                    verbose=None):
    """Read an EEGLAB .set file

    Parameters
    ----------
    input_fname : str
        Path to the .set file. If the data is stored in a separate .fdt file,
        it is expected to be in the same folder as the .set file.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory). Note that
        preload=False will be effective only if the data is stored in a
        separate binary file.
    eog : list | tuple | 'auto'
        Names or indices of channels that should be designated
        EOG channels. If 'auto', the channel names containing
        ``EOG`` or ``EYE`` are used. Defaults to empty tuple.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

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
                     eog=eog, verbose=verbose)


def read_epochs_eeglab(input_fname, events=None, event_id=None, montage=None,
                       eog=(), verbose=None):
    """Reader function for EEGLAB epochs files

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
        the keys can later be used to acces associated events. Example:
        dict(auditory=1, visual=3). If int, a dict will be created with
        the id as string. If a list, all events with the IDs specified
        in the list are used. If None, the event_id is constructed from the
        EEGLAB (.set) file with each descriptions copied from `eventtype`.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    eog : list | tuple | 'auto'
        Names or indices of channels that should be designated
        EOG channels. If 'auto', the channel names containing
        ``EOG`` or ``EYE`` are used. Defaults to empty tuple.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

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
                          event_id=event_id, montage=montage, verbose=verbose)
    return epochs


class RawEEGLAB(_BaseRaw):
    """Raw object from EEGLAB .set file.

    Parameters
    ----------
    input_fname : str
        Path to the .set file. If the data is stored in a separate .fdt file,
        it is expected to be in the same folder as the .set file.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    eog : list | tuple | 'auto'
        Names or indices of channels that should be designated
        EOG channels. If 'auto', the channel names containing
        ``EOG`` or ``EYE`` are used. Defaults to empty tuple.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

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
    def __init__(self, input_fname, montage, preload=False, eog=(),
                 verbose=None):
        """Read EEGLAB .set file.
        """
        from scipy import io
        basedir = op.dirname(input_fname)
        _check_mat_struct(input_fname)
        eeg = io.loadmat(input_fname, struct_as_record=False,
                         squeeze_me=True)['EEG']
        if eeg.trials != 1:
            raise TypeError('The number of trials is %d. It must be 1 for raw'
                            ' files. Please use `mne.io.read_epochs_eeglab` if'
                            ' the .set file contains epochs.' % eeg.trials)

        last_samps = [eeg.pnts - 1]
        info = _get_info(eeg, montage, eog=eog)
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
                warnings.warn('Data will be preloaded. preload=False or a'
                              ' string preload is not supported when the data'
                              ' is stored in the .set file')
            # can't be done in standard way with preload=True because of
            # different reading path (.set file)
            data = eeg.data.reshape(eeg.nbchan, -1, order='F')
            data = data.astype(np.double)
            data *= CAL
            super(RawEEGLAB, self).__init__(
                info, data, last_samps=last_samps, orig_format='double',
                verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data"""
        _read_segments_file(self, data, idx, fi, start, stop, cals, mult,
                            dtype=np.float32)


class EpochsEEGLAB(_BaseEpochs):
    """Epochs from EEGLAB .set file

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
        the keys can later be used to acces associated events. Example:
        dict(auditory=1, visual=3). If int, a dict will be created with
        the id as string. If a list, all events with the IDs specified
        in the list are used. If None, the event_id is constructed from the
        EEGLAB (.set) file with each descriptions copied from `eventtype`.
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
                          eeg=40e-6, # uV (EEG channels)
                          eog=250e-6 # uV (EOG channels)
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
        Names or indices of channels that should be designated
        EOG channels. If 'auto', the channel names containing
        ``EOG`` or ``EYE`` are used. Defaults to empty tuple.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Notes
    -----
    .. versionadded:: 0.11.0

    See Also
    --------
    mne.Epochs : Documentation of attribute and methods.
    """
    @verbose
    def __init__(self, input_fname, events=None, event_id=None, tmin=0,
                 baseline=None,  reject=None, flat=None, reject_tmin=None,
                 reject_tmax=None, montage=None, eog=(), verbose=None):
        from scipy import io
        _check_mat_struct(input_fname)
        eeg = io.loadmat(input_fname, struct_as_record=False,
                         squeeze_me=True)['EEG']

        if not ((events is None and event_id is None) or
                (events is not None and event_id is not None)):
            raise ValueError('Both `events` and `event_id` must be '
                             'None or not None')

        if events is None and eeg.trials > 1:
            # first extract the events and construct an event_id dict
            event_name, event_latencies, unique_ev = list(), list(), list()
            ev_idx = 0
            for ep in eeg.epoch:
                if not isinstance(ep.eventtype, string_types):
                    event_type = '/'.join(ep.eventtype.tolist())
                    event_name.append(event_type)
                    # store latency of only first event
                    event_latencies.append(eeg.event[ev_idx].latency)
                    ev_idx += len(ep.eventtype)
                    warnings.warn('An epoch has multiple events. '
                                  'Only the latency of first event will be '
                                  'retained.')
                else:
                    event_type = ep.eventtype
                    event_name.append(ep.eventtype)
                    event_latencies.append(eeg.event[ev_idx].latency)
                    ev_idx += 1

                if event_type not in unique_ev:
                    unique_ev.append(event_type)

                # invent event dict but use id > 0 so you know its a trigger
                event_id = dict((ev, idx + 1) for idx, ev
                                in enumerate(unique_ev))
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

        self._filename = input_fname
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
        data = data.transpose((2, 0, 1)).astype('double')
        data *= CAL
        assert data.shape == (eeg.trials, eeg.nbchan, eeg.pnts)
        tmin, tmax = eeg.xmin, eeg.xmax

        super(EpochsEEGLAB, self).__init__(
            info, data, events, event_id, tmin, tmax, baseline,
            reject=reject, flat=flat, reject_tmin=reject_tmin,
            reject_tmax=reject_tmax, add_eeg_ref=False, verbose=verbose)
        logger.info('Ready.')
