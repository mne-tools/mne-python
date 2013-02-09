"""IO with fif files containing events
"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
from os.path import splitext

import logging
logger = logging.getLogger('mne')

from .fiff.constants import FIFF
from .fiff.tree import dir_tree_find
from .fiff.tag import read_tag
from .fiff.open import fiff_open
from .fiff.write import write_int, start_block, start_file, end_block, end_file
from .fiff.pick import pick_channels
from .utils import get_config
from . import verbose


def pick_events(events, include=None, exclude=None):
    """Select some events

    Parameters
    ----------
    include : int | list | None
        A event id to include or a list of them.
        If None all events are included.
    exclude : int | list | None
        A event id to exclude or a list of them.
        If None no event is excluded. If include is not None
        the exclude parameter is ignored.

    Returns
    -------
    events : array, shape (n_events, 3)
        The list of events
    """
    if include is not None:
        if not isinstance(include, list):
            include = [include]
        mask = np.zeros(len(events), dtype=np.bool)
        for e in include:
            mask = np.logical_or(mask, events[:, 2] == e)
        events = events[mask]
    elif exclude is not None:
        if not isinstance(exclude, list):
            exclude = [exclude]
        mask = np.ones(len(events), dtype=np.bool)
        for e in exclude:
            mask = np.logical_and(mask, events[:, 2] != e)
        events = events[mask]
    else:
        events = np.copy(events)

    if len(events) == 0:
        raise RuntimeError("No events found")

    return events


def define_target_events(events, reference_id, target_id, sfreq, tmin, tmax,
                         new_id=None, fill_na=None):
    """ Define new events by co-occurrence of existing events

    This function can be used to evaluate events depending on the
    temporal lag to another event. For example, this can be used to
    analyze evoked responses which were followed by a button press wihtin
    a defined time window.

    Parameters
    ----------

    events : ndarray
        Array as returned by mne.find_events.
    reference_id : int
        The reference event. The event defining the epoch of interest.
    target_id : int
        The target event. The event co-occurring in within a certain time
        window around the reference event.
    sfreq : float
        The sampling frequency of the data.
    tmin : float
        The lower limit in seconds from the target event.
    tmax : float
        The upper limit border in seconds from the target event.
    new_id : int
        new_id for the new event
    fill_na : int | None
        Fill event to be inserted if target is not available within the time
        window specified. If None, the 'null' events will be dropped.

    Returns
    -------
    new_events : ndarray
        The new defined events
    lag : ndarray
        time lag between reference and target in milliseconds.
    """

    if new_id is None:
        new_id = reference_id

    tsample = 1e3 / sfreq
    imin = int(tmin * sfreq)
    imax = int(tmax * sfreq)

    new_events = []
    lag = []
    for event in events.copy().astype('f8'):
        if event[2] == reference_id:
            lower = event[0] + imin
            upper = event[0] + imax
            res = events[(events[:, 0] > lower) &
                         (events[:, 0] < upper) & (events[:, 2] == target_id)]
            if res.any():
                lag += [event[0] - res[0][0]]
                event[2] = new_id
                new_events += [event]
            elif fill_na is not None:
                event[2] = fill_na
                new_events += [event]
                lag += [fill_na]

    new_events = np.array(new_events)

    lag = np.abs(lag, dtype='f8')
    if lag.any():
        lag[lag != fill_na] *= tsample
    else:
        lag = np.array([])

    return new_events if new_events.any() else np.array([]), lag


def _read_events_fif(fid, tree):
    #   Find the desired block
    events = dir_tree_find(tree, FIFF.FIFFB_MNE_EVENTS)

    if len(events) == 0:
        fid.close()
        raise ValueError('Could not find event data')

    events = events[0]

    for d in events['directory']:
        kind = d.kind
        pos = d.pos
        if kind == FIFF.FIFF_MNE_EVENT_LIST:
            tag = read_tag(fid, pos)
            event_list = tag.data
            break
    else:
        raise ValueError('Could not find any events')

    mappings = dir_tree_find(tree, FIFF.FIFFB_MNE_EVENTS)
    mappings = mappings[0]

    for d in mappings['directory']:
        kind = d.kind
        pos = d.pos
        if kind == FIFF.FIFF_DESCRIPTION:
            tag = read_tag(fid, pos)
            mappings = tag.data
            break
    else:
        mappings = None

    if mappings is not None:
        m_ = (m.split(':') for m in mappings.split(';'))
        mappings = dict((k, int(v)) for k, v in m_)
    event_list = event_list.reshape(len(event_list) / 3, 3)
    return event_list, mappings


def read_events(filename, include=None, exclude=None):
    """Reads events from fif or text file

    Parameters
    ----------
    filename : string
        Name of the input file.
        If the extension is .fif, events are read assuming
        the file is in FIF format, otherwise (e.g., .eve,
        .lst, .txt) events are read as coming from text.
        Note that new format event files do not contain
        the "time" column (used to be the second column).
    include : int | list | None
        A event id to include or a list of them.
        If None all events are included.
    exclude : int | list | None
        A event id to exclude or a list of them.
        If None no event is excluded. If include is not None
        the exclude parameter is ignored.

    Returns
    -------
    events: array, shape (n_events, 3)
        The list of events

    Notes
    -----
    This function will discard the offset line (i.e., first line with zero
    event number) if it is present in a text file.
    """
    ext = splitext(filename)[1].lower()
    if ext == '.fif' or ext == '.gz':
        fid, tree, _ = fiff_open(filename)
        event_list, _ = _read_events_fif(fid, tree)
        fid.close()
    else:
        #  Have to read this in as float64 then convert because old style
        #  eve/lst files had a second float column that will raise errors
        lines = np.loadtxt(filename, dtype=np.float64).astype(np.uint32)
        if len(lines) == 0:
            raise ValueError('No text lines found')

        if lines.ndim == 1:  # Special case for only one event
            lines = lines[np.newaxis, :]

        if len(lines[0]) == 4:  # Old format eve/lst
            goods = [0, 2, 3]   # Omit "time" variable
        elif len(lines[0]) == 3:
            goods = [0, 1, 2]
        else:
            raise ValueError('Unknown number of columns in event text file')

        event_list = lines[:, goods]
        if event_list.shape[0] > 0 and event_list[0, 2] == 0:
            event_list = event_list[1:]

    event_list = pick_events(event_list, include, exclude)
    return event_list


def write_events(filename, event_list):
    """Write events to file

    Parameters
    ----------
    filename : string
        Name of the output file.
        If the extension is .fif, events are written in
        binary FIF format, otherwise (e.g., .eve, .lst,
        .txt) events are written as plain text.
        Note that new format event files do not contain
        the "time" column (used to be the second column).

    event_list : array, shape (n_events, 3)
        The list of events
    """
    ext = splitext(filename)[1].lower()
    if ext == '.fif' or ext == '.gz':
        #   Start writing...
        fid = start_file(filename)

        start_block(fid, FIFF.FIFFB_MNE_EVENTS)
        write_int(fid, FIFF.FIFF_MNE_EVENT_LIST, event_list.T)
        end_block(fid, FIFF.FIFFB_MNE_EVENTS)

        end_file(fid)
    else:
        f = open(filename, 'w')
        [f.write('%6d %6d %3d\n' % tuple(e)) for e in event_list]
        f.close()


def find_steps(raw, first_samp=0, pad_start=None, pad_stop=None,
               stim_channel=None):
    """Find all steps in data from a stim channel

    Parameters
    ----------
    raw : Raw object
        The raw data.
    first_samp : int
        The index of the first sample (if not 0)
    edges : None | int
        Value to assume outside of data.
    pad_start, pad_stop : None | int
        Values to assume outside of the stim channel (e.g., if pad_start=0 and
        the stim channel starts with value 5, an event of [0, 0, 5] will be
        inserted at the beginning). With None, no steps will be inserted.
    stim_channel : None | string | list of string
        Name of the stim channel or all the stim channels
        affected by the trigger. If None, the config variables
        'MNE_STIM_CHANNEL', 'MNE_STIM_CHANNEL_1', 'MNE_STIM_CHANNEL_2',
        etc. are read. If these are not found, it will default to
        'STI 014'.

    Returns
    -------
    events : array, shape = (n_samples, 3)
        For each step in the stim channel the values [sample, v_from, v_to].
        The first column contains the event time in samples (the first sample
        with the new value). The second column contains the stim channel value
        before the step, and the third column contains value after the step.

    See Also
    --------
    find_events : More sophisticated options for finding events in a Raw file.

    """
    # pull stim channel from config if necessary
    stim_channel = _get_stim_channel(stim_channel)

    pick = pick_channels(raw.info['ch_names'], include=stim_channel)
    if len(pick) == 0:
        raise ValueError('No stim channel found to extract event triggers.')
    data, _ = raw[pick, :]
    if np.any(data < 0):
        logger.warn('Trigger channel contains negative values. '
                    'Taking absolute value.')
        data = np.abs(data)  # make sure trig channel is positive
    data = data.astype(np.int)

    changed = np.diff(data, axis=1) != 0
    idx = np.where(np.all(changed, axis=0))[0]
    pre_step = data[0, idx]
    idx += 1
    post_step = data[0, idx]
    idx += int(first_samp)
    events = np.c_[idx, pre_step, post_step]

    if pad_start is not None:
        v = events[0, 1]
        if v != pad_start:
            events = np.insert(events, 0, [[0, pad_start, v]], axis=0)

    if pad_stop is not None:
        v = events[-1, 2]
        if v != pad_stop:
            last_idx = len(data[0]) + int(first_samp)
            events = np.append(events, [[last_idx, v, pad_stop]], axis=0)

    return events


@verbose
def find_events(raw, stim_channel=None, verbose=None, detect='onset',
                consecutive='increasing', min_duration=0):
    """Find events from raw file

    Parameters
    ----------
    raw : Raw object
        The raw data.
    stim_channel : None | string | list of string
        Name of the stim channel or all the stim channels
        affected by the trigger. If None, the config variables
        'MNE_STIM_CHANNEL', 'MNE_STIM_CHANNEL_1', 'MNE_STIM_CHANNEL_2',
        etc. are read. If these are not found, it will default to
        'STI 014'.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    detect : 'onset' | 'offset'
        Whether to report when events start or when events end.
    consecutive : bool | 'increasing'
        If True, consider instances where the value of the events
        channel changes without first returning to zero as multiple
        events. If False, report only instances where the value of the
        events channel changes from/to zero. If 'increasing', report
        adjacent events only when the second event code is greater than
        the first.
    min_duration : float
        The minimum duration of a change in the events channel required
        to consider it as an event.

    Returns
    -------
    events : array, shape = (n_events, 3)
        All events that were found. The first column contains the event time
        in samples and the third column contains the stim channel value of the
        event.

    Examples
    --------
    Consider data with a stim channel that looks like: [0, 32, 32, 33, 32, 0]

    By default, find_events returns all samples at which the value of the
    stim channel increases:

        >>> print(find_events(raw)) # doctest: +SKIP
        [[ 1  0 32]
         [ 3  0 33]]

    If consecutive is False, find_events only returns the samples at which
    the stim channel changes from zero to a non-zero value:

        >>> print(find_events(raw, consecutive=False)) # doctest: +SKIP
        [[ 1  0 32]]

    If consecutive is True, find_events returns samples at which the
    event changes, regardless of whether it first returns to zero:

        >>> print(find_events(raw, consecutive=True)) # doctest: +SKIP
        [[ 1  0 32]
         [ 3  0 33]
         [ 4  0 32]]

    If detect is 'offset', find_events returns the samples at which a new
    event starts, or the stim channel changes to zero (and the final sample
    if it is non-zero):

        >>> print(find_events(raw, consecutive=True, # doctest: +SKIP
        ...                   detect='offset'))
        [[ 2  0 32]
         [ 3  0 33]
         [ 4  0 32]]

    To ignore spurious events, it is also possible to specify a minimum
    event duration. Assuming our events channel has a sample rate of
    1000 Hz:

        >>> print(find_events(raw, consecutive=True, # doctest: +SKIP
        ...                   min_duration=0.002))
        [[ 1  0 32]]


    See Also
    --------
    find_steps : Find all the steps in the stim channel.

    """
    events = find_steps(raw, first_samp=raw.first_samp, pad_stop=0,
                        stim_channel=stim_channel)

    # Determine event onsets and offsets
    if consecutive == 'increasing':
        onsets = (events[:, 2] > events[:, 1])
        offsets = np.logical_and(np.logical_or(onsets, (events[:, 2] == 0)),
                                 (events[:, 1] > 0))
    elif consecutive:
        onsets = (events[:, 2] > 0)
        offsets = (events[:, 1] > 0)
    else:
        onsets = (events[:, 1] == 0)
        offsets = (events[:, 2] == 0)

    onset_idx = np.where(onsets)[0]
    offset_idx = np.where(offsets)[0]

    # delete orphaned onsets/offsets
    if onset_idx[0] > offset_idx[0]:
        logger.info("Removing orphaned offset at the beginning of the file.")
        offset_idx = np.delete(offset_idx, 0)

    if onset_idx[-1] > offset_idx[-1]:
        logger.info("Removing orphaned onset at the end of the file.")
        onset_idx = np.delete(onset_idx, -1)

    # Only keep events longer than min_duration
    if min_duration > 0:
        duration = events[offset_idx][:, 0] - events[onset_idx][:, 0]
        keep = (duration >= min_duration * raw.info['sfreq'])
    else:
        keep = None

    if detect == 'onset':
        idx = onset_idx
    elif detect == 'offset':
        idx = offset_idx
    else:
        raise Exception("Invalid detect parameter %r" % detect)

    if keep is not None:
        n_reject = keep.sum()
        if n_reject > 0:
            logger.info("Removing %s events with duration < "
                        "%s" % (n_reject, min_duration))
            idx = idx[keep]

    events = events[idx]

    if detect == 'offset':
        events[:, 1:] = events[:, 2:0:-1]
        events[:, 0] -= 1

    logger.info("%s events found" % len(events))
    logger.info("Events id: %s" % np.unique(events[:, 2]))
    return events


def merge_events(events, ids, new_id):
    """Merge a set of events

    Parameters
    ----------
    events : array
        Events.
    ids : array of int
        The ids of events to merge.
    new_id : int
        The new id.

    Returns
    -------
    new_events: array
        The new events
    """
    events = events.copy()
    events_numbers = events[:, 2]
    for i in ids:
        events_numbers[events_numbers == i] = new_id
    return events


def make_fixed_length_events(raw, id, start=0, stop=None, duration=1.):
    """Make a set of events separated by a fixed duration

    Parameters
    ----------
    raw : instance of Raw
        A raw object to use the data from.
    duration: float
        The duration to separate events by.
    id : int
        The id to use.

    Returns
    -------
    new_events: array
        The new events
    """
    start = raw.time_as_index(start)
    start = start[0] + raw.first_samp
    if stop is not None:
        stop = raw.time_as_index(stop)
        stop = min([stop[0] + raw.first_samp, raw.last_samp + 1])
    else:
        stop = raw.last_samp + 1
    if not isinstance(id, int):
        raise ValueError('id must be an integer')
    # Make sure we don't go out the end of the file:
    stop -= np.ceil(raw.info['sfreq'] * duration)
    ts = np.arange(start, stop, raw.info['sfreq'] * duration).astype(int)
    n_events = len(ts)
    events = np.c_[ts, np.zeros(n_events, dtype=int),
                   id * np.ones(n_events, dtype=int)]
    return events


def concatenate_events(events, first_samps, last_samps):
    """Concatenate event lists in a manner compatible with
    concatenate_raws

    This is useful, for example, if you processed and/or changed
    events in raw files separately before combining them using
    concatenate_raws.

    Parameters
    ----------
    events : list of arrays
        List of event arrays, typically each extracted from a
        corresponding raw file that is being concatenated.

    first_samps : list or array of int
        First sample numbers of the raw files concatenated.

    last_samps : list or array of int
        Last sample numbers of the raw files concatenated.

    Returns
    -------
    events : array
        The concatenated events.
    """
    if not isinstance(events, list):
        raise ValueError('events must be a list of arrays')
    if not (len(events) == len(last_samps) and
            len(events) == len(first_samps)):
        raise ValueError('events, first_samps, and last_samps must all have '
                         'the same lengths')
    first_samps = np.array(first_samps)
    last_samps = np.array(last_samps)
    n_samps = np.cumsum(last_samps - first_samps + 1)
    events_out = events[0]
    for e, f, n in zip(events[1:], first_samps[1:], n_samps[:-1]):
        # remove any skip since it doesn't exist in concatenated files
        e2 = e.copy()
        e2[:, 0] -= f
        # add offset due to previous files, plus original file offset
        e2[:, 0] += n + first_samps[0]
        events_out = np.concatenate((events_out, e2), axis=0)

    return events_out


def _get_stim_channel(stim_channel):
    """Helper to determine the appropriate stim_channel"""
    if stim_channel is not None:
        if not isinstance(stim_channel, list):
            if not isinstance(stim_channel, basestring):
                raise ValueError('stim_channel must be a str, list, or None')
            stim_channel = [stim_channel]
        if not all([isinstance(s, basestring) for s in stim_channel]):
            raise ValueError('stim_channel list must contain all strings')
        return stim_channel

    stim_channel = list()
    ch_count = 0
    ch = get_config('MNE_STIM_CHANNEL')
    while(ch is not None):
        stim_channel.append(ch)
        ch_count += 1
        ch = get_config('MNE_STIM_CHANNEL_%d' % ch_count)
    if ch_count == 0:
        stim_channel = ['STI 014']
    return stim_channel
