"""IO with fif files containing events
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Teon Brooks <teon.brooks@gmail.com>
#          Clement Moutard <clement.moutard@polytechnique.org>
#
# License: BSD (3-clause)

import numpy as np
from os.path import splitext

from .utils import check_fname, logger, verbose, _get_stim_channel
from .io.constants import FIFF
from .io.tree import dir_tree_find
from .io.tag import read_tag
from .io.open import fiff_open
from .io.write import write_int, start_block, start_file, end_block, end_file
from .io.pick import pick_channels


def pick_events(events, include=None, exclude=None, step=False):
    """Select some events

    Parameters
    ----------
    events : ndarray
        Array as returned by mne.find_events.
    include : int | list | None
        A event id to include or a list of them.
        If None all events are included.
    exclude : int | list | None
        A event id to exclude or a list of them.
        If None no event is excluded. If include is not None
        the exclude parameter is ignored.
    step : bool
        If True (default is False), events have a step format according
        to the argument output='step' in the function find_events().
        In this case, the two last columns are considered in inclusion/
        exclusion criteria.

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
            if step:
                mask = np.logical_or(mask, events[:, 1] == e)
        events = events[mask]
    elif exclude is not None:
        if not isinstance(exclude, list):
            exclude = [exclude]
        mask = np.ones(len(events), dtype=np.bool)
        for e in exclude:
            mask = np.logical_and(mask, events[:, 2] != e)
            if step:
                mask = np.logical_and(mask, events[:, 1] != e)
        events = events[mask]
    else:
        events = np.copy(events)

    if len(events) == 0:
        raise RuntimeError("No events found")

    return events


def define_target_events(events, reference_id, target_id, sfreq, tmin, tmax,
                         new_id=None, fill_na=None):
    """Define new events by co-occurrence of existing events

    This function can be used to evaluate events depending on the
    temporal lag to another event. For example, this can be used to
    analyze evoked responses which were followed by a button press within
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
    for event in events.copy().astype(int):
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
                lag.append(np.nan)

    new_events = np.array(new_events)

    with np.errstate(invalid='ignore'):  # casting nans
        lag = np.abs(lag, dtype='f8')
    if lag.any():
        lag *= tsample
    else:
        lag = np.array([])

    return new_events if new_events.any() else np.array([]), lag


def _read_events_fif(fid, tree):
    """Aux function"""
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

    if mappings is not None:  # deal with ':' in keys
        m_ = [[s[::-1] for s in m[::-1].split(':', 1)]
              for m in mappings.split(';')]
        mappings = dict((k, int(v)) for v, k in m_)
    event_list = event_list.reshape(len(event_list) // 3, 3)
    return event_list, mappings


def read_events(filename, include=None, exclude=None, mask=0):
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
    mask : int
        The value of the digital mask to apply to the stim channel values.
        The default value is 0.

    Returns
    -------
    events: array, shape (n_events, 3)
        The list of events

    See Also
    --------
    find_events, write_events

    Notes
    -----
    This function will discard the offset line (i.e., first line with zero
    event number) if it is present in a text file.

    Working with downsampled data: Events that were computed before the data
    was decimated are no longer valid. Please recompute your events after
    decimation.
    """
    check_fname(filename, 'events', ('.eve', '-eve.fif', '-eve.fif.gz',
                                     '-eve.lst', '-eve.txt'))

    ext = splitext(filename)[1].lower()
    if ext == '.fif' or ext == '.gz':
        fid, tree, _ = fiff_open(filename)
        try:
            event_list, _ = _read_events_fif(fid, tree)
        finally:
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
            goods = [0, 2, 3]  # Omit "time" variable
        elif len(lines[0]) == 3:
            goods = [0, 1, 2]
        else:
            raise ValueError('Unknown number of columns in event text file')

        event_list = lines[:, goods]
        if event_list.shape[0] > 0 and event_list[0, 2] == 0:
            event_list = event_list[1:]

    event_list = pick_events(event_list, include, exclude)
    event_list = _mask_trigs(event_list, mask)

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

    See Also
    --------
    read_events
    """
    check_fname(filename, 'events', ('.eve', '-eve.fif', '-eve.fif.gz',
                                     '-eve.lst', '-eve.txt'))

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
        for e in event_list:
            f.write('%6d %6d %3d\n' % tuple(e))
        f.close()


def _find_stim_steps(data, first_samp, pad_start=None, pad_stop=None, merge=0):
    changed = np.diff(data, axis=1) != 0
    idx = np.where(np.all(changed, axis=0))[0]
    if len(idx) == 0:
        return np.empty((0, 3), dtype='int32')

    pre_step = data[0, idx]
    idx += 1
    post_step = data[0, idx]
    idx += first_samp
    steps = np.c_[idx, pre_step, post_step]

    if pad_start is not None:
        v = steps[0, 1]
        if v != pad_start:
            steps = np.insert(steps, 0, [0, pad_start, v], axis=0)

    if pad_stop is not None:
        v = steps[-1, 2]
        if v != pad_stop:
            last_idx = len(data[0]) + first_samp
            steps = np.append(steps, [[last_idx, v, pad_stop]], axis=0)

    if merge != 0:
        diff = np.diff(steps[:, 0])
        idx = (diff <= abs(merge))
        if np.any(idx):
            where = np.where(idx)[0]
            keep = np.logical_not(idx)
            if merge > 0:
                # drop the earlier event
                steps[where + 1, 1] = steps[where, 1]
                keep = np.append(keep, True)
            else:
                # drop the later event
                steps[where, 2] = steps[where + 1, 2]
                keep = np.insert(keep, 0, True)

            is_step = (steps[:, 1] != steps[:, 2])
            keep = np.logical_and(keep, is_step)
            steps = steps[keep]

    return steps


def find_stim_steps(raw, pad_start=None, pad_stop=None, merge=0,
                    stim_channel=None):
    """Find all steps in data from a stim channel

    Parameters
    ----------
    raw : Raw object
        The raw data.
    pad_start: None | int
        Values to assume outside of the stim channel (e.g., if pad_start=0 and
        the stim channel starts with value 5, an event of [0, 0, 5] will be
        inserted at the beginning). With None, no steps will be inserted.
    pad_stop : None | int
        Values to assume outside of the stim channel, see ``pad_start``.
    merge : int
        Merge steps occurring in neighboring samples. The integer value
        indicates over how many samples events should be merged, and the sign
        indicates in which direction they should be merged (negative means
        towards the earlier event, positive towards the later event).
    stim_channel : None | string | list of string
        Name of the stim channel or all the stim channels
        affected by the trigger. If None, the config variables
        'MNE_STIM_CHANNEL', 'MNE_STIM_CHANNEL_1', 'MNE_STIM_CHANNEL_2',
        etc. are read. If these are not found, it will default to
        'STI 014'.

    Returns
    -------
    steps : array, shape = (n_samples, 3)
        For each step in the stim channel the values [sample, v_from, v_to].
        The first column contains the event time in samples (the first sample
        with the new value). The second column contains the stim channel value
        before the step, and the third column contains value after the step.

    See Also
    --------
    find_events : More sophisticated options for finding events in a Raw file.
    """

    # pull stim channel from config if necessary
    stim_channel = _get_stim_channel(stim_channel, raw.info)

    picks = pick_channels(raw.info['ch_names'], include=stim_channel)
    if len(picks) == 0:
        raise ValueError('No stim channel found to extract event triggers.')
    data, _ = raw[picks, :]
    if np.any(data < 0):
        logger.warning('Trigger channel contains negative values. '
                       'Taking absolute value.')
        data = np.abs(data)  # make sure trig channel is positive
    data = data.astype(np.int)

    return _find_stim_steps(data, raw.first_samp, pad_start=pad_start,
                            pad_stop=pad_stop, merge=merge)


@verbose
def _find_events(data, first_samp, verbose=None, output='onset',
                 consecutive='increasing', min_samples=0, mask=0):
    """Helper function for find events"""
    if min_samples > 0:
        merge = int(min_samples // 1)
        if merge == min_samples:
            merge -= 1
    else:
        merge = 0

    if np.any(data < 0):
        logger.warning('Trigger channel contains negative values. '
                       'Taking absolute value.')
        data = np.abs(data)  # make sure trig channel is positive
    data = data.astype(np.int)

    events = _find_stim_steps(data, first_samp, pad_stop=0, merge=merge)
    events = _mask_trigs(events, mask)

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

    if len(onset_idx) == 0 or len(offset_idx) == 0:
        return np.empty((0, 3), dtype='int32')

    # delete orphaned onsets/offsets
    if onset_idx[0] > offset_idx[0]:
        logger.info("Removing orphaned offset at the beginning of the file.")
        offset_idx = np.delete(offset_idx, 0)

    if onset_idx[-1] > offset_idx[-1]:
        logger.info("Removing orphaned onset at the end of the file.")
        onset_idx = np.delete(onset_idx, -1)

    if output == 'onset':
        events = events[onset_idx]
    elif output == 'step':
        idx = np.union1d(onset_idx, offset_idx)
        events = events[idx]
    elif output == 'offset':
        event_id = events[onset_idx, 2]
        events = events[offset_idx]
        events[:, 1] = events[:, 2]
        events[:, 2] = event_id
        events[:, 0] -= 1
    else:
        raise Exception("Invalid output parameter %r" % output)

    logger.info("%s events found" % len(events))
    logger.info("Events id: %s" % np.unique(events[:, 2]))

    return events


@verbose
def find_events(raw, stim_channel=None, verbose=None, output='onset',
                consecutive='increasing', min_duration=0,
                shortest_event=2, mask=0):
    """Find events from raw file

    Parameters
    ----------
    raw : Raw object
        The raw data.
    stim_channel : None | string | list of string
        Name of the stim channel or all the stim channels
        affected by the trigger. If None, the config variables
        'MNE_STIM_CHANNEL', 'MNE_STIM_CHANNEL_1', 'MNE_STIM_CHANNEL_2',
        etc. are read. If these are not found, it will fall back to
        'STI 014' if present, then fall back to the first channel of type
        'stim', if present.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    output : 'onset' | 'offset' | 'step'
        Whether to report when events start, when events end, or both.
    consecutive : bool | 'increasing'
        If True, consider instances where the value of the events
        channel changes without first returning to zero as multiple
        events. If False, report only instances where the value of the
        events channel changes from/to zero. If 'increasing', report
        adjacent events only when the second event code is greater than
        the first.
    min_duration : float
        The minimum duration of a change in the events channel required
        to consider it as an event (in seconds).
    shortest_event : int
        Minimum number of samples an event must last (default is 2). If the
        duration is less than this an exception will be raised.
    mask : int
        The value of the digital mask to apply to the stim channel values.
        The default value is 0.

    Returns
    -------
    events : array, shape = (n_events, 3)
        All events that were found. The first column contains the event time
        in samples and the third column contains the event id. For output =
        'onset' or 'step', the second column contains the value of the stim
        channel immediately before the the event/step. For output = 'offset',
        the second column contains the value of the stim channel after the
        event offset.

    Examples
    --------
    Consider data with a stim channel that looks like: [0, 32, 32, 33, 32, 0]

    By default, find_events returns all samples at which the value of the
    stim channel increases::

        >>> print(find_events(raw)) # doctest: +SKIP
        [[ 1  0 32]
         [ 3 32 33]]

    If consecutive is False, find_events only returns the samples at which
    the stim channel changes from zero to a non-zero value::

        >>> print(find_events(raw, consecutive=False)) # doctest: +SKIP
        [[ 1  0 32]]

    If consecutive is True, find_events returns samples at which the
    event changes, regardless of whether it first returns to zero::

        >>> print(find_events(raw, consecutive=True)) # doctest: +SKIP
        [[ 1  0 32]
         [ 3 32 33]
         [ 4 33 32]]

    If output is 'offset', find_events returns the last sample of each event
    instead of the first one::

        >>> print(find_events(raw, consecutive=True, # doctest: +SKIP
        ...                   output='offset'))
        [[ 2 33 32]
         [ 3 32 33]
         [ 4  0 32]]

    If output is 'step', find_events returns the samples at which an event
    starts or ends::

        >>> print(find_events(raw, consecutive=True, # doctest: +SKIP
        ...                   output='step'))
        [[ 1  0 32]
         [ 3 32 33]
         [ 4 33 32]
         [ 5 32  0]]

    To ignore spurious events, it is also possible to specify a minimum
    event duration. Assuming our events channel has a sample rate of
    1000 Hz::

        >>> print(find_events(raw, consecutive=True, # doctest: +SKIP
        ...                   min_duration=0.002))
        [[ 1  0 32]]

    For the digital mask, it will take the binary representation of the
    digital mask, e.g. 5 -> '00000101', and will block the values
    where mask is one, e.g.::

              7 '0000111' <- trigger value
             37 '0100101' <- mask
         ----------------
              2 '0000010'

    See Also
    --------
    find_stim_steps : Find all the steps in the stim channel.
    """
    min_samples = min_duration * raw.info['sfreq']

    # pull stim channel from config if necessary
    stim_channel = _get_stim_channel(stim_channel, raw.info)

    pick = pick_channels(raw.info['ch_names'], include=stim_channel)
    if len(pick) == 0:
        raise ValueError('No stim channel found to extract event triggers.')
    data, _ = raw[pick, :]

    events = _find_events(data, raw.first_samp, verbose=verbose, output=output,
                          consecutive=consecutive, min_samples=min_samples,
                          mask=mask)

    # add safety check for spurious events (for ex. from neuromag syst.) by
    # checking the number of low sample events
    n_short_events = np.sum(np.diff(events[:, 0]) < shortest_event)
    if n_short_events > 0:
        raise ValueError("You have %i events shorter than the "
                         "shortest_event. These are very unusual and you "
                         "may want to set min_duration to a larger value e.g."
                         " x / raw.info['sfreq']. Where x = 1 sample shorter "
                         "than the shortest event length." % (n_short_events))

    return events


def _mask_trigs(events, mask):
    """Helper function for masking digital trigger values"""
    if not isinstance(mask, int):
        raise TypeError('You provided a(n) %s. Mask must be an int.'
                        % type(mask))
    n_events = len(events)
    if n_events == 0:
        return events.copy()

    mask = np.bitwise_not(mask)
    events[:, 1:] = np.bitwise_and(events[:, 1:], mask)
    events = events[events[:, 1] != events[:, 2]]

    return events


def merge_events(events, ids, new_id, replace_events=True):
    """Merge a set of events

    Parameters
    ----------
    events : array
        Events.
    ids : array of int
        The ids of events to merge.
    new_id : int
        The new id.
    replace_events : bool
        If True (default), old event ids are replaced. Otherwise,
        new events will be added to the old event list.

    Returns
    -------
    new_events: array
        The new events
    """
    events_out = events.copy()
    where = np.empty(events.shape[0], dtype=bool)
    for col in [1, 2]:
        where.fill(False)
        for i in ids:
            where = (events[:, col] == i)
            events_out[where, col] = new_id
    if not replace_events:
        events_out = np.concatenate((events_out, events), axis=0)
        events_out = events_out[np.argsort(events_out[:, 0])]
    return events_out


def shift_time_events(events, ids, tshift, sfreq):
    """Shift an event

    Parameters
    ----------
    events : array, shape=(n_events, 3)
        The events
    ids : array int
        The ids of events to shift.
    tshift : float
        Time-shift event. Use positive value tshift for forward shifting
        the event and negative value for backward shift.
    sfreq : float
        The sampling frequency of the data.

    Returns
    -------
    new_events : array
        The new events.
    """
    events = events.copy()
    for ii in ids:
        events[events[:, 2] == ii, 0] += int(tshift * sfreq)
    return events


def make_fixed_length_events(raw, id, start=0, stop=None, duration=1.):
    """Make a set of events separated by a fixed duration

    Parameters
    ----------
    raw : instance of Raw
        A raw object to use the data from.
    id : int
        The id to use.
    start : float
        Time of first event.
    stop : float | None
        Maximum time of last event. If None, events extend to the end
        of the recording.
    duration: float
        The duration to separate events by.

    Returns
    -------
    new_events : array
        The new events.
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
    stop -= int(np.ceil(raw.info['sfreq'] * duration))
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
