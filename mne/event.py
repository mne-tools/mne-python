"""IO with fif files containing events."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Teon Brooks <teon.brooks@gmail.com>
#          Clement Moutard <clement.moutard@polytechnique.org>
#
# License: BSD (3-clause)

import numpy as np
from os.path import splitext


from .utils import (check_fname, logger, verbose, _get_stim_channel, warn,
                    _validate_type, _check_option)
from .io.constants import FIFF
from .io.tree import dir_tree_find
from .io.tag import read_tag
from .io.open import fiff_open
from .io.write import write_int, start_block, start_file, end_block, end_file
from .io.pick import pick_channels


def pick_events(events, include=None, exclude=None, step=False):
    """Select some events.

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
    """Define new events by co-occurrence of existing events.

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
    """Aux function."""
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
        mappings = {k: int(v) for v, k in m_}
    event_list = event_list.reshape(len(event_list) // 3, 3)
    return event_list, mappings


def read_events(filename, include=None, exclude=None, mask=None,
                mask_type='and'):
    """Read events from fif or text file.

    See :ref:`tut_epoching_and_averaging` as well as :ref:`ex-read-events`
    for more information about events.

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
    mask : int | None
        The value of the digital mask to apply to the stim channel values.
        If None (default), no masking is performed.
    mask_type: 'and' | 'not_and'
        The type of operation between the mask and the trigger.
        Choose 'and' (default) for MNE-C masking behavior.

        .. versionadded:: 0.13

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

    For more information on ``mask`` and ``mask_type``, see
    :func:`mne.find_events`.
    """
    check_fname(filename, 'events', ('.eve', '-eve.fif', '-eve.fif.gz',
                                     '-eve.lst', '-eve.txt', '_eve.fif',
                                     '_eve.fif.gz', '_eve.lst', '_eve.txt'))

    ext = splitext(filename)[1].lower()
    if ext == '.fif' or ext == '.gz':
        fid, tree, _ = fiff_open(filename)
        with fid as f:
            event_list, _ = _read_events_fif(f, tree)
        # hack fix for windows to avoid bincount problems
        event_list = event_list.astype(int)
    else:
        #  Have to read this in as float64 then convert because old style
        #  eve/lst files had a second float column that will raise errors
        lines = np.loadtxt(filename, dtype=np.float64).astype(int)
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
        if (mask is not None and event_list.shape[0] > 0 and
                event_list[0, 2] == 0):
            event_list = event_list[1:]
            warn('first row of event file discarded (zero-valued)')

    event_list = pick_events(event_list, include, exclude)
    unmasked_len = event_list.shape[0]
    if mask is not None:
        event_list = _mask_trigs(event_list, mask, mask_type)
        masked_len = event_list.shape[0]
        if masked_len < unmasked_len:
            warn('{} of {} events masked'.format(unmasked_len - masked_len,
                                                 unmasked_len))
    return event_list


def write_events(filename, event_list):
    """Write events to file.

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
                                     '-eve.lst', '-eve.txt', '_eve.fif',
                                     '_eve.fif.gz', '_eve.lst', '_eve.txt'))

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
    """Find all steps in data from a stim channel.

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
        'STI101' or 'STI 014', whichever is present.

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
        warn('Trigger channel contains negative values, using absolute value.')
        data = np.abs(data)  # make sure trig channel is positive
    data = data.astype(np.int)

    return _find_stim_steps(data, raw.first_samp, pad_start=pad_start,
                            pad_stop=pad_stop, merge=merge)


@verbose
def _find_events(data, first_samp, verbose=None, output='onset',
                 consecutive='increasing', min_samples=0, mask=None,
                 uint_cast=False, mask_type='and', initial_event=False):
    """Help find events."""
    assert data.shape[0] == 1  # data should be only a row vector

    if min_samples > 0:
        merge = int(min_samples // 1)
        if merge == min_samples:
            merge -= 1
    else:
        merge = 0

    data = data.astype(np.int)
    if uint_cast:
        data = data.astype(np.uint16).astype(np.int)
    if data.min() < 0:
        warn('Trigger channel contains negative values, using absolute '
             'value. If data were acquired on a Neuromag system with '
             'STI016 active, consider using uint_cast=True to work around '
             'an acquisition bug')
        data = np.abs(data)  # make sure trig channel is positive

    events = _find_stim_steps(data, first_samp, pad_stop=0, merge=merge)
    initial_value = data[0, 0]
    if initial_value != 0:
        if initial_event:
            events = np.insert(events, 0, [0, 0, initial_value], axis=0)
        else:
            logger.info('Trigger channel has a non-zero initial value of {} '
                        '(consider using initial_event=True to detect this '
                        'event)'.format(initial_value))

    events = _mask_trigs(events, mask, mask_type)

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
        raise ValueError("Invalid output parameter %r" % output)

    logger.info("%s events found" % len(events))
    logger.info("Event IDs: %s" % np.unique(events[:, 2]))

    return events


def _find_unique_events(events):
    """Uniquify events (ie remove duplicated rows."""
    e = np.ascontiguousarray(events).view(
        np.dtype((np.void, events.dtype.itemsize * events.shape[1])))
    _, idx = np.unique(e, return_index=True)
    n_dupes = len(events) - len(idx)
    if n_dupes > 0:
        warn("Some events are duplicated in your different stim channels."
             " %d events were ignored during deduplication." % n_dupes)
    return events[idx]


@verbose
def find_events(raw, stim_channel=None, output='onset',
                consecutive='increasing', min_duration=0,
                shortest_event=2, mask=None, uint_cast=False,
                mask_type='and', initial_event=False, verbose=None):
    """Find events from raw file.

    See :ref:`tut_epoching_and_averaging` as well as :ref:`ex-read-events`
    for more information about events.

    Parameters
    ----------
    raw : Raw object
        The raw data.
    stim_channel : None | string | list of string
        Name of the stim channel or all the stim channels
        affected by triggers. If None, the config variables
        'MNE_STIM_CHANNEL', 'MNE_STIM_CHANNEL_1', 'MNE_STIM_CHANNEL_2',
        etc. are read. If these are not found, it will fall back to
        'STI 014' if present, then fall back to the first channel of type
        'stim', if present. If multiple channels are provided
        then the returned events are the union of all the events
        extracted from individual stim channels.
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
    mask : int | None
        The value of the digital mask to apply to the stim channel values.
        If None (default), no masking is performed.
    uint_cast : bool
        If True (default False), do a cast to ``uint16`` on the channel
        data. This can be used to fix a bug with STI101 and STI014 in
        Neuromag acquisition setups that use channel STI016 (channel 16
        turns data into e.g. -32768), similar to ``mne_fix_stim14 --32``
        in MNE-C.

        .. versionadded:: 0.12
    mask_type: 'and' | 'not_and'
        The type of operation between the mask and the trigger.
        Choose 'and' (default) for MNE-C masking behavior.

        .. versionadded:: 0.13
    initial_event : bool
        If True (default False), an event is created if the stim channel has a
        value different from 0 as its first sample. This is useful if an event
        at t=0s is present.

        .. versionadded:: 0.16
    %(verbose)s

    Returns
    -------
    events : array, shape = (n_events, 3)
        All events that were found. The first column contains the event time
        in samples and the third column contains the event id. For output =
        'onset' or 'step', the second column contains the value of the stim
        channel immediately before the event/step. For output = 'offset',
        the second column contains the value of the stim channel after the
        event offset.

    See Also
    --------
    find_stim_steps : Find all the steps in the stim channel.
    read_events : Read events from disk.
    write_events : Write events to disk.

    Notes
    -----
    .. warning:: If you are working with downsampled data, events computed
                 before decimation are no longer valid. Please recompute
                 your events after decimation, but note this reduces the
                 precision of event timing.

    Examples
    --------
    Consider data with a stim channel that looks like::

        [0, 32, 32, 33, 32, 0]

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

    For the digital mask, if mask_type is set to 'and' it will take the
    binary representation of the digital mask, e.g. 5 -> '00000101', and will
    allow the values to pass where mask is one, e.g.::

              7 '0000111' <- trigger value
             37 '0100101' <- mask
         ----------------
              5 '0000101'

    For the digital mask, if mask_type is set to 'not_and' it will take the
    binary representation of the digital mask, e.g. 5 -> '00000101', and will
    block the values where mask is one, e.g.::

              7 '0000111' <- trigger value
             37 '0100101' <- mask
         ----------------
              2 '0000010'

    """
    min_samples = min_duration * raw.info['sfreq']

    # pull stim channel from config if necessary
    try:
        stim_channel = _get_stim_channel(stim_channel, raw.info)
    except ValueError:
        if len(raw.annotations) > 0:
            raise ValueError("No stim channels found, but the raw object has "
                             "annotations. Consider using "
                             "mne.events_from_annotations to convert these to "
                             "events.")
        else:
            raise

    picks = pick_channels(raw.info['ch_names'], include=stim_channel)
    if len(picks) == 0:
        raise ValueError('No stim channel found to extract event triggers.')
    data, _ = raw[picks, :]

    events_list = []
    for d in data:
        events = _find_events(d[np.newaxis, :], raw.first_samp,
                              verbose=verbose, output=output,
                              consecutive=consecutive, min_samples=min_samples,
                              mask=mask, uint_cast=uint_cast,
                              mask_type=mask_type, initial_event=initial_event)
        # add safety check for spurious events (for ex. from neuromag syst.) by
        # checking the number of low sample events
        n_short_events = np.sum(np.diff(events[:, 0]) < shortest_event)
        if n_short_events > 0:
            raise ValueError("You have %i events shorter than the "
                             "shortest_event. These are very unusual and you "
                             "may want to set min_duration to a larger value "
                             "e.g. x / raw.info['sfreq']. Where x = 1 sample "
                             "shorter than the shortest event "
                             "length." % (n_short_events))

        events_list.append(events)

    events = np.concatenate(events_list, axis=0)
    events = _find_unique_events(events)
    events = events[np.argsort(events[:, 0])]
    return events


def _mask_trigs(events, mask, mask_type):
    """Mask digital trigger values."""
    _check_option('mask_type', mask_type, ['not_and', 'and'])
    if mask is not None:
        _validate_type(mask, "int", "mask", "int or None")
    n_events = len(events)
    if n_events == 0:
        return events.copy()

    if mask is not None:
        if mask_type == 'not_and':
            mask = np.bitwise_not(mask)
        elif mask_type != 'and':
            raise ValueError("'mask_type' should be either 'and'"
                             " or 'not_and', instead of '%s'" % mask_type)
        events[:, 1:] = np.bitwise_and(events[:, 1:], mask)
    events = events[events[:, 1] != events[:, 2]]

    return events


def merge_events(events, ids, new_id, replace_events=True):
    """Merge a set of events.

    Parameters
    ----------
    events : array, shape (n_events_in, 3)
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
    new_events : array, shape (n_events_out, 3)
        The new events

    Examples
    --------
    Here is quick example of the behavior::

        >>> events = [[134, 0, 1], [341, 0, 2], [502, 0, 3]]
        >>> merge_events(events, [1, 2], 12, replace_events=True)
        array([[134,   0,  12],
               [341,   0,  12],
               [502,   0,   3]])
        >>> merge_events(events, [1, 2], 12, replace_events=False)
        array([[134,   0,   1],
               [134,   0,  12],
               [341,   0,   2],
               [341,   0,  12],
               [502,   0,   3]])

    Notes
    -----
    Rather than merging events you can use hierarchical event_id
    in Epochs. For example, here::

        >>> event_id = {'auditory/left': 1, 'auditory/right': 2}

    And the condition 'auditory' would correspond to either 1 or 2.
    """
    events = np.asarray(events)
    events_out = events.copy()
    idx_touched = []  # to keep track of the original events we can keep
    for col in [1, 2]:
        for i in ids:
            mask = events[:, col] == i
            events_out[mask, col] = new_id
            idx_touched.append(np.where(mask)[0])
    if not replace_events:
        idx_touched = np.unique(np.concatenate(idx_touched))
        events_out = np.concatenate((events_out, events[idx_touched]), axis=0)
        # Now sort in lexical order
        events_out = events_out[np.lexsort(events_out.T[::-1])]
    return events_out


def shift_time_events(events, ids, tshift, sfreq):
    """Shift an event.

    Parameters
    ----------
    events : array, shape=(n_events, 3)
        The events
    ids : ndarray of int | None
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
    if ids is None:
        mask = slice(None)
    else:
        mask = np.in1d(events[:, 2], ids)
    events[mask, 0] += int(tshift * sfreq)

    return events


def make_fixed_length_events(raw, id=1, start=0, stop=None, duration=1.,
                             first_samp=True, overlap=0.):
    """Make a set of events separated by a fixed duration.

    Parameters
    ----------
    raw : instance of Raw
        A raw object to use the data from.
    id : int
        The id to use (default 1).
    start : float
        Time of first event.
    stop : float | None
        Maximum time of last event. If None, events extend to the end
        of the recording.
    duration: float
        The duration to separate events by.
    first_samp: bool
        If True (default), times will have raw.first_samp added to them, as
        in :func:`mne.find_events`. This behavior is not desirable if the
        returned events will be combined with event times that already
        have ``raw.first_samp`` added to them, e.g. event times that come
        from :func:`mne.find_events`.
    overlap : float
        The overlap between events. Must be ``0 <= overlap < duration``.

        .. versionadded:: 0.18

    Returns
    -------
    new_events : array
        The new events.
    """
    from .io.base import BaseRaw
    _validate_type(raw, BaseRaw, "raw")
    _validate_type(id, int, "id")
    _validate_type(duration, "numeric", "duration")
    _validate_type(overlap, "numeric", "overlap")
    duration, overlap = float(duration), float(overlap)
    if not 0 <= overlap < duration:
        raise ValueError('overlap must be >=0 but < duration (%s), got %s'
                         % (duration, overlap))

    start = raw.time_as_index(start, use_rounding=True)[0]
    if stop is not None:
        stop = raw.time_as_index(stop, use_rounding=True)[0]
    else:
        stop = raw.last_samp + 1
    if first_samp:
        start = start + raw.first_samp
        stop = min([stop + raw.first_samp, raw.last_samp + 1])
    else:
        stop = min([stop, len(raw.times)])
    # Make sure we don't go out the end of the file:
    stop -= int(np.round(raw.info['sfreq'] * duration))
    # This should be inclusive due to how we generally use start and stop...
    ts = np.arange(start, stop + 1,
                   raw.info['sfreq'] * (duration - overlap)).astype(int)
    n_events = len(ts)
    if n_events == 0:
        raise ValueError('No events produced, check the values of start, '
                         'stop, and duration')
    events = np.c_[ts, np.zeros(n_events, dtype=int),
                   id * np.ones(n_events, dtype=int)]
    return events


def concatenate_events(events, first_samps, last_samps):
    """Concatenate event lists to be compatible with concatenate_raws.

    This is useful, for example, if you processed and/or changed
    events in raw files separately before combining them using
    :func:`mne.concatenate_raws`.

    Parameters
    ----------
    events : list of array
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

    See Also
    --------
    mne.concatenate_raws
    """
    _validate_type(events, list, "events")
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


class AcqParserFIF(object):
    """Parser for Elekta data acquisition settings.

    This class parses parameters (e.g. events and averaging categories) that
    are defined in the Elekta TRIUX/VectorView data acquisition software (DACQ)
    and stored in ``info['acq_pars']``. It can be used to reaverage raw data
    according to DACQ settings and modify original averaging settings if
    necessary.

    Parameters
    ----------
    info : Info
        An instance of Info where the DACQ parameters will be taken from.

    Attributes
    ----------
    categories : list
        List of averaging categories marked active in DACQ.
    events : list
        List of events that are in use (referenced by some averaging category).
    reject : dict
        Rejection criteria from DACQ that can be used with mne.Epochs.
        Note that mne does not support all DACQ rejection criteria
        (e.g. spike, slope).
    flat : dict
        Flatness rejection criteria from DACQ that can be used with mne.Epochs.
    acq_dict : dict
        All DACQ parameters.

    See Also
    --------
    mne.io.Raw.acqparser : access the parser through a Raw attribute

    Notes
    -----
    Any averaging category (also non-active ones) can be accessed by indexing
    as ``acqparserfif['category_name']``.
    """

    # DACQ variables always start with one of these
    _acq_var_magic = ['ERF', 'DEF', 'ACQ', 'TCP']

    # averager related DACQ variable names (without preceding 'ERF')
    # old versions (DACQ < 3.4)
    _dacq_vars_compat = ('megMax', 'megMin', 'megNoise', 'megSlope',
                         'megSpike', 'eegMax', 'eegMin', 'eegNoise',
                         'eegSlope', 'eegSpike', 'eogMax', 'ecgMax', 'ncateg',
                         'nevent', 'stimSource', 'triggerMap', 'update',
                         'artefIgnore', 'averUpdate')

    _event_vars_compat = ('Comment', 'Delay')

    _cat_vars = ('Comment', 'Display', 'Start', 'State', 'End', 'Event',
                 'Nave', 'ReqEvent', 'ReqWhen', 'ReqWithin', 'SubAve')

    # new versions only (DACQ >= 3.4)
    _dacq_vars = _dacq_vars_compat + ('magMax', 'magMin', 'magNoise',
                                      'magSlope', 'magSpike', 'version')

    _event_vars = _event_vars_compat + ('Name', 'Channel', 'NewBits',
                                        'OldBits', 'NewMask', 'OldMask')

    def __init__(self, info):  # noqa: D102
        acq_pars = info['acq_pars']
        if not acq_pars:
            raise ValueError('No acquisition parameters')
        self.acq_dict = dict(self._acqpars_gen(acq_pars))
        if 'ERFversion' in self.acq_dict:
            self.compat = False  # DACQ ver >= 3.4
        elif 'ERFncateg' in self.acq_dict:  # probably DACQ < 3.4
            self.compat = True
        else:
            raise ValueError('Cannot parse acquisition parameters')
        dacq_vars = self._dacq_vars_compat if self.compat else self._dacq_vars
        # set instance variables
        for var in dacq_vars:
            val = self.acq_dict['ERF' + var]
            if var[:3] in ['mag', 'meg', 'eeg', 'eog', 'ecg']:
                val = float(val)
            elif var in ['ncateg', 'nevent']:
                val = int(val)
            setattr(self, var.lower(), val)
        self.stimsource = (
            'Internal' if self.stimsource == '1' else 'External')
        # collect all events and categories
        self._events = self._events_from_acq_pars()
        self._categories = self._categories_from_acq_pars()
        # mark events that are used by a category
        for cat in self._categories.values():
            if cat['event']:
                self._events[cat['event']]['in_use'] = True
            if cat['reqevent']:
                self._events[cat['reqevent']]['in_use'] = True
        # make mne rejection dicts based on the averager parameters
        self.reject = {'grad': self.megmax, 'eeg': self.eegmax,
                       'eog': self.eogmax, 'ecg': self.ecgmax}
        if not self.compat:
            self.reject['mag'] = self.magmax
        self.reject = {k: float(v) for k, v in self.reject.items()
                       if float(v) > 0}
        self.flat = {'grad': self.megmin, 'eeg': self.eegmin}
        if not self.compat:
            self.flat['mag'] = self.magmin
        self.flat = {k: float(v) for k, v in self.flat.items()
                     if float(v) > 0}

    def __repr__(self):  # noqa: D105
        s = '<AcqParserFIF | '
        s += 'categories: %d ' % self.ncateg
        cats_in_use = len(self._categories_in_use)
        s += '(%d in use), ' % cats_in_use
        s += 'events: %d ' % self.nevent
        evs_in_use = len(self._events_in_use)
        s += '(%d in use)' % evs_in_use
        if self.categories:
            s += '\nAveraging categories:'
            for cat in self.categories:
                s += '\n%d: "%s"' % (cat['index'], cat['comment'])
        s += '>'
        return s

    def __getitem__(self, item):
        """Return an averaging category, or list of categories.

        Parameters
        ----------
        item : str | list of str
            Name of the category (comment field in DACQ).

        Returns
        -------
        conds : dict | list of dict
            Each dict should have the following keys:

            comment: str
                The comment field in DACQ.
            state : bool
                Whether the category was marked enabled in DACQ.
            index : int
                The index of the category in DACQ. Indices start from 1.
            event : int
                DACQ index of the reference event (trigger event, zero time for
                the corresponding epochs). Note that the event indices start
                from 1.
            start : float
                Start time of epoch relative to the reference event.
            end : float
                End time of epoch relative to the reference event.
            reqevent : int
                Index of the required (conditional) event.
            reqwhen : int
                Whether the required event is required before (1) or after (2)
                the reference event.
            reqwithin : float
                The time range within which the required event must occur,
                before or after the reference event.
            display : bool
                Whether the category was displayed online in DACQ.
            nave : int
                Desired number of averages. DACQ stops collecting averages once
                this number is reached.
            subave : int
                Whether to compute normal and alternating subaverages, and
                how many epochs to include. See the Elekta data acquisition
                manual for details. Currently the class does not offer any
                facility for computing subaverages, but it can be done manually
                by the user after collecting the epochs.

        """
        if isinstance(item, str):
            item = [item]
        else:
            _validate_type(item, list, "Keys", "category names")
        cats = list()
        for it in item:
            if it in self._categories:
                cats.append(self._categories[it])
            else:
                raise KeyError('No such category')
        return cats[0] if len(cats) == 1 else cats

    def __len__(self):
        """Return number of averaging categories marked active in DACQ."""
        return len(self.categories)

    def _events_from_acq_pars(self):
        """Collect DACQ events into a dict.

        Events are keyed by number starting from 1 (DACQ index of event).
        Each event is itself represented by a dict containing the event
        parameters.
        """
        # lookup table for event number -> bits for old DACQ versions
        _compat_event_lookup = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 3,
                                8: 5, 9: 6, 10: 7, 11: 9, 12: 10, 13: 11,
                                14: 12, 15: 13, 16: 14, 17: 15}
        events = dict()
        for evnum in range(1, self.nevent + 1):
            evnum_s = str(evnum).zfill(2)  # '01', '02' etc.
            evdi = dict()
            event_vars = (self._event_vars_compat if self.compat
                          else self._event_vars)
            for var in event_vars:
                # name of DACQ variable, e.g. 'ERFeventNewBits01'
                acq_key = 'ERFevent' + var + evnum_s
                # corresponding dict key, e.g. 'newbits'
                dict_key = var.lower()
                val = self.acq_dict[acq_key]
                # type convert numeric values
                if dict_key in ['newbits', 'oldbits', 'newmask', 'oldmask']:
                    val = int(val)
                elif dict_key in ['delay']:
                    val = float(val)
                evdi[dict_key] = val
                evdi['in_use'] = False  # __init__() will set this
            evdi['index'] = evnum
            if self.compat:
                evdi['name'] = str(evnum)
                evdi['oldmask'] = 63
                evdi['newmask'] = 63
                evdi['oldbits'] = 0
                evdi['newbits'] = _compat_event_lookup[evnum]
            events[evnum] = evdi
        return events

    def _acqpars_gen(self, acq_pars):
        """Yield key/value pairs from ``info['acq_pars'])``."""
        key, val = '', ''
        for line in acq_pars.split():
            if any([line.startswith(x) for x in self._acq_var_magic]):
                key = line
                val = ''
            else:
                if not key:
                    raise ValueError('Cannot parse acquisition parameters')
                # DACQ splits items with spaces into multiple lines
                val += ' ' + line if val else line
            yield key, val

    def _categories_from_acq_pars(self):
        """Collect DACQ averaging categories into a dict.

        Categories are keyed by the comment field in DACQ. Each category is
        itself represented a dict containing the category parameters.
        """
        cats = dict()
        for catnum in [str(x).zfill(2) for x in range(1, self.nevent + 1)]:
            catdi = dict()
            # read all category variables
            for var in self._cat_vars:
                acq_key = 'ERFcat' + var + catnum
                class_key = var.lower()
                val = self.acq_dict[acq_key]
                catdi[class_key] = val
            # some type conversions
            catdi['display'] = (catdi['display'] == '1')
            catdi['state'] = (catdi['state'] == '1')
            for key in ['start', 'end', 'reqwithin']:
                catdi[key] = float(catdi[key])
            for key in ['nave', 'event', 'reqevent', 'reqwhen', 'subave']:
                catdi[key] = int(catdi[key])
            # some convenient extra (non-DACQ) vars
            catdi['index'] = int(catnum)  # index of category in DACQ list
            cats[catdi['comment']] = catdi
        return cats

    def _events_mne_to_dacq(self, mne_events):
        """Create list of DACQ events based on mne trigger transitions list.

        mne_events is typically given by mne.find_events (use consecutive=True
        to get all transitions). Output consists of rows in the form
        [t, 0, event_codes] where t is time in samples and event_codes is all
        DACQ events compatible with the transition, bitwise ORed together:
        e.g. [t1, 0, 5] means that events 1 and 3 occurred at time t1,
        as 2**(1 - 1) + 2**(3 - 1) = 5.
        """
        events_ = mne_events.copy()
        events_[:, 1:3] = 0
        for n, ev in self._events.items():
            if ev['in_use']:
                pre_ok = (
                    np.bitwise_and(ev['oldmask'],
                                   mne_events[:, 1]) == ev['oldbits'])
                post_ok = (
                    np.bitwise_and(ev['newmask'],
                                   mne_events[:, 2]) == ev['newbits'])
                ok_ind = np.where(pre_ok & post_ok)
                events_[ok_ind, 2] |= 1 << (n - 1)
        return events_

    def _mne_events_to_category_t0(self, cat, mne_events, sfreq):
        """Translate mne_events to epoch zero times (t0).

        First mne events (trigger transitions) are converted into DACQ events.
        Then the zero times for the epochs are obtained by considering the
        reference and conditional (required) events and the delay to stimulus.
        """
        cat_ev = cat['event']
        cat_reqev = cat['reqevent']
        # first convert mne events to dacq event list
        events = self._events_mne_to_dacq(mne_events)
        # next, take req. events and delays into account
        times = events[:, 0]
        # indices of times where ref. event occurs
        refEvents_inds = np.where(events[:, 2] & (1 << cat_ev - 1))[0]
        refEvents_t = times[refEvents_inds]
        if cat_reqev:
            # indices of times where req. event occurs
            reqEvents_inds = np.where(events[:, 2] & (
                1 << cat_reqev - 1))[0]
            reqEvents_t = times[reqEvents_inds]
            # relative (to refevent) time window where req. event
            # must occur (e.g. [0 .2])
            twin = [0, (-1)**(cat['reqwhen']) * cat['reqwithin']]
            win = np.round(np.array(sorted(twin)) * sfreq)  # to samples
            refEvents_wins = refEvents_t[:, None] + win
            req_acc = np.zeros(refEvents_inds.shape, dtype=bool)
            for t in reqEvents_t:
                # mark time windows where req. condition is satisfied
                reqEvent_in_win = np.logical_and(
                    t >= refEvents_wins[:, 0], t <= refEvents_wins[:, 1])
                req_acc |= reqEvent_in_win
            # drop ref. events where req. event condition is not satisfied
            refEvents_inds = refEvents_inds[np.where(req_acc)]
            refEvents_t = times[refEvents_inds]
        # adjust for trigger-stimulus delay by delaying the ref. event
        refEvents_t += int(np.round(self._events[cat_ev]['delay'] * sfreq))
        return refEvents_t

    @property
    def categories(self):
        """Return list of averaging categories ordered by DACQ index.

        Only returns categories marked active in DACQ.
        """
        cats = sorted(self._categories_in_use.values(),
                      key=lambda cat: cat['index'])
        return cats

    @property
    def events(self):
        """Return events ordered by DACQ index.

        Only returns events that are in use (referred to by a category).
        """
        evs = sorted(self._events_in_use.values(), key=lambda ev: ev['index'])
        return evs

    @property
    def _categories_in_use(self):
        return {k: v for k, v in self._categories.items() if v['state']}

    @property
    def _events_in_use(self):
        return {k: v for k, v in self._events.items() if v['in_use']}

    def get_condition(self, raw, condition=None, stim_channel=None, mask=None,
                      uint_cast=None, mask_type='and', delayed_lookup=True):
        """Get averaging parameters for a condition (averaging category).

        Output is designed to be used with the Epochs class to extract the
        corresponding epochs.

        Parameters
        ----------
        raw : Raw object
            An instance of Raw.
        condition : None | str | dict | list of dict
            Condition or a list of conditions. Conditions can be strings
            (DACQ comment field, e.g. 'Auditory left') or category dicts
            (e.g. acqp['Auditory left'], where acqp is an instance of
            AcqParserFIF). If None, get all conditions marked active in
            DACQ.
        stim_channel : None | string | list of string
            Name of the stim channel or all the stim channels
            affected by the trigger. If None, the config variables
            'MNE_STIM_CHANNEL', 'MNE_STIM_CHANNEL_1', 'MNE_STIM_CHANNEL_2',
            etc. are read. If these are not found, it will fall back to
            'STI101' or 'STI 014' if present, then fall back to the first
            channel of type 'stim', if present.
        mask : int | None
            The value of the digital mask to apply to the stim channel values.
            If None (default), no masking is performed.
        uint_cast : bool
            If True (default False), do a cast to ``uint16`` on the channel
            data. This can be used to fix a bug with STI101 and STI014 in
            Neuromag acquisition setups that use channel STI016 (channel 16
            turns data into e.g. -32768), similar to ``mne_fix_stim14 --32``
            in MNE-C.
        mask_type: 'and' | 'not_and'
            The type of operation between the mask and the trigger.
            Choose 'and' for MNE-C masking behavior.
        delayed_lookup: bool
            If True, use the 'delayed lookup' procedure implemented in Elekta
            software. When a trigger transition occurs, the lookup of
            the new trigger value will not happen immediately at the following
            sample, but with a 1-sample delay. This allows a slight
            asynchrony between trigger onsets, when they are intended to be
            synchronous. If you have accurate hardware and want to detect
            transitions with a resolution of one sample, use
            delayed_lookup=False.

        Returns
        -------
        conds_data : dict or list of dict
            Each dict has the following keys:

            events : array, shape (n_epochs_out, 3)
                List of zero time points (t0) for the epochs matching the
                condition. Use as the ``events`` parameter to Epochs. Note
                that these are not (necessarily) actual events.
            event_id : dict
                Name of condition and index compatible with ``events``.
                Should be passed as the ``event_id`` parameter to Epochs.
            tmin : float
                Epoch starting time relative to t0. Use as the ``tmin``
                parameter to Epochs.
            tmax : float
                Epoch ending time relative to t0. Use as the ``tmax``
                parameter to Epochs.

        """
        if condition is None:
            condition = self.categories  # get all
        if not isinstance(condition, list):
            condition = [condition]  # single cond -> listify
        conds_data = list()
        for cat in condition:
            if isinstance(cat, str):
                cat = self[cat]
            mne_events = find_events(raw, stim_channel=stim_channel, mask=mask,
                                     mask_type=mask_type, output='step',
                                     uint_cast=uint_cast, consecutive=True,
                                     verbose=False, shortest_event=1)
            if delayed_lookup:
                ind = np.where(np.diff(mne_events[:, 0]) == 1)[0]
                if 1 in np.diff(ind):
                    raise ValueError('There are several subsequent '
                                     'transitions on the trigger channel. '
                                     'This will not work well with '
                                     'delayed_lookup=True. You may want to '
                                     'check your trigger data and '
                                     'set delayed_lookup=False.')
                mne_events[ind, 2] = mne_events[ind + 1, 2]
                mne_events = np.delete(mne_events, ind + 1, axis=0)
            sfreq = raw.info['sfreq']
            cat_t0_ = self._mne_events_to_category_t0(cat, mne_events, sfreq)
            # make it compatible with the usual events array
            cat_t0 = np.c_[cat_t0_, np.zeros(cat_t0_.shape),
                           cat['index'] * np.ones(cat_t0_.shape)
                           ].astype(np.uint32)
            cat_id = {cat['comment']: cat['index']}
            tmin, tmax = cat['start'], cat['end']
            conds_data.append(dict(events=cat_t0, event_id=cat_id,
                                   tmin=tmin, tmax=tmax))
        return conds_data[0] if len(conds_data) == 1 else conds_data
