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
from . import verbose


def pick_events(events, include=None, exclude=None):
    """Select some events

    Parameters
    ----------
    include: int | list | None
        A event id to include or a list of them.
        If None all events are included.
    exclude: int | list | None
        A event id to exclude or a list of them.
        If None no event is excluded. If include is not None
        the exclude parameter is ignored.

    Returns
    -------
    events: array, shape (n_events, 3)
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

    event_list = event_list.reshape(len(event_list) / 3, 3)
    return event_list


def read_events(filename, include=None, exclude=None):
    """Reads events from fif or text file

    Parameters
    ----------
    filename: string
        Name of the input file.
        If the extension is .fif, events are read assuming
        the file is in FIF format, otherwise (e.g., .eve,
        .lst, .txt) events are read as coming from text.
        Note that new format event files do not contain
        the "time" column (used to be the second column).
    include: int | list | None
        A event id to include or a list of them.
        If None all events are included.
    exclude: int | list | None
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
        event_list = _read_events_fif(fid, tree)
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


@verbose
def find_events(raw, stim_channel='STI 014', verbose=None):
    """Find events from raw file

    Parameters
    ----------
    raw : Raw object
        The raw data.
    stim_channel : string or list of string
        Name of the stim channel or all the stim channels
        affected by the trigger.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    events : array
        The array of event onsets in time samples.
    """
    if not isinstance(stim_channel, list):
        stim_channel = [stim_channel]

    pick = pick_channels(raw.info['ch_names'], include=stim_channel,
                         exclude=[])
    if len(pick) == 0:
        raise ValueError('No stim channel found to extract event triggers.')
    data, times = raw[pick, :]
    if np.any(data < 0):
        logger.warn('Trigger channel contains negative values. '
                    'Taking absolute value.')
        data = np.abs(data)  # make sure trig channel is positive
    data = data.astype(np.int)
    idx = np.where(np.all(np.diff(data, axis=1) > 0, axis=0))[0]
    events_id = data[0, idx + 1].astype(np.int)
    idx += raw.first_samp + 1
    events = np.c_[idx, np.zeros_like(idx), events_id]
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
        stop = min([stop[0] + raw.fist_samp, raw.last_samp + 1])
    else:
        stop = raw.last_samp + 1
    # Make sure we don't go out the end of the file:
    stop -= np.ceil(raw.info['sfreq'] * duration)
    ts = np.arange(start, stop, raw.info['sfreq'] * duration).astype(int)
    n_events = len(ts)
    events = np.c_[ts, np.zeros(n_events), id * np.ones(n_events)]
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
