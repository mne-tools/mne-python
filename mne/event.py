"""IO with fif files containing events
"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np

from .fiff.constants import FIFF
from .fiff.tree import dir_tree_find
from .fiff.tag import read_tag
from .fiff.open import fiff_open
from .fiff.write import write_int, start_block, start_file, end_block, end_file
from .fiff.pick import pick_channels


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


def read_events(filename, include=None, exclude=None):
    """Reads events from fif file

    Parameters
    ----------
    filename: string
        name of the fif file
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

    fid, tree, _ = fiff_open(filename)

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
            fid.close()
            break
    else:
        fid.close()
        raise ValueError('Could not find any events')

    event_list = event_list.reshape(len(event_list) / 3, 3)

    event_list = pick_events(event_list, include, exclude)

    return event_list


def write_events(filename, event_list):
    """Write events to file

    Parameters
    ----------
    filename: string
        name of the fif file

    events: array, shape (n_events, 3)
        The list of events
    """
    #   Start writing...
    fid = start_file(filename)

    start_block(fid, FIFF.FIFFB_MNE_EVENTS)
    write_int(fid, FIFF.FIFF_MNE_EVENT_LIST, event_list.T)
    end_block(fid, FIFF.FIFFB_MNE_EVENTS)

    end_file(fid)


def find_events(raw, stim_channel='STI 014'):
    """Find events from raw file

    Parameters
    ----------
    raw : Raw object
        The raw data

    stim_channel : string or list of string
        Name of the stim channel or all the stim channels
        affected by the trigger.

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
    idx = np.where(np.all(np.diff(data, axis=1) > 0, axis=0))[0]
    events_id = data[0, idx + 1].astype(np.int)
    idx += raw.first_samp + 1
    events = np.c_[idx, np.zeros_like(idx), events_id]
    return events


def merge_events(events, ids, new_id):
    """Merge a set of events

    Parameters
    ----------
    events : array
        Events
    ids : array of int
        The ids of events to merge
    new_id : int
        The new id

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
