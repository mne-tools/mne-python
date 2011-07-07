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


def read_events(filename):
    """Reads events from fif file

    Parameters
    ----------
    filename: string
        name of the fif file

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

    stim_channel : string
        Name of the stim channel

    Returns
    -------
    events : array
        The array of event onsets in time samples.
    """

    pick = pick_channels(raw.info['ch_names'], include=['STI 014'],
                         exclude=[])
    if len(pick) == 0:
        raise ValueError('No stim channel found to extract event triggers.')
    data, times = raw[pick, :]
    data = data.ravel()
    idx = np.where(np.diff(data.ravel()) > 0)[0]
    events_id = data[idx + 1].astype(np.int)
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