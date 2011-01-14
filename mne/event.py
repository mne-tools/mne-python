"""IO with fif files containing events
"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD Style.

from .fiff.constants import FIFF
from .fiff.tree import dir_tree_find
from .fiff.tag import read_tag
from .fiff.open import fiff_open
from .fiff.write import write_int, start_block, start_file, end_block, end_file


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
        raise ValueError, 'Could not find event data'

    events = events[0]

    for d in events.directory:
        kind = d.kind
        pos = d.pos
        if kind == FIFF.FIFF_MNE_EVENT_LIST:
            tag = read_tag(fid, pos)
            event_list = tag.data
            fid.close()
            break
    else:
        fid.close()
        raise ValueError, 'Could not find any events'

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
    write_int(fid, FIFF.FIFF_MNE_EVENT_LIST, event_list)
    end_block(fid, FIFF.FIFFB_MNE_EVENTS)

    end_file(fid)
