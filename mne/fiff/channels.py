# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from .tree import dir_tree_find
from .tag import find_tag
from .constants import FIFF


def _read_bad_channels(fid, node):
    """Read bad channels

    Parameters
    ----------
    fid: file
        The file descriptor

    node: dict
        The node of the FIF tree that contains info on the bad channels

    Returns
    -------
    bads: list
        A list of bad channel's names
    """
    nodes = dir_tree_find(node, FIFF.FIFFB_MNE_BAD_CHANNELS)

    bads = []
    if len(nodes) > 0:
        for node in nodes:
            tag = find_tag(fid, node, FIFF.FIFF_MNE_CH_NAME_LIST)
            if tag.data is not None:
                bads = tag.data.split(':')
    return bads
