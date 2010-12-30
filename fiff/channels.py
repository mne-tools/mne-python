from .tree import dir_tree_find
from .tag import find_tag
from .constants import FIFF


def _read_bad_channels(fid, node):
    """Read bad channels

    It returns the list of channel's names.
    """

    nodes = dir_tree_find(node, FIFF.FIFFB_MNE_BAD_CHANNELS)

    bads = [];
    if len(nodes) > 0:
        for node in nodes:
            tag = find_tag(fid, node, FIFF.FIFF_MNE_CH_NAME_LIST)
            if tag.data is not None:
                bads = tag.data.split(':')
    return bads

