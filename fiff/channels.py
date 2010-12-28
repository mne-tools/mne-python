from .tree import dir_tree_find
from .tag import find_tag
from .constants import FIFF


def read_bad_channels(fid, node):
    """
    %
    % [bads] = fiff_read_bad_channels(fid,node)
    %
    % Reas the bad channel list from a node if it exists
    %
    % fid      - The file id
    % node     - The node of interes
    %
    """

    nodes = dir_tree_find(node, FIFF.FIFFB_MNE_BAD_CHANNELS)

    bads = [];
    if len(nodes) > 0:
        for node in nodes:
            tag = find_tag(fid, node, FIFF.FIFF_MNE_CH_NAME_LIST)
            if tag is not None:
                bads = tag.data.split(':')
    return bads

