# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


from ..utils import logger, verbose
from .constants import FIFF
from .tag import read_tag


def dir_tree_find(tree, kind):
    """Find nodes of the given kind from a directory tree structure.

    Parameters
    ----------
    tree : dict
        Directory tree.
    kind : int
        Kind to find.

    Returns
    -------
    nodes : list
        List of matching nodes.
    """
    nodes = []

    if isinstance(tree, list):
        for t in tree:
            nodes += dir_tree_find(t, kind)
    else:
        #   Am I desirable myself?
        if tree["block"] == kind:
            nodes.append(tree)

        #   Search the subtrees
        for child in tree["children"]:
            nodes += dir_tree_find(child, kind)
    return nodes


@verbose
def make_dir_tree(fid, directory, start=0, indent=0, verbose=None):
    """Create the directory tree structure."""
    if directory[start].kind == FIFF.FIFF_BLOCK_START:
        tag = read_tag(fid, directory[start].pos)
        block = tag.data.item()
    else:
        block = 0

    start_separate = False

    this = start

    tree = dict()
    tree["block"] = block
    tree["id"] = None
    tree["parent_id"] = None
    tree["nent"] = 0
    tree["nchild"] = 0
    tree["directory"] = directory[this]
    tree["children"] = []

    while this < len(directory):
        if directory[this].kind == FIFF.FIFF_BLOCK_START:
            if this != start:
                if not start_separate:
                    start_separate = True
                    logger.debug("    " * indent + f"start {{ {block}")
                child, this = make_dir_tree(fid, directory, this, indent + 1)
                tree["nchild"] += 1
                tree["children"].append(child)
        elif directory[this].kind == FIFF.FIFF_BLOCK_END:
            tag = read_tag(fid, directory[start].pos)
            if tag.data == block:
                break
        else:
            tree["nent"] += 1
            if tree["nent"] == 1:
                tree["directory"] = list()
            tree["directory"].append(directory[this])

            #  Add the id information if available
            if block == 0:
                if directory[this].kind == FIFF.FIFF_FILE_ID:
                    tag = read_tag(fid, directory[this].pos)
                    tree["id"] = tag.data
            else:
                if directory[this].kind == FIFF.FIFF_BLOCK_ID:
                    tag = read_tag(fid, directory[this].pos)
                    tree["id"] = tag.data
                elif directory[this].kind == FIFF.FIFF_PARENT_BLOCK_ID:
                    tag = read_tag(fid, directory[this].pos)
                    tree["parent_id"] = tag.data

        this += 1

    # Eliminate the empty directory
    if tree["nent"] == 0:
        tree["directory"] = None

    content = f"block = {tree['block']} nent = {tree['nent']} nchild = {tree['nchild']}"
    if start_separate:
        logger.debug("    " * indent + f"end }} {content}")
    else:
        logger.debug("    " * indent + content)
    last = this
    return tree, last
