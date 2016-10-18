# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np

from .constants import FIFF
from .tag import Tag
from .tag import read_tag
from .write import write_id, start_block, end_block, _write
from ..utils import logger, verbose


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
        if tree['block'] == kind:
            nodes.append(tree)

        #   Search the subtrees
        for child in tree['children']:
            nodes += dir_tree_find(child, kind)
    return nodes


@verbose
def make_dir_tree(fid, directory, start=0, indent=0, verbose=None):
    """Create the directory tree structure."""
    FIFF_BLOCK_START = 104
    FIFF_BLOCK_END = 105
    FIFF_FILE_ID = 100
    FIFF_BLOCK_ID = 103
    FIFF_PARENT_BLOCK_ID = 110

    if directory[start].kind == FIFF_BLOCK_START:
        tag = read_tag(fid, directory[start].pos)
        block = tag.data
    else:
        block = 0

    logger.debug('    ' * indent + 'start { %d' % block)

    this = start

    tree = dict()
    tree['block'] = block
    tree['id'] = None
    tree['parent_id'] = None
    tree['nent'] = 0
    tree['nchild'] = 0
    tree['directory'] = directory[this]
    tree['children'] = []

    while this < len(directory):
        if directory[this].kind == FIFF_BLOCK_START:
            if this != start:
                child, this = make_dir_tree(fid, directory, this, indent + 1)
                tree['nchild'] += 1
                tree['children'].append(child)
        elif directory[this].kind == FIFF_BLOCK_END:
            tag = read_tag(fid, directory[start].pos)
            if tag.data == block:
                break
        else:
            tree['nent'] += 1
            if tree['nent'] == 1:
                tree['directory'] = list()
            tree['directory'].append(directory[this])

            #  Add the id information if available
            if block == 0:
                if directory[this].kind == FIFF_FILE_ID:
                    tag = read_tag(fid, directory[this].pos)
                    tree['id'] = tag.data
            else:
                if directory[this].kind == FIFF_BLOCK_ID:
                    tag = read_tag(fid, directory[this].pos)
                    tree['id'] = tag.data
                elif directory[this].kind == FIFF_PARENT_BLOCK_ID:
                    tag = read_tag(fid, directory[this].pos)
                    tree['parent_id'] = tag.data

        this += 1

    # Eliminate the empty directory
    if tree['nent'] == 0:
        tree['directory'] = None

    logger.debug('    ' * (indent + 1) + 'block = %d nent = %d nchild = %d'
                 % (tree['block'], tree['nent'], tree['nchild']))
    logger.debug('    ' * indent + 'end } %d' % block)
    last = this
    return tree, last


###############################################################################
# Writing

def copy_tree(fidin, in_id, nodes, fidout):
    """Copy directory subtrees from fidin to fidout."""
    if len(nodes) <= 0:
        return

    if not isinstance(nodes, list):
        nodes = [nodes]

    for node in nodes:
        start_block(fidout, node['block'])
        if node['id'] is not None:
            if in_id is not None:
                write_id(fidout, FIFF.FIFF_PARENT_FILE_ID, in_id)

            write_id(fidout, FIFF.FIFF_BLOCK_ID, in_id)
            write_id(fidout, FIFF.FIFF_PARENT_BLOCK_ID, node['id'])

        if node['directory'] is not None:
            for d in node['directory']:
                #   Do not copy these tags
                if d.kind == FIFF.FIFF_BLOCK_ID or \
                        d.kind == FIFF.FIFF_PARENT_BLOCK_ID or \
                        d.kind == FIFF.FIFF_PARENT_FILE_ID:
                    continue

                #   Read and write tags, pass data through transparently
                fidin.seek(d.pos, 0)

                s = fidin.read(4 * 4)
                tag = Tag(*np.fromstring(s, dtype=('>i4,>I4,>i4,>i4'))[0])
                tag.data = np.fromstring(fidin.read(tag.size), dtype='>B')

                _write(fidout, tag.data, tag.kind, 1, tag.type, '>B')

        for child in node['children']:
            copy_tree(fidin, in_id, child, fidout)

        end_block(fidout, node['block'])
