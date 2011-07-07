# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from .constants import FIFF
from .tag import find_tag, has_tag


def _transpose_named_matrix(mat):
    """Transpose mat inplace (no copy)
    """
    mat['nrow'], mat['ncol'] = mat['ncol'], mat['nrow']
    mat['row_names'], mat['col_names'] = mat['col_names'], mat['row_names']
    mat['data'] = mat['data'].T
    return mat


def _read_named_matrix(fid, node, matkind):
    """Read named matrix from the given node

    Parameters
    ----------
    fid: file
        The opened file descriptor
    node: dict
        The node in the tree
    matkind: int
        The type of matrix

    Returns
    -------
    mat: dict
        The matrix data
    """
    #   Descend one level if necessary
    if node['block'] != FIFF.FIFFB_MNE_NAMED_MATRIX:
        for k in range(node['nchild']):
            if node['children'][k]['block'] == FIFF.FIFFB_MNE_NAMED_MATRIX:
                if has_tag(node['children'][k], matkind):
                    node = node['children'][k]
                    break
        else:
            raise ValueError('Desired named matrix (kind = %d) not available'
                             % matkind)
    else:
        if not has_tag(node, matkind):
            raise ValueError('Desired named matrix (kind = %d) not available'
                             % matkind)

    #   Read everything we need
    tag = find_tag(fid, node, matkind)
    if tag is None:
        raise ValueError('Matrix data missing')
    else:
        data = tag.data

    nrow, ncol = data.shape
    tag = find_tag(fid, node, FIFF.FIFF_MNE_NROW)
    if tag is not None:
        if tag.data != nrow:
            raise ValueError('Number of rows in matrix data and FIFF_MNE_NROW '
                             'tag do not match')

    tag = find_tag(fid, node, FIFF.FIFF_MNE_NCOL)
    if tag is not None:
        if tag.data != ncol:
            raise ValueError('Number of columns in matrix data and '
                             'FIFF_MNE_NCOL tag do not match')

    tag = find_tag(fid, node, FIFF.FIFF_MNE_ROW_NAMES)
    if tag is not None:
        row_names = tag.data.split(':')
    else:
        row_names = []

    tag = find_tag(fid, node, FIFF.FIFF_MNE_COL_NAMES)
    if tag is not None:
        col_names = tag.data.split(':')
    else:
        col_names = []

    mat = dict(nrow=nrow, ncol=ncol, row_names=row_names, col_names=col_names,
               data=data)
    return mat
