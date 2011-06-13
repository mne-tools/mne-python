# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from math import sqrt
import numpy as np
from scipy import linalg

from .tree import dir_tree_find
from .constants import FIFF
from .tag import find_tag


def read_proj(fid, node):
    """Read a projection operator from a FIF file.

    Parameters
    ----------
    fid: file
        The file descriptor of the open file

    node: tree node
        The node of the tree where to look

    Returns
    -------
    projdata: dict
        The projection operator
    """

    projdata = []

    #   Locate the projection data
    nodes = dir_tree_find(node, FIFF.FIFFB_PROJ)
    if len(nodes) == 0:
        return projdata

    tag = find_tag(fid, nodes[0], FIFF.FIFF_NCHAN)
    if tag is not None:
        global_nchan = int(tag.data)

    items = dir_tree_find(nodes[0], FIFF.FIFFB_PROJ_ITEM)
    for i in range(len(items)):

        #   Find all desired tags in one item
        item = items[i]
        tag = find_tag(fid, item, FIFF.FIFF_NCHAN)
        if tag is not None:
            nchan = int(tag.data)
        else:
            nchan = global_nchan

        tag = find_tag(fid, item, FIFF.FIFF_DESCRIPTION)
        if tag is not None:
            desc = tag.data
        else:
            tag = find_tag(fid, item, FIFF.FIFF_NAME)
            if tag is not None:
                desc = tag.data
            else:
                raise ValueError('Projection item description missing')

        # XXX : is this useful ?
        # tag = find_tag(fid, item, FIFF.FIFF_PROJ_ITEM_CH_NAME_LIST)
        # if tag is not None:
        #     namelist = tag.data
        # else:
        #     raise ValueError('Projection item channel list missing')

        tag = find_tag(fid, item, FIFF.FIFF_PROJ_ITEM_KIND)
        if tag is not None:
            kind = int(tag.data)
        else:
            raise ValueError('Projection item kind missing')

        tag = find_tag(fid, item, FIFF.FIFF_PROJ_ITEM_NVEC)
        if tag is not None:
            nvec = int(tag.data)
        else:
            raise ValueError('Number of projection vectors not specified')

        tag = find_tag(fid, item, FIFF.FIFF_PROJ_ITEM_CH_NAME_LIST)
        if tag is not None:
            names = tag.data.split(':')
        else:
            raise ValueError('Projection item channel list missing')

        tag = find_tag(fid, item, FIFF.FIFF_PROJ_ITEM_VECTORS)
        if tag is not None:
            data = tag.data
        else:
            raise ValueError('Projection item data missing')

        tag = find_tag(fid, item, FIFF.FIFF_MNE_PROJ_ITEM_ACTIVE)
        if tag is not None:
            active = True
        else:
            active = False

        if data.shape[1] != len(names):
            raise ValueError('Number of channel names does not match the '
                             'size of data matrix')

        #   Use exactly the same fields in data as in a named matrix
        one = dict(kind=kind, active=active, desc=desc,
                    data=dict(nrow=nvec, ncol=nchan, row_names=None,
                              col_names=names, data=data))

        projdata.append(one)

    if len(projdata) > 0:
        print '\tRead a total of %d projection items:' % len(projdata)
        for k in range(len(projdata)):
            if projdata[k]['active']:
                misc = 'active'
            else:
                misc = ' idle'
            print '\t\t%s (%d x %d) %s' % (projdata[k]['desc'],
                                        projdata[k]['data']['nrow'],
                                        projdata[k]['data']['ncol'],
                                        misc)

    return projdata

###############################################################################
# Write

from .write import write_int, write_float, write_string, write_name_list, \
                   write_float_matrix, end_block, start_block


def write_proj(fid, projs):
    """Write a projection operator to a file.

    Parameters
    ----------
    fid: file
        The file descriptor of the open file

    projs: dict
        The projection operator

    """
    start_block(fid, FIFF.FIFFB_PROJ)

    for proj in projs:
        start_block(fid, FIFF.FIFFB_PROJ_ITEM)
        write_string(fid, FIFF.FIFF_NAME, proj['desc'])
        write_int(fid, FIFF.FIFF_PROJ_ITEM_KIND, proj['kind'])
        if proj['kind'] == FIFF.FIFFV_PROJ_ITEM_FIELD:
            write_float(fid, FIFF.FIFF_PROJ_ITEM_TIME, 0.0)

        write_int(fid, FIFF.FIFF_NCHAN, proj['data']['ncol'])
        write_int(fid, FIFF.FIFF_PROJ_ITEM_NVEC, proj['data']['nrow'])
        write_int(fid, FIFF.FIFF_MNE_PROJ_ITEM_ACTIVE, proj['active'])
        write_name_list(fid, FIFF.FIFF_PROJ_ITEM_CH_NAME_LIST,
                             proj['data']['col_names'])
        write_float_matrix(fid, FIFF.FIFF_PROJ_ITEM_VECTORS,
                           proj['data']['data'])
        end_block(fid, FIFF.FIFFB_PROJ_ITEM)

    end_block(fid, FIFF.FIFFB_PROJ)


###############################################################################
# Utils

def make_projector(projs, ch_names, bads=[]):
    """Create an SSP operator from SSP projection vectors

    Parameters
    ----------
    projs : list
        List of projection vectors
    ch_names : list of strings
        List of channels to include in the projection matrix
    bads : list of strings
        Some bad channels to exclude

    Returns
    -------
    proj : array of shape [n_channels, n_channels]
        The projection operator to apply to the data
    nproj : int
        How many items in the projector
    U : array
        The orthogonal basis of the projection vectors (optional)
    """
    nchan = len(ch_names)
    if nchan == 0:
        raise ValueError('No channel names specified')

    proj = np.eye(nchan, nchan)
    nproj = 0
    U = []

    #   Check trivial cases first
    if projs is None:
        return proj, nproj, U

    nactive = 0
    nvec = 0
    for p in projs:
        if p['active']:
            nactive += 1
            nvec += p['data']['nrow']

    if nactive == 0:
        return proj, nproj, U

    #   Pick the appropriate entries
    vecs = np.zeros((nchan, nvec))
    nvec = 0
    nonzero = 0
    for k, p in enumerate(projs):
        if p['active']:
            if len(p['data']['col_names']) != \
                        len(np.unique(p['data']['col_names'])):
                raise ValueError('Channel name list in projection item %d'
                                 ' contains duplicate items' % k)

            # Get the two selection vectors to pick correct elements from
            # the projection vectors omitting bad channels
            sel = []
            vecsel = []
            for c, name in enumerate(ch_names):
                if name in p['data']['col_names']:
                    sel.append(c)
                    vecsel.append(p['data']['col_names'].index(name))

            # If there is something to pick, pickit
            if len(sel) > 0:
                for v in range(p['data']['nrow']):
                    vecs[sel, nvec + v] = p['data']['data'][v, vecsel].T

            # Rescale for better detection of small singular values
            for v in range(p['data']['nrow']):
                psize = sqrt(np.sum(vecs[:, nvec + v] * vecs[:, nvec + v]))
                if psize > 0:
                    vecs[:, nvec + v] /= psize
                    nonzero += 1

            nvec += p['data']['nrow']

    #   Check whether all of the vectors are exactly zero
    if nonzero == 0:
        return proj, nproj, U

    # Reorthogonalize the vectors
    U, S, V = linalg.svd(vecs[:, :nvec], full_matrices=False)

    # Throw away the linearly dependent guys
    nproj = np.sum((S / S[0]) > 1e-2)
    U = U[:, :nproj]

    # Here is the celebrated result
    proj -= np.dot(U, U.T)

    return proj, nproj, U


def make_projector_info(info):
    """Make an SSP operator using the measurement info

    Calls make_projector on good channels.

    Parameters
    ----------
    info : dict
        Measurement info

    Returns
    -------
    proj : array of shape [n_channels, n_channels]
        The projection operator to apply to the data
    nproj : int
        How many items in the projector
    """
    proj, nproj, _ = make_projector(info['projs'], info['ch_names'],
                                    info['bads'])
    return proj, nproj
