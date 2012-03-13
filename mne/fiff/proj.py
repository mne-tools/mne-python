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
from .pick import pick_types


class Projection(dict):
    """Projection vector

    A basic class to proj a meaningful print for projection vectors.
    """
    def __repr__(self):
        s = "%s" % self['desc']
        s += ", active : %s " % self['active']
        s += ", nb of channels : %s " % self['data']['ncol']
        return "Projection (%s)" % s


def proj_equal(a, b):
    """ Test if two projectors are equal """

    equal = a['active'] == b['active']\
            and a['kind'] == b['kind']\
            and a['desc'] == b['desc']\
            and a['data']['col_names'] == b['data']['col_names']\
            and a['data']['row_names'] == b['data']['row_names']\
            and a['data']['ncol'] == b['data']['ncol']\
            and a['data']['nrow'] == b['data']['nrow']\
            and np.all(a['data']['data'] == b['data']['data'])

    return equal


def read_proj(fid, node):
    """Read spatial projections from a FIF file.

    Parameters
    ----------
    fid: file
        The file descriptor of the open file

    node: tree node
        The node of the tree where to look

    Returns
    -------
    projs: dict
        The list of projections
    """
    projs = list()

    #   Locate the projection data
    nodes = dir_tree_find(node, FIFF.FIFFB_PROJ)
    if len(nodes) == 0:
        return projs

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
        one = Projection(kind=kind, active=active, desc=desc,
                    data=dict(nrow=nvec, ncol=nchan, row_names=None,
                              col_names=names, data=data))

        projs.append(one)

    if len(projs) > 0:
        print '    Read a total of %d projection items:' % len(projs)
        for k in range(len(projs)):
            if projs[k]['active']:
                misc = 'active'
            else:
                misc = ' idle'
            print '        %s (%d x %d) %s' % (projs[k]['desc'],
                                        projs[k]['data']['nrow'],
                                        projs[k]['data']['ncol'],
                                        misc)

    return projs

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


def compute_spatial_vectors(epochs, n_grad=2, n_mag=2, n_eeg=2):
    """Compute SSP (spatial space projection) vectors

    Parameters
    ----------
    epochs: instance of Epochs
        The epochs containing the artifact
    n_grad: int
        Number of vectors for gradiometers
    n_mag: int
        Number of vectors for gradiometers
    n_eeg: int
        Number of vectors for gradiometers

    Returns
    -------
    projs: list
        List of projection vectors
    """
    data = sum(np.dot(e, e.T) for e in epochs)  # compute data covariance

    mag_ind = pick_types(epochs.info, meg='mag')
    grad_ind = pick_types(epochs.info, meg='grad')
    eeg_ind = pick_types(epochs.info, meg=False, eeg=True)

    if (n_grad > 0) and len(grad_ind) == 0:
        print "No gradiometers found. Forcing n_grad to 0"
        n_grad = 0
    if (n_mag > 0) and len(mag_ind) == 0:
        print "No magnetometers found. Forcing n_mag to 0"
        n_mag = 0
    if (n_eeg > 0) and len(eeg_ind) == 0:
        print "No EEG channels found. Forcing n_eeg to 0"
        n_eeg = 0

    grad_names, mag_names, eeg_names = ([epochs.ch_names[k] for k in ind]
                                     for ind in [grad_ind, mag_ind, eeg_ind])

    event_id = epochs.event_id
    projs = []
    for n, ind, names, desc in zip([n_grad, n_mag, n_eeg],
                      [grad_ind, mag_ind, eeg_ind],
                      [grad_names, mag_names, eeg_names],
                      ['planar', 'axial', 'eeg']):
        if n == 0:
            continue
        data_ind = data[ind][:, ind]
        U = linalg.svd(data_ind, full_matrices=False,
                                         overwrite_a=True)[0][:, :n]
        for k, u in enumerate(U.T):
            proj_data = dict(col_names=names, row_names=None,
                             data=u[np.newaxis, :], nrow=1, ncol=u.size)
            this_desc = "%s-%-d-%-.3f-%-.3f-PCA-%02d" % (desc, event_id,
                                             epochs.tmin, epochs.tmax, k + 1)
            print "Adding projection: %s" % this_desc
            proj = dict(active=True, data=proj_data, desc=this_desc, kind=1)
            projs.append(proj)

    return projs
