# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from copy import deepcopy
from math import sqrt
import numpy as np
from scipy import linalg

from .tree import dir_tree_find
from .constants import FIFF
from .tag import find_tag
from .pick import pick_types
from ..utils import deprecated


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
            active = bool(tag.data)
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
        write_int(fid, FIFF.FIFF_NCHAN, proj['data']['ncol'])
        write_name_list(fid, FIFF.FIFF_PROJ_ITEM_CH_NAME_LIST,
                             proj['data']['col_names'])
        write_string(fid, FIFF.FIFF_NAME, proj['desc'])
        write_int(fid, FIFF.FIFF_PROJ_ITEM_KIND, proj['kind'])
        if proj['kind'] == FIFF.FIFFV_PROJ_ITEM_FIELD:
            write_float(fid, FIFF.FIFF_PROJ_ITEM_TIME, 0.0)

        write_int(fid, FIFF.FIFF_PROJ_ITEM_NVEC, proj['data']['nrow'])
        write_int(fid, FIFF.FIFF_MNE_PROJ_ITEM_ACTIVE, proj['active'])
        write_float_matrix(fid, FIFF.FIFF_PROJ_ITEM_VECTORS,
                           proj['data']['data'])
        end_block(fid, FIFF.FIFFB_PROJ_ITEM)

    end_block(fid, FIFF.FIFFB_PROJ)


###############################################################################
# Utils

def make_projector(projs, ch_names, bads=[], include_active=True):
    """Create an SSP operator from SSP projection vectors

    Parameters
    ----------
    projs : list
        List of projection vectors
    ch_names : list of strings
        List of channels to include in the projection matrix
    bads : list of strings
        Some bad channels to exclude
    include_active : bool
        Also include projectors that are already active.

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

    nvec = 0
    for p in projs:
        if not p['active'] or include_active:
            nproj += 1
            nvec += p['data']['nrow']

    if nproj == 0:
        return proj, nproj, U

    #   Pick the appropriate entries
    vecs = np.zeros((nchan, nvec))
    nvec = 0
    nonzero = 0
    for k, p in enumerate(projs):
        if not p['active'] or include_active:
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
        return proj, 0, U

    # Reorthogonalize the vectors
    U, S, V = linalg.svd(vecs[:, :nvec], full_matrices=False)

    # Throw away the linearly dependent guys
    nproj = np.sum((S / S[0]) > 1e-2)
    U = U[:, :nproj]

    # Here is the celebrated result
    proj -= np.dot(U, U.T)

    return proj, nproj, U


def make_projector_info(info, include_active=True):
    """Make an SSP operator using the measurement info

    Calls make_projector on good channels.

    Parameters
    ----------
    info : dict
        Measurement info
   include_active : bool
        Also include projectors that are already active.

    Returns
    -------
    proj : array of shape [n_channels, n_channels]
        The projection operator to apply to the data
    nproj : int
        How many items in the projector
    """
    proj, nproj, _ = make_projector(info['projs'], info['ch_names'],
                                    info['bads'], include_active)
    return proj, nproj


@deprecated("Use mne.compute_proj_epochs")
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
    import mne  # XXX : ugly due to circular mess in imports
    return mne.compute_proj_epochs(epochs, n_grad, n_mag, n_eeg)


def activate_proj(projs, copy=True):
    """Set all projections to active

    Useful before passing them to make_projector
    """
    if copy:
        projs = deepcopy(projs)

    #   Activate the projection items
    for proj in projs:
        proj['active'] = True

    print '%d projection items activated' % len(projs)

    return projs


def make_eeg_average_ref_proj(info):
    """Create an EEG average reference SSP projection vector

    Parameters
    ----------
    info: dict
        Measurement info

    Returns
    -------
    eeg_proj: instance of Projection
        The SSP/PCA projector
    """
    print "Adding average EEG reference projection."
    eeg_sel = pick_types(info, meg=False, eeg=True)
    ch_names = info['ch_names']
    eeg_names = [ch_names[k] for k in eeg_sel]
    n_eeg = len(eeg_sel)
    if n_eeg == 0:
        raise ValueError('Cannot create EEG average reference projector '
                         '(no EEG data found)')
    vec = np.ones((1, n_eeg)) / n_eeg
    eeg_proj_data = dict(col_names=eeg_names, row_names=None,
                         data=vec, nrow=1, ncol=n_eeg)
    eeg_proj = Projection(active=True, data=eeg_proj_data,
                    desc='Average EEG reference', kind=1)
    return eeg_proj
