# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy
from math import sqrt
import numpy as np
from scipy import linalg
from itertools import count

from .tree import dir_tree_find
from .tag import find_tag
from .constants import FIFF
from .pick import pick_types
from ..utils import logger, verbose


class Projection(dict):
    """Projection vector

    A basic class to proj a meaningful print for projection vectors.
    """
    def __repr__(self):
        s = "%s" % self['desc']
        s += ", active : %s" % self['active']
        s += ", n_channels : %s" % self['data']['ncol']
        return "<Projection  |  %s>" % s


class ProjMixin(object):
    """Mixin class for Raw, Evoked, Epochs
    """
    def add_proj(self, projs, remove_existing=False):
        """Add SSP projection vectors

        Parameters
        ----------
        projs : list
            List with projection vectors.
        remove_existing : bool
            Remove the projection vectors currently in the file.

        Returns
        -------
        self : instance of Raw | Epochs | Evoked
            The data container.
        """
        if isinstance(projs, Projection):
            projs = [projs]

        if (not isinstance(projs, list) and
                not all([isinstance(p, Projection) for p in projs])):
            raise ValueError('Only projs can be added. You supplied '
                             'something else.')

        # mark proj as inactive, as they have not been applied
        projs = deactivate_proj(projs, copy=True, verbose=self.verbose)
        if remove_existing:
            # we cannot remove the proj if they are active
            if any(p['active'] for p in self.info['projs']):
                raise ValueError('Cannot remove projectors that have '
                                 'already been applied')
            self.info['projs'] = projs
        else:
            self.info['projs'].extend(projs)

        return self

    def apply_proj(self):
        """Apply the signal space projection (SSP) operators to the data.

        Notes
        -----
        Once the projectors have been applied, they can no longer be
        removed. It is usually not recommended to apply the projectors at
        too early stages, as they are applied automatically later on
        (e.g. when computing inverse solutions).
        Hint: using the copy method individual projection vectors
        can be tested without affecting the original data.
        With evoked data, consider the following example::

            projs_a = mne.read_proj('proj_a.fif')
            projs_b = mne.read_proj('proj_b.fif')
            # add the first, copy, apply and see ...
            evoked.add_proj(a).copy().apply_proj().plot()
            # add the second, copy, apply and see ...
            evoked.add_proj(b).copy().apply_proj().plot()
            # drop the first and see again
            evoked.copy().del_proj(0).apply_proj().plot()
            evoked.apply_proj()  # finally keep both

        Returns
        -------
        self : instance of Raw | Epochs | Evoked
            The instance.
        """
        if self.info['projs'] is None:
            logger.info('No projector specified for this dataset.'
                        'Please consider the method self.add_proj.')
            return self

        if all([p['active'] for p in self.info['projs']]):
            logger.info('Projections have already been applied. Doing '
                        'nothing.')
            return self

        _projector, info = setup_proj(deepcopy(self.info), activate=True,
                                      verbose=self.verbose)
        # let's not raise a RuntimeError here, otherwise interactive plotting
        if _projector is None:  # won't be fun.
            logger.info('The projections don\'t apply to these data.'
                        ' Doing nothing.')
            return self

        self._projector, self.info = _projector, info
        self.proj = True  # track that proj were applied
        # handle different data / preload attrs and create reference
        # this also helps avoiding circular imports
        for attr in ('get_data', '_data', 'data'):
            data = getattr(self, attr, None)
            if data is None:
                continue
            elif callable(data):
                if self.preload:
                    data = np.empty_like(self._data)
                    for ii, e in enumerate(self._data):
                        data[ii] = self._preprocess(np.dot(self._projector, e),
                                                    self.verbose)
                else:  # get data knows what to do.
                    data = data()
            else:
                data = np.dot(self._projector, data)
            break
        logger.info('SSP projectors applied...')
        if hasattr(self, '_data'):
            self._data = data
        else:
            self.data = data

        return self

    def del_proj(self, idx):
        """Remove SSP projection vector

        Note: The projection vector can only be removed if it is inactive
              (has not been applied to the data).

        Parameters
        ----------
        idx : int
            Index of the projector to remove.

        Returns
        -------
        self : instance of Raw | Epochs | Evoked
        """
        if self.info['projs'][idx]['active']:
            raise ValueError('Cannot remove projectors that have already '
                             'been applied')

        self.info['projs'].pop(idx)

        return self


def proj_equal(a, b):
    """ Test if two projectors are equal """

    equal = (a['active'] == b['active'] and
             a['kind'] == b['kind'] and
             a['desc'] == b['desc'] and
             a['data']['col_names'] == b['data']['col_names'] and
             a['data']['row_names'] == b['data']['row_names'] and
             a['data']['ncol'] == b['data']['ncol'] and
             a['data']['nrow'] == b['data']['nrow'] and
             np.all(a['data']['data'] == b['data']['data']))
    return equal


@verbose
def _read_proj(fid, node, verbose=None):
    """Read spatial projections from a FIF file.

    Parameters
    ----------
    fid : file
        The file descriptor of the open file.
    node : tree node
        The node of the tree where to look.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    projs: dict
        The list of projections.
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

        # handle the case when data is transposed for some reason
        if data.shape[0] == len(names) and data.shape[1] == nvec:
            data = data.T

        if data.shape[1] != len(names):
            raise ValueError('Number of channel names does not match the '
                             'size of data matrix')

        #   Use exactly the same fields in data as in a named matrix
        one = Projection(kind=kind, active=active, desc=desc,
                         data=dict(nrow=nvec, ncol=nchan, row_names=None,
                                   col_names=names, data=data))

        projs.append(one)

    if len(projs) > 0:
        logger.info('    Read a total of %d projection items:' % len(projs))
        for k in range(len(projs)):
            if projs[k]['active']:
                misc = 'active'
            else:
                misc = ' idle'
            logger.info('        %s (%d x %d) %s'
                        % (projs[k]['desc'], projs[k]['data']['nrow'],
                           projs[k]['data']['ncol'], misc))

    return projs

###############################################################################
# Write

from .write import (write_int, write_float, write_string, write_name_list,
                    write_float_matrix, end_block, start_block)


def _write_proj(fid, projs):
    """Write a projection operator to a file.

    Parameters
    ----------
    fid : file
        The file descriptor of the open file.
    projs : dict
        The projection operator.
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
        List of projection vectors.
    ch_names : list of strings
        List of channels to include in the projection matrix.
    bads : list of strings
        Some bad channels to exclude. If bad channels were marked
        in the raw file when projs were calculated using mne-python,
        they should not need to be included here as they will
        have been automatically omitted from the projectors.
    include_active : bool
        Also include projectors that are already active.

    Returns
    -------
    proj : array of shape [n_channels, n_channels]
        The projection operator to apply to the data.
    nproj : int
        How many items in the projector.
    U : array
        The orthogonal basis of the projection vectors (optional).
    """
    nchan = len(ch_names)
    if nchan == 0:
        raise ValueError('No channel names specified')

    default_return = (np.eye(nchan, nchan), 0, [])

    #   Check trivial cases first
    if projs is None:
        return default_return

    nvec = 0
    nproj = 0
    for p in projs:
        if not p['active'] or include_active:
            nproj += 1
            nvec += p['data']['nrow']

    if nproj == 0:
        return default_return

    #   Pick the appropriate entries
    vecs = np.zeros((nchan, nvec))
    nvec = 0
    nonzero = 0
    for k, p in enumerate(projs):
        if not p['active'] or include_active:
            if (len(p['data']['col_names']) !=
                    len(np.unique(p['data']['col_names']))):
                raise ValueError('Channel name list in projection item %d'
                                 ' contains duplicate items' % k)

            # Get the two selection vectors to pick correct elements from
            # the projection vectors omitting bad channels
            sel = []
            vecsel = []
            for c, name in enumerate(ch_names):
                if name in p['data']['col_names'] and name not in bads:
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
        return default_return

    # Reorthogonalize the vectors
    U, S, V = linalg.svd(vecs[:, :nvec], full_matrices=False)

    # Throw away the linearly dependent guys
    nproj = np.sum((S / S[0]) > 1e-2)
    U = U[:, :nproj]

    # Here is the celebrated result
    proj = np.eye(nchan, nchan) - np.dot(U, U.T)

    return proj, nproj, U


def make_projector_info(info, include_active=True):
    """Make an SSP operator using the measurement info

    Calls make_projector on good channels.

    Parameters
    ----------
    info : dict
        Measurement info.
    include_active : bool
        Also include projectors that are already active.

    Returns
    -------
    proj : array of shape [n_channels, n_channels]
        The projection operator to apply to the data.
    nproj : int
        How many items in the projector.
    """
    proj, nproj, _ = make_projector(info['projs'], info['ch_names'],
                                    info['bads'], include_active)
    return proj, nproj


@verbose
def activate_proj(projs, copy=True, verbose=None):
    """Set all projections to active

    Useful before passing them to make_projector.

    Parameters
    ----------
    projs : list
        The projectors.
    copy : bool
        Modify projs in place or operate on a copy.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    projs : list
        The projectors.
    """
    if copy:
        projs = deepcopy(projs)

    #   Activate the projection items
    for proj in projs:
        proj['active'] = True

    logger.info('%d projection items activated' % len(projs))

    return projs


@verbose
def deactivate_proj(projs, copy=True, verbose=None):
    """Set all projections to inactive

    Useful before saving raw data without projectors applied.

    Parameters
    ----------
    projs : list
        The projectors.
    copy : bool
        Modify projs in place or operate on a copy.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    projs : list
        The projectors.
    """
    if copy:
        projs = deepcopy(projs)

    #   Deactivate the projection items
    for proj in projs:
        proj['active'] = False

    logger.info('%d projection items deactivated' % len(projs))

    return projs


@verbose
def make_eeg_average_ref_proj(info, activate=True, verbose=None):
    """Create an EEG average reference SSP projection vector

    Parameters
    ----------
    info : dict
        Measurement info.
    activate : bool
        If True projections are activated.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    eeg_proj: instance of Projection
        The SSP/PCA projector.
    """
    logger.info("Adding average EEG reference projection.")
    eeg_sel = pick_types(info, meg=False, eeg=True, ref_meg=False,
                         exclude='bads')
    ch_names = info['ch_names']
    eeg_names = [ch_names[k] for k in eeg_sel]
    n_eeg = len(eeg_sel)
    if n_eeg == 0:
        raise ValueError('Cannot create EEG average reference projector '
                         '(no EEG data found)')
    vec = np.ones((1, n_eeg)) / n_eeg
    eeg_proj_data = dict(col_names=eeg_names, row_names=None,
                         data=vec, nrow=1, ncol=n_eeg)
    eeg_proj = Projection(active=activate, data=eeg_proj_data,
                          desc='Average EEG reference',
                          kind=FIFF.FIFFV_MNE_PROJ_ITEM_EEG_AVREF)
    return eeg_proj


def _has_eeg_average_ref_proj(projs):
    """Determine if a list of projectors has an average EEG ref"""
    for proj in projs:
        if proj['desc'] == 'Average EEG reference' or \
                proj['kind'] == FIFF.FIFFV_MNE_PROJ_ITEM_EEG_AVREF:
            return True
    return False


@verbose
def setup_proj(info, add_eeg_ref=True, activate=True,
               verbose=None):
    """Set up projection for Raw and Epochs

    Parameters
    ----------
    info : dict
        The measurement info.
    add_eeg_ref : bool
        If True, an EEG average reference will be added (unless one
        already exists).
    activate : bool
        If True projections are activated.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    projector : array of shape [n_channels, n_channels]
        The projection operator to apply to the data.
    info : dict
        The modified measurement info (Warning: info is modified inplace).
    """
    # Add EEG ref reference proj if necessary
    eeg_sel = pick_types(info, meg=False, eeg=True, ref_meg=False,
                         exclude='bads')
    if len(eeg_sel) > 0 and not _has_eeg_average_ref_proj(info['projs']) \
            and add_eeg_ref is True:
        eeg_proj = make_eeg_average_ref_proj(info, activate=activate)
        info['projs'].append(eeg_proj)

    #   Create the projector
    projector, nproj = make_projector_info(info)
    if nproj == 0:
        if verbose:
            logger.info('The projection vectors do not apply to these '
                        'channels')
        projector = None
    else:
        logger.info('Created an SSP operator (subspace dimension = %d)'
                    % nproj)

    #   The projection items have been activated
    if activate:
        info['projs'] = activate_proj(info['projs'], copy=False)

    return projector, info


def _uniquify_projs(projs):
    """Aux function"""
    final_projs = []
    for proj in projs:  # flatten
        if not any([proj_equal(p, proj) for p in final_projs]):
            final_projs.append(proj)

    my_count = count(len(final_projs))

    def sorter(x):
        """sort in a nice way"""
        digits = [s for s in x['desc'] if s.isdigit()]
        if digits:
            sort_idx = int(digits[-1])
        else:
            sort_idx = next(my_count)
        return (sort_idx, x['desc'])

    return sorted(final_projs, key=sorter)
