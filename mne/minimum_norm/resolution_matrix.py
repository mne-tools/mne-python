"""Coompute resolution matrix for linear estimators."""
from copy import deepcopy

import numpy as np

from mne import pick_channels
from mne.utils import logger
from mne.forward.forward import is_fixed_orient, convert_forward_solution

from mne.evoked import EvokedArray
from mne.minimum_norm import apply_inverse


def make_resolution_matrix(fwd, invop, method, lambda2):
    """Compute resolution matrix for linear inverse operator.

    Parameters
    ----------
    fwd: forward solution
        Used to get leadfield matrix.
    invop: inverse operator
        Inverse operator to get inverse matrix.
        pick_ori='normal' will be selected.
    method: string
        Inverse method to use (MNE, dSPM, sLORETA).
    lambda2: float
        The regularisation parameter.

    Returns
    -------
        resmat: 2D numpy array.
        Resolution matrix (inverse matrix times leadfield).
    """
    # make sure forward and inverse operator match
    fwd = _convert_forward_match_inv(fwd, invop)

    # don't include bad channels
    # only use good channels from inverse operator
    bads_inv = invop['info']['bads']

    # good channels
    ch_names = [c for c in invop['info']['ch_names'] if (c not in bads_inv)]

    # get leadfield matrix from forward solution
    leadfield = _pick_leadfield(fwd['sol']['data'], fwd, ch_names)

    invmat = _get_matrix_from_inverse_operator(invop, fwd,
                                               method=method, lambda2=lambda2)

    resmat = invmat.dot(leadfield)

    dims = resmat.shape

    print('Dimensions of resolution matrix: %d by %d.' % (dims[0], dims[1]))

    return resmat


def _convert_forward_match_inv(fwd, inv):
    """Helper to ensure forward and inverse operators match."""
    # did inverse operator use fixed orientation?
    is_fixed_inv = inv['eigen_leads']['data'].shape[0] == inv['nsource']

    # ...or loose orientation?
    if not is_fixed_inv:
        is_loose = not (inv['orient_prior']['data'] == 1.).all()

    # if inv op is fixed or loose, do the same with fwd
    if is_fixed_inv:
        if not is_fixed_orient(fwd):
            fwd = convert_forward_solution(fwd, force_fixed=True)
    elif is_loose:
        if not fwd['surf_ori']:
            fwd = convert_forward_solution(fwd, surf_ori=True)
    else:  # free orientation, change surface orientation for fwd
        if fwd['surf_ori']:
            fwd = convert_forward_solution(fwd, surf_ori=False)

    # The following finds a difference, probably because too sensitive
    # # check if source spaces for inv and fwd are the same
    # differences = object_diff(fwd['src'], inv['src'])

    # if differences:
    #     raise RuntimeError('fwd["src"] and inv["src"] did not match: %s'
    #                        % (differences,))
    return fwd


def _pick_leadfield(leadfield, forward, ch_names):
    """Helper to pick out correct lead field components."""
    picks_fwd = pick_channels(forward['info']['ch_names'], ch_names)
    return leadfield[picks_fwd]


def _prepare_info(inverse_operator):
    """Helper to get a usable dict."""
    # in order to convert sub-leadfield matrix to evoked data type (pretending
    # it's an epoch, see in loop below), uses 'info' from inverse solution
    # because this has all the correct projector information
    info = deepcopy(inverse_operator['info'])
    info['sfreq'] = 1000.  # necessary
    info['projs'] = inverse_operator['projs']
    return info


def _get_matrix_from_inverse_operator(inverse_operator, forward, method='dSPM',
                                      lambda2=1. / 9.):
    """Get inverse matrix from an inverse operator.

    Currently works only for fixed/loose orientation constraints
    For loose orientation constraint, the CTFs are computed for the normal
    component (pick_ori='normal').
    Parameters
    ----------
    inverse_operator : instance of InverseOperator
        The inverse operator.
    forward : dict
        The forward operator.
    method : 'MNE' | 'dSPM' | 'sLORETA'
        Inverse methods (for apply_inverse).
    lambda2 : float
        The regularization parameter (for apply_inverse).
    Returns
    -------
    invmat : ndarray
        Inverse matrix associated with inverse operator and specified
        parameters.
    """
    # make sure forward and inverse operators match
    _convert_forward_match_inv(forward, inverse_operator)

    print('Free Orientation version.')

    logger.info("Computing whole inverse operator.")

    info_inv = _prepare_info(inverse_operator)

    # only use channels that are good for inverse operator and forward sol
    ch_names_inv = info_inv['ch_names']
    n_chs_inv = len(ch_names_inv)
    bads_inv = inverse_operator['info']['bads']

    # indices of bad channels
    ch_idx_bads = [ch_names_inv.index(ch) for ch in bads_inv]

    # create identity matrix as input for inverse operator
    # set elements to zero for non-selected channels
    id_mat = np.eye(n_chs_inv)

    # convert identity matrix to evoked data type (pretending it's an epoch)
    ev_id = EvokedArray(id_mat, info=info_inv, tmin=0.)

    # apply inverse operator to identity matrix in order to get inverse matrix
    # free orientation constraint not possible because apply_inverse would
    # combine components

    # pick_ori='normal' required because apply_inverse won't give separate
    # orientations
    # if ~inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI:
    #     pick_ori = 'vector'
    # else:
    #     pick_ori = 'normal'

    # check if inverse operator uses fixed source orientations
    is_fixed_inv = inverse_operator['eigen_leads']['data'].shape[0] == \
        inverse_operator['nsource']

    # choose pick_ori according to inverse operator
    if is_fixed_inv:
        pick_ori = None
    else:
        pick_ori = 'vector'

    # columns for bad channels will be zero
    invmat_op = apply_inverse(ev_id, inverse_operator, lambda2=lambda2,
                              method=method, pick_ori=pick_ori)

    # turn source estimate into numpty array
    invmat = invmat_op.data

    dims = invmat.shape

    # remove columns for bad channels
    # take into account it may be 3D array
    invmat = np.delete(invmat, ch_idx_bads, axis=len(dims) - 1)

    # if 3D array, i.e. multiple values per location (fixed and loose),
    # reshape into 2D array
    if len(dims) == 3:
        v0o1 = invmat[0, 1].copy()
        v3o2 = invmat[3, 2].copy()
        invmat = invmat.reshape(dims[0] * dims[1], dims[2])
        # make sure that reshaping worked
        assert np.array_equal(v0o1, invmat[1])
        assert np.array_equal(v3o2, invmat[11])

    logger.info("Dimension of Inverse Matrix: %s" % str(invmat.shape))

    return invmat
