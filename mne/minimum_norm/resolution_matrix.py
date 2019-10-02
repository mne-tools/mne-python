"""Compute resolution matrix for linear estimators."""
from copy import deepcopy

import numpy as np

from mne import pick_channels_forward, EvokedArray, SourceEstimate
from mne.io.constants import FIFF
from mne.utils import logger
from mne.forward.forward import is_fixed_orient, convert_forward_solution

from mne.minimum_norm import apply_inverse


def make_resolution_matrix(forward, inverse_operator, method='dSPM',
                           lambda2=1. / 9.):
    """Compute resolution matrix for linear inverse operator.

    Parameters
    ----------
    forward: dict
        Forward Operator.
    inverse_operator: Instance of InverseOperator
        Inverse operator.
    method: str
        Inverse method to use (MNE, dSPM, sLORETA).
    lambda2: float
        The regularisation parameter.

    Returns
    -------
    resmat: array, shape (n_dipoles, n_dipoles)
        Resolution matrix (inverse operator times forward operator).
    """
    # make sure forward and inverse operator match
    inv = inverse_operator
    fwd = _convert_forward_match_inv(forward, inv)

    # don't include bad channels
    # only use good channels from inverse operator
    bads_inv = inv['info']['bads']

    # good channels
    ch_names = [c for c in inv['info']['ch_names'] if (c not in bads_inv)]

    fwd = pick_channels_forward(fwd, ch_names, ordered=True)

    # get leadfield matrix from forward solution
    leadfield = fwd['sol']['data']

    invmat = _get_matrix_from_inverse_operator(inv, fwd,
                                               method=method, lambda2=lambda2)

    resmat = invmat.dot(leadfield)

    logger.info('Dimensions of resolution matrix: %d by %d.' % resmat.shape)

    return resmat


def get_psf_ctf_vertex(resmat, src, idx, func, norm=False):
    """Get point-spread (PSFs) or cross-talk (CTFs) functions for vertices.

    Parameters
        ----------
        resmat: array, shape (n_dipoles, n_dipoles)
            Forward Operator.
        src: Source Space
            Source space used to compute resolution matrix.
        idx: list of int
            Vertex indices for which PSFs or CTFs to produce.
        func: str ('psf' | 'ctf')
            Whether to produce PSFs or CTFs.
        norm: Bool
            Whether to normalise to maximum across all PSFs and CTFs (default:
            False)

    Returns
        -------
        stc: STC object
            PSFs or CTFs as stc object.
    """
    # vertices used in forward and inverse operator
    vertno_lh = src[0]['vertno']
    vertno_rh = src[1]['vertno']
    vertno = [vertno_lh, vertno_rh]

    # in everything below indices refer to columns
    if func == 'ctf':
        resmat = resmat.T

    # column of resolution matrix
    funcs = resmat[:, idx]

    if norm:
        maxval = np.abs(funcs).max()
        funcs = funcs / maxval

    # convert to source estimate
    stc = SourceEstimate(funcs, vertno, tmin=0., tstep=1.)

    return stc


def _convert_forward_match_inv(fwd, inv):
    """Helper to ensure forward and inverse operators match."""
    # did inverse operator use fixed orientation?
    is_fixed_inv = _check_inv_fixed_ori(inv)

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
    elif fwd['surf_ori']:  # free orientation, change fwd surface orientation
        fwd = convert_forward_solution(fwd, surf_ori=False)

    return fwd


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
    invmat : array, shape (n_dipoles, n_channels)
        Inverse matrix associated with inverse operator and specified
        parameters.
    """
    # make sure forward and inverse operators match
    _convert_forward_match_inv(forward, inverse_operator)

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

    # check if inverse operator uses fixed source orientations
    is_fixed_inv = _check_inv_fixed_ori(inverse_operator)

    # choose pick_ori according to inverse operator
    if is_fixed_inv:
        pick_ori = None
    else:
        pick_ori = 'vector'

    # columns for bad channels will be zero
    invmat_op = apply_inverse(ev_id, inverse_operator, lambda2=lambda2,
                              method=method, pick_ori=pick_ori)

    # turn source estimate into numpy array
    invmat = invmat_op.data

    # remove columns for bad channels
    # take into account it may be 3D array
    invmat = np.delete(invmat, ch_idx_bads, axis=invmat.ndim - 1)

    # if 3D array, i.e. multiple values per location (fixed and loose),
    # reshape into 2D array
    if invmat.ndim == 3:
        v0o1 = invmat[0, 1].copy()
        v3o2 = invmat[3, 2].copy()
        shape = invmat.shape
        invmat = invmat.reshape(shape[0] * shape[1], shape[2])
        # make sure that reshaping worked
        assert np.array_equal(v0o1, invmat[1])
        assert np.array_equal(v3o2, invmat[11])

    logger.info("Dimension of Inverse Matrix: %s" % str(invmat.shape))

    return invmat


def _check_inv_fixed_ori(inverse_operator):
    """Check if inverse operator compuated for fixed source orientations."""
    is_fixed_inv = inverse_operator['source_ori'] != FIFF.FIFFV_MNE_FREE_ORI
    return is_fixed_inv
