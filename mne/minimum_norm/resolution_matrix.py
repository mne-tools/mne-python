# -*- coding: utf-8 -*-
"""Compute resolution matrix for linear estimators."""
# Authors: olaf.hauk@mrc-cbu.cam.ac.uk
#
# License: BSD (3-clause)
from copy import deepcopy

import numpy as np

from mne import pick_channels_forward, EvokedArray, SourceEstimate
from mne.io.constants import FIFF
from mne.utils import logger, verbose
from mne.forward.forward import convert_forward_solution
from mne.minimum_norm import apply_inverse


@verbose
def make_inverse_resolution_matrix(forward, inverse_operator, method='dSPM',
                                   lambda2=1. / 9., verbose=None):
    """Compute resolution matrix for linear inverse operator.

    Parameters
    ----------
    forward : instance of Forward
        Forward Operator.
    inverse_operator : instance of InverseOperator
        Inverse operator.
    method : 'MNE' | 'dSPM' | 'sLORETA'
        Inverse method to use (MNE, dSPM, sLORETA).
    lambda2 : float
        The regularisation parameter.
    %(verbose)s

    Returns
    -------
    resmat: array, shape (n_orient_inv * n_dipoles, n_orient_fwd * n_dipoles)
        Resolution matrix (inverse operator times forward operator).
        The result of applying the inverse operator to the forward operator.
        If source orientations are not fixed, all source components will be
        computed (i.e. for n_orient_inv > 1 or n_orient_fwd > 1).
        The columns of the resolution matrix are the point-spread functions
        (PSFs) and the rows are the cross-talk functions (CTFs).
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


def _get_psf_ctf(resmat, src, idx, func='psf', norm=False):
    """Get point-spread (PSFs) or cross-talk (CTFs) functions for vertices.

    Parameters
    ----------
    resmat : array, shape (n_dipoles, n_dipoles)
        Forward Operator.
    src : Source Space
        Source space used to compute resolution matrix.
    idx : list of int
        Vertex indices for which PSFs or CTFs to produce.
    func : str ('psf' | 'ctf')
        Whether to produce PSFs or CTFs. Defaults to psf.
    norm : bool
        Whether to normalise to maximum across all PSFs and CTFs (default:
        False).

    Returns
    -------
    stc: instance of SourceEstimate
        PSFs or CTFs as an stc object.
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


def get_point_spread(resmat, src, idx, norm=False):
    """Get point-spread (PSFs) functions for vertices.

    Parameters
    ----------
    resmat : array, shape (n_dipoles, n_dipoles)
        Forward Operator.
    src : instance of SourceSpaces
        Source space used to compute resolution matrix.
    idx : list of int
        Vertex indices for which PSFs or CTFs to produce.
    norm : bool
        Whether to normalise to maximum across all PSFs (default: False).

    Returns
    -------
    stc: instance of SourceEstimate
        PSFs as an stc object.
    """
    return _get_psf_ctf(resmat, src, idx, func='psf', norm=norm)


def get_cross_talk(resmat, src, idx, norm=False):
    """Get cross-talk (CTFs) function for vertices.

    Parameters
    ----------
    resmat : array, shape (n_dipoles, n_dipoles)
        Forward Operator.
    src : instance of SourceSpaces
        Source space used to compute resolution matrix.
    idx : list of int
        Vertex indices for which PSFs or CTFs to produce.
    norm : bool
        Whether to normalise to maximum across all CTFs (default: False).

    Returns
    -------
    stc: instance of SourceEstimate
        CTFs as an stc object.
    """
    return _get_psf_ctf(resmat, src, idx, func='ctf', norm=norm)


def _convert_forward_match_inv(fwd, inv):
    """Ensure forward and inverse operators match.

    Inverse operator and forward operator must have same surface orientations,
    but can have different source orientation constraints.
    """
    # did inverse operator use fixed orientation?
    is_fixed_inv = _check_fixed_ori(inv)

    # did forward operator use fixed orientation?
    is_fixed_fwd = _check_fixed_ori(fwd)

    # if inv or fwd fixed: do nothing
    # if inv loose: surf_ori must be True
    # if inv free: surf_ori must be False
    if not is_fixed_inv and not is_fixed_fwd:
        is_loose_inv = not (inv['orient_prior']['data'] == 1.).all()

        if is_loose_inv:
            if not fwd['surf_ori']:
                fwd = convert_forward_solution(fwd, surf_ori=True)
        elif fwd['surf_ori']:  # free orientation, change fwd
            fwd = convert_forward_solution(fwd, surf_ori=False)

    return fwd


def _prepare_info(inverse_operator):
    """Get a usable dict."""
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
    forward : instance of Forward
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
    # make sure forward and inverse operators match with respect to
    # surface orientation
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
    is_fixed_inv = _check_fixed_ori(inverse_operator)

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


def _check_fixed_ori(inst):
    """Check if inverse or forward was computed for fixed orientations."""
    is_fixed = inst['source_ori'] != FIFF.FIFFV_MNE_FREE_ORI
    return is_fixed
