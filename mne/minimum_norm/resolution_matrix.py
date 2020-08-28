# -*- coding: utf-8 -*-
"""Compute resolution matrix for linear estimators."""
# Authors: olaf.hauk@mrc-cbu.cam.ac.uk
#
# License: BSD (3-clause)
from copy import deepcopy

import numpy as np

from mne import pick_channels_forward, EvokedArray, SourceEstimate
from mne.io.constants import FIFF
from mne.utils import logger, verbose, _validate_type, _pl, warn
from mne.forward.forward import convert_forward_solution
from mne.minimum_norm import apply_inverse
from mne.label import Label


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


def _get_psf_ctf(resmat, src, idx, func='psf', mode=None, n_comp=1,
                 norm=False):
    """Get point-spread (PSFs) or cross-talk (CTFs) functions.

    Parameters
    ----------
    resmat : array, shape (n_dipoles, n_dipoles)
        Forward Operator.
    src : Source Space
        Source space used to compute resolution matrix.
    idx : list of int | Label | list of Label
        Source for indices for which to compute PSFs or CTFs. If mode is not
        None, PSFs/CTFs will be returned for all indices. If mode is not None,
        an according summary measure will be computed across all PSFs/CTFs
        available from idx.
        idx can be:

        - list of integers:
            Compute PSFs/CTFs for all indices specified in idx.
        - Label:
            Compute PSFs/CTFs for all indices in this label.
        - list of Label:
            Compute PSFs/CTFs for all indices in specified labels.

    func : str ('psf' | 'ctf')
        Whether to produce PSFs or CTFs. Defaults to psf.
    mode: None | 'mean' | 'max' | 'svd'
        Compute summary of PSFs/CTFs across all indices specified in 'idx'.

        Can be:
        - None (default):
            Output individual PSFs/CTFs for each specific vertex.
        - 'mean':
            Mean of PSFs/CTFs across vertices.
        - 'max':
            PSFs/CTFs with maximum norm across vertices. Returns the n_comp
            largest PSFs/CTFs.
        - 'svd':
            SVD components across PSFs/CTFs across vertices. Returns the n_comp
            first SVD components.

    n_comp: int
        Number of PSF/CTF components to return for mode='max' or mode='svd'.
    norm : bool
        Whether to normalise to maximum across all PSFs and CTFs (default:
        False). This will be applied before computing summaries as specified in
        'mode'.

    Returns
    -------
    stc: instance of SourceEstimate
        PSFs or CTFs as an STC object.
        All functions will be returned as successive samples in one STC
        file, in the order they are specified in idx. Functions for labels
        are grouped together.
    """
    # easier later if it is list
    if type(idx) is not list:

        idx = [idx]

    # if label(s) specified get the indices, otherwise just carry on
    if type(idx[0]) is Label:

        idx = _get_source_space_vertices(stc, idx, src, allow_empty=False,
                                         use_sparse=False)

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


def get_point_spread(resmat, src, idx, mode=None, n_comp=1, norm=False):
    """Get point-spread (PSFs) functions for vertices.

    Parameters
    ----------
    resmat : array, shape (n_dipoles, n_dipoles)
        Forward Operator.
    src : instance of SourceSpaces
        Source space used to compute resolution matrix.
    idx : list of int
        Vertex indices for which PSFs or CTFs to produce.
    mode: None | 'mean' | 'max' | 'svd'
        Compute summary of PSFs/CTFs across all indices specified in 'idx'.

        Can be:
        - None (default):
            Output individual PSFs/CTFs for each specific vertex.
        - 'mean':
            Mean of PSFs/CTFs across vertices.
        - 'max':
            PSFs/CTFs with maximum norm across vertices. Returns the n_comp
            largest PSFs/CTFs.
        - 'svd':
            SVD components across PSFs/CTFs across vertices. Returns the n_comp
            first SVD components.

    n_comp: int
        Number of PSF/CTF components to return for mode='max' or mode='svd'.
    norm : bool
        Whether to normalise to maximum across all PSFs and CTFs (default:
        False). This will be applied before computing summaries as specified in
        'mode'.

    Returns
    -------
    stc: instance of SourceEstimate
        PSFs as an stc object.
    """
    return _get_psf_ctf(resmat, src, idx, func='psf', mode=mode, n_comp=n_comp,
                        norm=norm)

    


def get_cross_talk(resmat, src, idx, mode=None, n_comp=1, norm=False):
    """Get cross-talk (CTFs) function for vertices.

    Parameters
    ----------
    resmat : array, shape (n_dipoles, n_dipoles)
        Forward Operator.
    src : instance of SourceSpaces
        Source space used to compute resolution matrix.
    idx : list of int
        Vertex indices for which PSFs or CTFs to produce.
    mode: None | 'mean' | 'max' | 'svd'
        Compute summary of PSFs/CTFs across all indices specified in 'idx'.

        Can be:
        - None (default):
            Output individual PSFs/CTFs for each specific vertex.
        - 'mean':
            Mean of PSFs/CTFs across vertices.
        - 'max':
            PSFs/CTFs with maximum norm across vertices. Returns the n_comp
            largest PSFs/CTFs.
        - 'svd':
            SVD components across PSFs/CTFs across vertices. Returns the n_comp
            first SVD components.

    n_comp: int
        Number of PSF/CTF components to return for mode='max' or mode='svd'.
    norm : bool
        Whether to normalise to maximum across all PSFs and CTFs (default:
        False). This will be applied before computing summaries as specified in
        'mode'.

    Returns
    -------
    stc: instance of SourceEstimate
        CTFs as an stc object.
    """
    return _get_psf_ctf(resmat, src, idx, func='ctf', mode=mode, n_comp=n_comp,
                        norm=norm)


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


# def _get_source_space_vertices(stc, labels, src, allow_empty=False,
#                                use_sparse=False):
#     """Get indices of vertices in labels for vertices in source space."""
#     # calls _prepare_label_extraction from source_estimate.py with specific
#     # parameters and only outputs vertices
#     # if src is a mixed src space, the first 2 src spaces are surf type and
#     # the other ones are vol type. For mixed source space n_labels will be
#     # given by the number of ROIs of the cortical parcellation plus the number
#     # of vol src space
#     # stc: SourceEstimate
#     # labels: Label | list of Label
#     # src: Source Space (must match stc)
#     # allow_empty: whether to continue if a label does not overlap with source
#     # space
#     # use_sparse: whether source estimate is sparce
#     # returns: vertidx (indices to source space vertices and columns of
#     # leadfield)

#     vertidx, _ = _prepare_label_extraction(
#         stc, labels, src, mode=None, allow_empty=allow_empty, use_sparse=False)

#     return vertidx


# TO DO; removed stc, make sure it's called properly
def _get_source_space_vertices(labels, src, allow_empty=False, use_sparse=False):
    """Get indices of vertices in labels for vertices in source space."""
    # if src is a mixed src space, the first 2 src spaces are surf type and
    # the other ones are vol type. For mixed source space n_labels will be the
    # given by the number of ROIs of the cortical parcellation plus the number
    # of vol src space
    # Based on _prepare_label_extraction() from module source_estimate.py, but
    # removed "flipping" bits, and no requirement to specify stc
    # labels: Label | list of Label
    # src: Source Space
    # allow_empty: whether to continue if a label does not overlap with source
    # space
    # use_sparse: whether source estimate is sparce
    # returns: label_vertidx (indices to source space vertices and columns of
    # leadfield)

    # from .label import label_sign_flip, Label, BiHemiLabel
    from ..label import Label, BiHemiLabel

    vertno = src['vertno']
    nvert = [len(vn) for vn in vertno]

    # do the initialization
    label_vertidx = list()
    bad_labels = list()

    for li, label in enumerate(labels):
        if use_sparse:  # I don't understand sparse, just left this here
            assert isinstance(label, dict)
            vertidx = label['csr']
            # This can happen if some labels aren't present in the space
            if vertidx.shape[0] == 0:
                bad_labels.append(label['name'])
                vertidx = None
            # Efficiency shortcut: use linearity early to avoid redundant
            # calculations
            # elif mode == 'mean':
            #     vertidx = sparse.csr_matrix(vertidx.mean(axis=0))
            label_vertidx.append(vertidx)
            # label_flip.append(None)
            continue
        # standard case
        _validate_type(label, (Label, BiHemiLabel), 'labels[%d]' % (li,))

        # check if label for one or both hemispheres
        if label.hemi == 'both':
            # handle BiHemiLabel
            sub_labels = [label.lh, label.rh]
        else:
            sub_labels = [label]
        this_vertidx = list()
        for slabel in sub_labels:

            # adjust vertex indices for right hemisphere
            if slabel.hemi == 'lh':
                this_vertices = np.intersect1d(vertno[0], slabel.vertices)
                vertidx = np.searchsorted(vertno[0], this_vertices)
            elif slabel.hemi == 'rh':
                this_vertices = np.intersect1d(vertno[1], slabel.vertices)
                vertidx = nvert[0] + np.searchsorted(vertno[1], this_vertices)
            else:
                raise ValueError('label %s has invalid hemi' % label.name)
            this_vertidx.append(vertidx)

        # convert it to an array
        this_vertidx = np.concatenate(this_vertidx)
        # this_flip = None
        if len(this_vertidx) == 0:
            bad_labels.append(label.name)
            this_vertidx = None  # to later check if label is empty
        # elif mode not in ('mean', 'max'):  # mode-dependent initialization
        #     # label_sign_flip uses two properties:
        #     #
        #     # - src[ii]['nn']
        #     # - src[ii]['vertno']
        #     #
        #     # So if we override vertno with the stc vertices, it will pick
        #     # the correct normals.
        #     with _temporary_vertices(src, stc.vertices):
        #         this_flip = label_sign_flip(label, src[:2])[:, None]

        label_vertidx.append(this_vertidx)
        # label_flip.append(this_flip)

    if len(bad_labels):
        msg = ('source space does not contain any vertices for %d label%s:\n%s'
               % (len(bad_labels), _pl(bad_labels), bad_labels))
        if not allow_empty:
            raise ValueError(msg)
        else:
            # msg += '\nAssigning all-zero time series.'
            msg += '\nOutputting empty array of indices.'
            if allow_empty == 'ignore':
                logger.info(msg)
            else:
                warn(msg)

    # return label_vertidx, label_flip
    return label_vertidx
