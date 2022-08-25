# -*- coding: utf-8 -*-
"""Compute resolution matrix for linear estimators."""
# Authors: olaf.hauk@mrc-cbu.cam.ac.uk
#
# License: BSD-3-Clause
from copy import deepcopy

import numpy as np

from .. import pick_types, pick_channels_forward, EvokedArray, SourceEstimate
from ..io.constants import FIFF
from ..utils import logger, verbose
from ..forward.forward import convert_forward_solution
from ..minimum_norm import (apply_inverse, apply_inverse_cov,
                            prepare_inverse_operator)
from ..minimum_norm.spatial_resolution import _rectify_resolution_matrix
from ..source_estimate import _prepare_label_extraction
from ..label import Label


@verbose
def make_inverse_resolution_matrix(forward, inverse_operator, method='dSPM',
                                   lambda2=1. / 9., noise_cov=None, snr=None,
                                   verbose=None):
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
    noise_cov : None | instance of Covariance
        Noise covariance matrix to compute noise power in source space.
    snr : None | float
        Signal-to-noise ratio for source signal vs noise power.
    %(verbose)s

    Returns
    -------
    resmat: array, shape (n_orient_inv * n_dipoles, n_orient_fwd * n_dipoles)
        Resolution matrix (inverse operator times forward operator). The
        columns of the resolution matrix are the point-spread functions (PSFs)
        and the rows are the cross-talk functions (CTFs).
        The result of applying the inverse operator to the forward operator.
        If source orientations are not fixed, all source components will be
        computed (i.e. for n_orient_inv > 1 or n_orient_fwd > 1).
        If 'noise_cov' and 'snr' are not None, then source noise estimated from
        noise covariance matrix will be added to columns of the resolution
        matrix (PSFs). In this case, 'resmat' will be of shape
        (n_dipoles, n_orient_fwd * n_dipoles) with values pooled across
        n_orient_inv source orientations per location.
        Note: If the resolution matrix is computed with a noise covariance
        matrix then only is columns, i.e. PSFs, can meaningfully be
        interpreted. It must not be used to compute CTFs or resolution metrics
        for CTFs!
    """
    if ((noise_cov is None and snr is not None) or
            (noise_cov is not None and snr is None)):
        msg = 'noise_cov and snr must both be None or not be None.'
        raise ValueError(msg)

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

    # add source noise power to columns of resolution matrix
    if noise_cov is not None:
        inv = prepare_inverse_operator(inv, 1, lambda2, method, copy='non-src')
        info = _prepare_info(inv)
        # compute source noise poweralph
        stc = apply_inverse_cov(
            noise_cov, info, inv, nave=1, lambda2=lambda2,
            method=method, pick_ori=None, prepared=True, label=None,
            method_params=None, use_cps=True, verbose=None)

        # output will be square-root of power, so take intensities before
        # adding noise
        shape = resmat.shape
        if not shape[0] == shape[1]:
            # pool loose/free orientations
            resmat = _rectify_resolution_matrix(resmat)
        else:
            resmat = np.abs(resmat)

        # scaling factor

        idx_ch_types = {}
        idx_ch_types['eeg'] = pick_types(info, eeg=True, meg=False)
        idx_ch_types['mag'] = pick_types(info, eeg=False, meg='mag')
        idx_ch_types['gra'] = pick_types(info, eeg=False, meg='grad')

        # scaling similar to Samuelsson et al., Neuroimage 2021 (p. 4)
        alphas = {}
        for cht in idx_ch_types:
            # signal intensity of all sources across channels of this type
            alphas[cht] = (
                np.abs(leadfield[idx_ch_types[cht], :]).mean())
            # divided by noise intensity across channels of this type
            alphas[cht] /= (
                np.sqrt(np.diag(noise_cov.data)[idx_ch_types[cht]].mean()))

        alpha = (alphas['eeg'] + alphas['mag'] + alphas['gra']) / (3 * snr)

        # # previous attempt to compute scaling via whitened gain matrix
        # whitener = inv['whitener']
        # gain = np.dot(whitener, leadfield)
        # gram_gain = np.dot(gain, gain.T)
        # alpha = np.sqrt(np.trace(gram_gain))

        # Add square root of source noise power to every column, scale
        # depending on SNR
        resmat += alpha * np.sqrt(stc.data)

    return resmat


@verbose
def _get_psf_ctf(resmat, src, idx, func, mode, n_comp, norm, return_pca_vars,
                 verbose=None):
    """Get point-spread (PSFs) or cross-talk (CTFs) functions.

    Parameters
    ----------
    resmat : array, shape (n_dipoles, n_dipoles)
        Forward Operator.
    src : Source Space
        Source space used to compute resolution matrix.
    %(idx_pctf)s
    func : str ('psf' | 'ctf')
        Whether to produce PSFs or CTFs. Defaults to psf.
    %(mode_pctf)s
    %(n_comp_pctf_n)s
    %(norm_pctf)s
    %(return_pca_vars_pctf)s
    %(verbose)s

    Returns
    -------
    %(stcs_pctf)s
    %(pca_vars_pctf)s
    """
    # check for consistencies in input parameters
    _check_get_psf_ctf_params(mode, n_comp, return_pca_vars)

    # backward compatibility
    if norm is True:
        norm = 'max'

    # get relevant vertices in source space
    verts_all = _vertices_for_get_psf_ctf(idx, src)
    # vertices used in forward and inverse operator
    vertno_lh = src[0]['vertno']
    vertno_rh = src[1]['vertno']
    vertno = [vertno_lh, vertno_rh]

    # the following will operate on columns of funcs
    if func == 'ctf':
        resmat = resmat.T
    # Functions and variances per label
    stcs = []
    pca_vars = []

    for verts in verts_all:
        # get relevant PSFs or CTFs for specified vertices
        funcs = resmat[:, verts]

        # normalise PSFs/CTFs if requested
        if norm is not None:
            funcs = _normalise_psf_ctf(funcs, norm)

        # summarise PSFs/CTFs across vertices if requested
        pca_var = None  # variances computed only if return_pca_vars=True
        if mode is not None:
            funcs, pca_var = _summarise_psf_ctf(funcs, mode, n_comp,
                                                return_pca_vars)

        # convert to source estimate
        stc = SourceEstimate(funcs, vertno, tmin=0., tstep=1.)
        stcs.append(stc)
        pca_vars.append(pca_var)

    # if just one list or label specified, simplify output
    if len(stcs) == 1:
        stcs = stc
    if len(pca_vars) == 1:
        pca_vars = pca_var
    if pca_var is not None:
        return stcs, pca_vars
    else:
        return stcs


def _check_get_psf_ctf_params(mode, n_comp, return_pca_vars):
    """Check input parameters of _get_psf_ctf() for consistency."""
    if mode in [None, 'sum', 'mean'] and n_comp > 1:
        msg = 'n_comp must be 1 for mode=%s.' % mode
        raise ValueError(msg)
    if mode != 'pca' and return_pca_vars:
        msg = 'SVD variances can only be returned if mode=''pca''.'
        raise ValueError(msg)


def _vertices_for_get_psf_ctf(idx, src):
    """Get vertices in source space for PSFs/CTFs in _get_psf_ctf()."""
    # idx must be list
    # if label(s) specified get the indices, otherwise just carry on
    if type(idx[0]) is Label:
        # specify without source time courses, gets indices per label
        verts_labs, _ = _prepare_label_extraction(
            stc=None, labels=idx, src=src, mode='mean', allow_empty=False,
            use_sparse=False)
        # verts_labs can be list of lists
        # concatenate indices per label across hemispheres
        # one list item per label
        verts = []

        for v in verts_labs:
            # if two hemispheres present
            if type(v) is list:
                # indices for both hemispheres in one list
                this_verts = np.concatenate((v[0], v[1]))
            else:
                this_verts = np.array(v)
            verts.append(this_verts)
    # check if list of list or just list
    else:
        if type(idx[0]) is list:  # if list of list of integers
            verts = idx
        else:  # if list of integers
            verts = [idx]

    return verts


def _normalise_psf_ctf(funcs, norm):
    """Normalise PSFs/CTFs in _get_psf_ctf()."""
    # normalise PSFs/CTFs if specified
    if norm == 'max':
        maxval = max(-funcs.min(), funcs.max())
        funcs = funcs / maxval
    elif norm == 'norm':  # normalise to maximum norm across columns
        norms = np.linalg.norm(funcs, axis=0)
        funcs = funcs / norms.max()

    return funcs


def _summarise_psf_ctf(funcs, mode, n_comp, return_pca_vars):
    """Summarise PSFs/CTFs across vertices."""
    from scipy import linalg
    s_var = None  # only computed for return_pca_vars=True

    if mode == 'maxval':  # pick PSF/CTF with maximum absolute value
        absvals = np.maximum(-np.min(funcs, axis=0), np.max(funcs, axis=0))
        if n_comp > 1:  # only keep requested number of sorted PSFs/CTFs
            sortidx = np.argsort(absvals)
            maxidx = sortidx[-n_comp:]
        else:  # faster if only one required
            maxidx = absvals.argmax()
        funcs = funcs[:, maxidx]

    elif mode == 'maxnorm':  # pick PSF/CTF with maximum norm
        norms = np.linalg.norm(funcs, axis=0)
        if n_comp > 1:  # only keep requested number of sorted PSFs/CTFs
            sortidx = np.argsort(norms)
            maxidx = sortidx[-n_comp:]
        else:  # faster if only one required
            maxidx = norms.argmax()
        funcs = funcs[:, maxidx]

    elif mode == 'sum':  # sum across PSFs/CTFs
        funcs = np.sum(funcs, axis=1)

    elif mode == 'mean':  # mean of PSFs/CTFs
        funcs = np.mean(funcs, axis=1)

    elif mode == 'pca':  # SVD across PSFs/CTFs
        # compute SVD of PSFs/CTFs across vertices
        u, s, _ = linalg.svd(funcs, full_matrices=False)
        funcs = u[:, :n_comp]
        # if explained variances for SVD components requested
        if return_pca_vars:
            # explained variance of individual SVD components
            s2 = s * s
            s_var = 100 * s2[:n_comp] / s2.sum()

    return funcs, s_var


@verbose
def get_point_spread(resmat, src, idx, mode=None, n_comp=1, norm=False,
                     return_pca_vars=False, verbose=None):
    """Get point-spread (PSFs) functions for vertices.

    Parameters
    ----------
    resmat : array, shape (n_dipoles, n_dipoles)
        Forward Operator.
    src : instance of SourceSpaces
        Source space used to compute resolution matrix.
    %(idx_pctf)s
    %(mode_pctf)s
    %(n_comp_pctf_n)s
    %(norm_pctf)s
    %(return_pca_vars_pctf)s
    %(verbose)s

    Returns
    -------
    %(stcs_pctf)s
    %(pca_vars_pctf)s
    """
    return _get_psf_ctf(resmat, src, idx, func='psf', mode=mode, n_comp=n_comp,
                        norm=norm, return_pca_vars=return_pca_vars)


@verbose
def get_cross_talk(resmat, src, idx, mode=None, n_comp=1, norm=False,
                   return_pca_vars=False, verbose=None):
    """Get cross-talk (CTFs) function for vertices.

    Parameters
    ----------
    resmat : array, shape (n_dipoles, n_dipoles)
        Forward Operator.
    src : instance of SourceSpaces
        Source space used to compute resolution matrix.
    %(idx_pctf)s
    %(mode_pctf)s
    %(n_comp_pctf_n)s
    %(norm_pctf)s
    %(return_pca_vars_pctf)s
    %(verbose)s

    Returns
    -------
    %(stcs_pctf)s
    %(pca_vars_pctf)s
    """
    return _get_psf_ctf(resmat, src, idx, func='ctf', mode=mode, n_comp=n_comp,
                        norm=norm, return_pca_vars=return_pca_vars)


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
    with info._unlock():
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
    method : 'MNE' | 'dSPM' | 'sLORETA' | 'eLORETA'
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
