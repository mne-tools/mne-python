# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

from copy import deepcopy

import numpy as np
from scipy import linalg

from ..io.constants import FIFF
from ..io.pick import pick_channels
from ..utils import logger, verbose, _check_option, deprecated
from ..forward import convert_forward_solution
from ..evoked import EvokedArray
from ..source_estimate import SourceEstimate
from .inverse import _subject_from_inverse
from . import apply_inverse


def _prepare_info(inverse_operator):
    """Get a usable dict."""
    # in order to convert sub-leadfield matrix to evoked data type (pretending
    # it's an epoch, see in loop below), uses 'info' from inverse solution
    # because this has all the correct projector information
    info = deepcopy(inverse_operator['info'])
    info['sfreq'] = 1000.  # necessary
    info['projs'] = inverse_operator['projs']
    return info


def _pick_leadfield(leadfield, forward, ch_names):
    """Pick out correct lead field components."""
    # NB must pick from fwd['sol']['row_names'], not ['info']['ch_names'],
    # because ['sol']['data'] may be ordered differently from functional data
    picks_fwd = pick_channels(forward['sol']['row_names'], ch_names)
    return leadfield[picks_fwd]


@deprecated('point_spread_function is deprecated and will be removed in 0.21;'
            'please get_point_spread instead.')
@verbose
def point_spread_function(inverse_operator, forward, labels, method='dSPM',
                          lambda2=1 / 9., pick_ori=None, mode='mean',
                          n_svd_comp=1, use_cps=True, verbose=None):
    """Compute point-spread functions (PSFs) for linear estimators.

    Compute point-spread functions (PSF) in labels for a combination of inverse
    operator and forward solution. PSFs are computed for test sources that are
    perpendicular to cortical surface.

    Parameters
    ----------
    inverse_operator : instance of InverseOperator
        Inverse operator.
    forward : dict
        Forward solution. Note: (Bad) channels not included in forward
        solution will not be used in PSF computation.
    labels : list of Label
        Labels for which PSFs shall be computed.
    method : 'MNE' | 'dSPM' | 'sLORETA' | 'eLORETA'
        Inverse method for which PSFs shall be computed
        (for :func:`apply_inverse`).
    lambda2 : float
        The regularization parameter (for :func:`apply_inverse`).
    pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations (for :func:`apply_inverse`).
    mode : 'mean' | 'sum' | 'svd'
        PSFs can be computed for different summary measures with labels:
        'sum' or 'mean': sum or means of sub-leadfields for labels
        This corresponds to situations where labels can be assumed to be
        homogeneously activated.
        'svd': SVD components of sub-leadfields for labels
        This is better suited for situations where activation patterns are
        assumed to be more variable.
        "sub-leadfields" are the parts of the forward solutions that belong to
        vertices within individual labels.
    n_svd_comp : int
        Number of SVD components for which PSFs will be computed and output
        (irrelevant for 'sum' and 'mean'). Explained variances within
        sub-leadfields are shown in screen output.
    use_cps : None | bool (default True)
        Whether to use cortical patch statistics to define normal
        orientations. Only used when surf_ori and/or force_fixed are True.
    %(verbose)s

    Returns
    -------
    stc_psf : SourceEstimate
        The PSFs for the specified labels
        If mode='svd': n_svd_comp components per label are created
        (i.e. n_svd_comp successive time points in mne_analyze)
        The last sample is the summed PSF across all labels
        Scaling of PSFs is arbitrary, and may differ greatly among methods
        (especially for MNE compared to noise-normalized estimates).
    evoked_fwd : Evoked
        Forward solutions corresponding to PSFs in stc_psf
        If mode='svd': n_svd_comp components per label are created
        (i.e. n_svd_comp successive time points in mne_analyze)
        The last sample is the summed forward solution across all labels
        (sum is taken across summary measures).
    """
    mode = mode.lower()
    _check_option('mode', mode, ['mean', 'sum', 'svd'])

    logger.info("About to process %d labels" % len(labels))

    forward = convert_forward_solution(forward, force_fixed=False,
                                       surf_ori=True, use_cps=use_cps)
    info = _prepare_info(inverse_operator)
    leadfield = _pick_leadfield(forward['sol']['data'][:, 2::3], forward,
                                info['ch_names'])

    # will contain means of subleadfields for all labels
    label_psf_summary = []
    # if mode='svd', this will collect all SVD singular values for labels
    label_singvals = []

    # loop over labels
    for ll in labels:
        logger.info(ll)
        if ll.hemi == 'rh':
            # for RH labels, add number of LH vertices
            offset = forward['src'][0]['vertno'].shape[0]
            # remember whether we are in the LH or RH
            this_hemi = 1
        elif ll.hemi == 'lh':
            offset = 0
            this_hemi = 0

        # get vertices on cortical surface inside label
        idx = np.intersect1d(ll.vertices, forward['src'][this_hemi]['vertno'])

        # get vertices in source space inside label
        fwd_idx = np.searchsorted(forward['src'][this_hemi]['vertno'], idx)

        # get sub-leadfield matrix for label vertices
        sub_leadfield = leadfield[:, fwd_idx + offset]

        # compute summary data for labels
        if mode == 'sum':  # sum across forward solutions in label
            logger.info("Computing sums within labels")
            this_label_psf_summary = sub_leadfield.sum(axis=1)[np.newaxis, :]
        elif mode == 'mean':
            logger.info("Computing means within labels")
            this_label_psf_summary = sub_leadfield.mean(axis=1)[np.newaxis, :]
        elif mode == 'svd':  # takes svd of forward solutions in label
            logger.info("Computing SVD within labels, using %d component(s)"
                        % n_svd_comp)

            # compute SVD of sub-leadfield
            u_svd, s_svd, _ = linalg.svd(sub_leadfield,
                                         full_matrices=False,
                                         compute_uv=True)

            # keep singular values (might be useful to some people)
            label_singvals.append(s_svd)

            # get first n_svd_comp components, weighted with their
            # corresponding singular values
            logger.info("First 5 singular values: %s" % s_svd[0:5])
            logger.info("(This tells you something about variability of "
                        "forward solutions in sub-leadfield for label)")
            # explained variance by chosen components within sub-leadfield
            my_comps = s_svd[:n_svd_comp]
            comp_var = (100. * np.sum(my_comps * my_comps) /
                        np.sum(s_svd * s_svd))
            logger.info("Your %d component(s) explain(s) %.1f%% "
                        "variance in label." % (n_svd_comp, comp_var))
            this_label_psf_summary = (u_svd[:, :n_svd_comp] *
                                      s_svd[:n_svd_comp][np.newaxis, :])
            # transpose required for conversion to "evoked"
            this_label_psf_summary = this_label_psf_summary.T

        # initialise or append to existing collection
        label_psf_summary.append(this_label_psf_summary)

    label_psf_summary = np.concatenate(label_psf_summary, axis=0)
    # compute sum across forward solutions for labels, append to end
    label_psf_summary = np.r_[label_psf_summary,
                              label_psf_summary.sum(axis=0)[np.newaxis, :]].T

    # convert sub-leadfield matrix to evoked data type (a bit of a hack)
    evoked_fwd = EvokedArray(label_psf_summary, info=info, tmin=0.)

    # compute PSFs by applying inverse operator to sub-leadfields
    logger.info("About to apply inverse operator for method='%s' and "
                "lambda2=%s" % (method, lambda2))

    stc_psf = apply_inverse(evoked_fwd, inverse_operator, lambda2,
                            method=method, pick_ori=pick_ori)

    return stc_psf, evoked_fwd


def _get_matrix_from_inverse_operator(inverse_operator, forward, labels=None,
                                      method='dSPM', lambda2=1. / 9.,
                                      mode='mean', n_svd_comp=1):
    """Get inverse matrix from an inverse operator.

    Currently works only for fixed/loose orientation constraints
    For loose orientation constraint, the CTFs are computed for the radial
    component (pick_ori='normal').

    Returns
    -------
    invmat : ndarray
        Inverse matrix associated with inverse operator and specified
        parameters.
    label_singvals : list of ndarray
        Singular values of svd for sub-inverses.
        Provides information about how well labels are represented by chosen
        components. Explained variances within sub-inverses are shown in
        screen output.
    """
    mode = mode.lower()

    if not forward['surf_ori']:
        raise RuntimeError('Forward has to be surface oriented and '
                           'force_fixed=True.')
    if not ((forward['source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI) or
            (forward['source_ori'] == FIFF.FIFFV_MNE_FIXED_CPS_ORI)):
        raise RuntimeError('Forward has to be surface oriented and '
                           'force_fixed=True.')

    if labels:
        logger.info("About to process %d labels" % len(labels))
    else:
        logger.info("Computing whole inverse operator.")

    info = _prepare_info(inverse_operator)

    # create identity matrix as input for inverse operator
    id_mat = np.eye(len(info['ch_names']))

    # convert identity matrix to evoked data type (pretending it's an epoch)
    ev_id = EvokedArray(id_mat, info=info, tmin=0.)

    # apply inverse operator to identity matrix in order to get inverse matrix
    # free orientation constraint not possible because apply_inverse would
    # combined components
    invmat_mat_op = apply_inverse(ev_id, inverse_operator, lambda2=lambda2,
                                  method=method, pick_ori='normal')

    logger.info("Dimension of inverse matrix: %s" % str(invmat_mat_op.shape))

    # turn source estimate into numpty array
    invmat_mat = invmat_mat_op.data
    invmat_summary = []
    # if mode='svd', label_singvals will collect all SVD singular values for
    # labels
    label_singvals = []

    if labels:
        for ll in labels:
            if ll.hemi == 'rh':
                # for RH labels, add number of LH vertices
                offset = forward['src'][0]['vertno'].shape[0]
                # remember whether we are in the LH or RH
                this_hemi = 1
            elif ll.hemi == 'lh':
                offset = 0
                this_hemi = 0
            else:
                raise RuntimeError("Cannot determine hemisphere of label.")

            # get vertices on cortical surface inside label
            idx = np.intersect1d(ll.vertices,
                                 forward['src'][this_hemi]['vertno'])

            # get vertices in source space inside label
            fwd_idx = np.searchsorted(forward['src'][this_hemi]['vertno'], idx)

            # get sub-inverse for label vertices, one row per vertex
            invmat_lbl = invmat_mat[fwd_idx + offset, :]

            # compute summary data for labels
            if mode == 'sum':  # takes sum across estimators in label
                logger.info("Computing sums within labels")
                this_invmat_summary = invmat_lbl.sum(axis=0)
                this_invmat_summary = np.vstack(this_invmat_summary).T
            elif mode == 'mean':
                logger.info("Computing means within labels")
                this_invmat_summary = invmat_lbl.mean(axis=0)
                this_invmat_summary = np.vstack(this_invmat_summary).T
            elif mode == 'svd':  # takes svd of sub-inverse in label
                logger.info("Computing SVD within labels, using %d "
                            "component(s)" % n_svd_comp)

                # compute SVD of sub-inverse
                u_svd, s_svd, _ = linalg.svd(invmat_lbl.T,
                                             full_matrices=False,
                                             compute_uv=True)

                # keep singular values (might be useful to some people)
                label_singvals.append(s_svd)

                # get first n_svd_comp components, weighted with their
                # corresponding singular values
                logger.info("First 5 singular values: %s" % s_svd[:5])
                logger.info("(This tells you something about variability of "
                            "estimators in sub-inverse for label)")
                # explained variance by chosen components within sub-inverse
                my_comps = s_svd[:n_svd_comp]
                comp_var = ((100 * np.sum(my_comps * my_comps)) /
                            np.sum(s_svd * s_svd))
                logger.info("Your %d component(s) explain(s) %.1f%% "
                            "variance in label." % (n_svd_comp, comp_var))
                this_invmat_summary = (u_svd[:, :n_svd_comp].T *
                                       s_svd[:n_svd_comp][:, np.newaxis])

            invmat_summary.append(this_invmat_summary)

        invmat = np.concatenate(invmat_summary, axis=0)
    else:   # no labels provided: return whole matrix
        invmat = invmat_mat

    return invmat, label_singvals


@deprecated('cross_talk_function is deprecated and will be removed in 0.21;'
            'please get_cross_talk instead.')
@verbose
def cross_talk_function(inverse_operator, forward, labels,
                        method='dSPM', lambda2=1 / 9., signed=False,
                        mode='mean', n_svd_comp=1, use_cps=True, verbose=None):
    """Compute cross-talk functions (CTFs) for linear estimators.

    Compute cross-talk functions (CTF) in labels for a combination of inverse
    operator and forward solution. CTFs are computed for test sources that are
    perpendicular to cortical surface.

    Parameters
    ----------
    inverse_operator : instance of InverseOperator
        Inverse operator.
    forward : dict
        Forward solution. Note: (Bad) channels not included in forward
        solution will not be used in CTF computation.
    labels : list of Label
        Labels for which CTFs shall be computed.
    method : 'MNE' | 'dSPM' | 'sLORETA' | 'eLORETA'
        Inverse method for which CTFs shall be computed.
    lambda2 : float
        The regularization parameter.
    signed : bool
        If True, CTFs will be written as signed source estimates. If False,
        absolute (unsigned) values will be written
    mode : 'mean' | 'sum' | 'svd'
        CTFs can be computed for different summary measures with labels:
        'sum' or 'mean': sum or means of sub-inverses for labels
        This corresponds to situations where labels can be assumed to be
        homogeneously activated.
        'svd': SVD components of sub-inverses for labels
        This is better suited for situations where activation patterns are
        assumed to be more variable. "sub-inverse" is the part of the inverse
        matrix that belongs to vertices within individual labels.
    n_svd_comp : int
        Number of SVD components for which CTFs will be computed and output
        (irrelevant for 'sum' and 'mean'). Explained variances within
        sub-inverses are shown in screen output.
    use_cps : None | bool (default True)
        Whether to use cortical patch statistics to define normal
        orientations. Only used when surf_ori and/or force_fixed are True.
    %(verbose)s

    Returns
    -------
    stc_ctf : SourceEstimate
        The CTFs for the specified labels.
        If mode='svd': n_svd_comp components per label are created
        (i.e. n_svd_comp successive time points in mne_analyze)
        The last sample is the summed CTF across all labels.
    """
    forward = convert_forward_solution(forward, force_fixed=True,
                                       surf_ori=True, use_cps=use_cps)

    # get the inverse matrix corresponding to inverse operator
    out = _get_matrix_from_inverse_operator(inverse_operator, forward,
                                            labels=labels, method=method,
                                            lambda2=lambda2, mode=mode,
                                            n_svd_comp=n_svd_comp)
    invmat, label_singvals = out

    # get the leadfield matrix from forward solution
    leadfield = _pick_leadfield(forward['sol']['data'], forward,
                                inverse_operator['info']['ch_names'])

    # compute cross-talk functions (CTFs)
    ctfs = np.dot(invmat, leadfield)

    # compute sum across forward solutions for labels, append to end
    ctfs = np.vstack((ctfs, ctfs.sum(axis=0)))

    # if unsigned output requested, take absolute values
    if not signed:
        ctfs = np.abs(ctfs, out=ctfs)

    # create source estimate object
    vertno = [ss['vertno'] for ss in inverse_operator['src']]
    stc_ctf = SourceEstimate(ctfs.T, vertno, tmin=0., tstep=1.)

    stc_ctf.subject = _subject_from_inverse(inverse_operator)

    return stc_ctf
