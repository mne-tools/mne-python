"""Coompute resolution matrix for linear estimators."""
from copy import deepcopy

import numpy as np

from mne import pick_channels
from mne.io.constants import FIFF
from mne.utils import logger

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
    # don't include bad channels
    # only use good channels from inverse operator
    bads_inv = invop['info']['bads']

    # good channels
    ch_names = [c for c in invop['info']['ch_names'] if (c not in bads_inv)]

    # get leadfield matrix from forward solution
    leadfield = _pick_leadfield(fwd['sol']['data'], fwd, ch_names)

    invmat, _ = _get_matrix_from_inverse_operator(invop, fwd, method=method,
                                                  lambda2=lambda2,
                                                  pick_ori='normal')

    resmat = invmat.dot(leadfield)

    return resmat


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


def _get_matrix_from_inverse_operator(inverse_operator, forward, labels=None,
                                      method='dSPM', lambda2=1. / 9.,
                                      pick_ori=None,
                                      mode='mean', n_svd_comp=1):
    """Get inverse matrix from an inverse operator.

    Currently works only for fixed/loose orientation constraints
    For loose orientation constraint, the CTFs are computed for the radial
    component (pick_ori='normal').
    Parameters
    ----------
    inverse_operator : instance of InverseOperator
        The inverse operator.
    forward : dict
        The forward operator.
    method : 'MNE' | 'dSPM' | 'sLORETA'
        Inverse methods (for apply_inverse).
    labels : list of Label | None
        Labels for which CTFs shall be computed. If None, inverse matrix for
        all vertices will be returned.
    lambda2 : float
        The regularization parameter (for apply_inverse).
    pick_ori : None | "normal"
        pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations (for apply_inverse).
        Determines whether whole inverse matrix G will have one or three rows
        per vertex. This will also affect summary measures for labels.
    mode : 'mean' | 'sum' | 'svd'
        CTFs can be computed for different summary measures with labels:
        'sum' or 'mean': sum or means of sub-inverse for labels
        This corresponds to situations where labels can be assumed to be
        homogeneously activated.
        'svd': SVD components of sub-inverse for labels
        This is better suited for situations where activation patterns are
        assumed to be more variable.
        "sub-inverse" is the part of the inverse matrix that belongs to
        vertices within invidual labels.
    n_svd_comp : int
        Number of SVD components for which CTFs will be computed and output
        (irrelevant for 'sum' and 'mean'). Explained variances within
        sub-inverses are shown in screen output.
    Returns
    -------
    invmat : ndarray
        Inverse matrix associated with inverse operator and specified
        parameters.
    label_singvals : list of dict
        Singular values of SVD for sub-matrices of inverse operator
        (only if mode='svd').
        Diffent list entries per label. Since SVD is applied separately for
        channel types, dictionaries may contain keys 'mag', 'grad' or 'eeg'.
    """
    mode = mode.lower()

    # apply_inverse cannot produce 3 separate orientations
    # therefore 'force_fixed=True' is required
    if not forward['surf_ori']:
        raise RuntimeError('Forward has to be surface oriented and '
                           'force_fixed=True.')
    if not (forward['source_ori'] == 1):
        raise RuntimeError('Forward has to be surface oriented and '
                           'force_fixed=True.')

    if labels:
        logger.info("About to process %d labels" % len(labels))
    else:
        logger.info("Computing whole inverse operator.")

    info_inv = _prepare_info(inverse_operator)

    # only use channels that are good for inverse operator and forward sol
    ch_names_inv = info_inv['ch_names']
    n_chs_inv = len(ch_names_inv)
    bads_inv = inverse_operator['info']['bads']

    # indices of bad channels
    ch_idx_bads = np.array([ch_names_inv.index(ch) for ch in bads_inv])

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
    if ~inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI:
        pick_ori = 'normal'
    else:
        pick_ori = None

    # columns for bad channels will be zero
    invmat_mat_op = apply_inverse(ev_id, inverse_operator, lambda2=lambda2,
                                  method=method, pick_ori=pick_ori)

    # turn source estimate into numpty array
    invmat_mat = invmat_mat_op.data

    # remove columns for bad channels (better for SVD)
    invmat_mat = np.delete(invmat_mat, ch_idx_bads, axis=1)

    logger.info("Dimension of inverse matrix: %s" % str(invmat_mat.shape))

    invmat_summary = []

    # if mode='svd', label_singvals will collect all SVD singular values for
    # labels
    label_singvals = []

    if labels:  # if labels specified, get summary of inverse matrix for labels
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
            # TO BE CHANGED: assumes that both fwd and inv have fixed
            # orientations
            # i.e. one source per vertex
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

                this_invmat_summary, s_svd_types = _label_svd(
                    invmat_lbl.T, n_svd_comp, info_inv)

                this_invmat_summary = this_invmat_summary.T
                label_singvals.append(s_svd_types)

            invmat_summary.append(this_invmat_summary)

        invmat = np.concatenate(invmat_summary, axis=0)

    else:   # no labels provided: return whole matrix
        invmat = invmat_mat

    return invmat, label_singvals


def _label_svd(a, b, c):
    """Dummy function for now."""
    return
