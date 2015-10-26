# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

from copy import deepcopy

import numpy as np
from scipy import linalg

from .. import pick_channels_forward
from ..io.pick import pick_channels
from ..utils import logger, verbose
from ..forward import convert_forward_solution
from ..evoked import EvokedArray
from ..source_estimate import SourceEstimate
from .inverse import _subject_from_inverse
from . import apply_inverse


def _prepare_info(inverse_operator):
    """Helper to get a usable dict"""
    # in order to convert sub-leadfield matrix to evoked data type (pretending
    # it's an epoch, see in loop below), uses 'info' from inverse solution
    # because this has all the correct projector information
    info = deepcopy(inverse_operator['info'])
    info['sfreq'] = 1000.  # necessary
    info['projs'] = inverse_operator['projs']
    return info


def _pick_leadfield(leadfield, forward, ch_names):
    """Helper to pick out correct lead field components"""
    picks_fwd = pick_channels(forward['info']['ch_names'], ch_names)
    return leadfield[picks_fwd]


@verbose
def point_spread_function(inverse_operator, forward, labels, method='dSPM',
                          lambda2=1 / 9., pick_ori=None, mode='mean',
                          n_svd_comp=1, verbose=None):
    """Compute point-spread functions (PSFs) for linear estimators

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
    method : 'MNE' | 'dSPM' | 'sLORETA'
        Inverse method for which PSFs shall be computed (for apply_inverse).
    lambda2 : float
        The regularization parameter (for apply_inverse).
    pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations (for apply_inverse).
    mode : 'mean' | 'sum' | 'svd' |
        PSFs can be computed for different summary measures with labels:
        'sum' or 'mean': sum or means of sub-leadfields for labels
        This corresponds to situations where labels can be assumed to be
        homogeneously activated.
        'svd': SVD components of sub-leadfields for labels
        This is better suited for situations where activation patterns are
        assumed to be more variable.
        "sub-leadfields" are the parts of the forward solutions that belong to
        vertices within invidual labels.
    n_svd_comp : integer
        Number of SVD components for which PSFs will be computed and output
        (irrelevant for 'sum' and 'mean'). Explained variances within
        sub-leadfields are shown in screen output.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

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
    if mode not in ['mean', 'sum', 'svd']:
        raise ValueError("mode must be 'svd', 'mean' or 'sum'. Got %s."
                         % mode)

    logger.info("About to process %d labels" % len(labels))

    forward = convert_forward_solution(forward, force_fixed=False,
                                       surf_ori=True)

    info_inv = _prepare_info(inverse_operator)

    fwd_ch_names = forward['info']['ch_names']
    inv_ch_names = info_inv['ch_names']
    ch_names = [c for c in inv_ch_names if (c not in info_inv['bads']) ]

    # select those bad channels that are actually used in current operator
    # (otherwise could cause trouble later)
    ch_bads = [c for c in info_inv['bads'] if (c in inv_ch_names)]
    forward['info']['bads'] = ch_bads
    inverse_operator['info']['bads'] = ch_bads

    # reduce forward to channels in invop
    forward = pick_channels_forward(forward, ch_names)

    leadfield = _pick_leadfield(forward['sol']['data'][:, 2::3], forward,
                                                                   ch_names)

    # average-referencing leadfield for EEG before SVD    
    EEG_idx = [cc for cc in range(len(ch_names)) if ch_names[cc][:3]=='EEG']
    nr_eeg = len(EEG_idx)
    if nr_eeg: # if EEG present
        lfdmean = leadfield[EEG_idx,:].mean(axis=0)
        leadfield[EEG_idx,:] = leadfield[EEG_idx,:] - lfdmean[np.newaxis,:]

    label_psf_summary = _deflect_make_subleadfields(labels, forward, leadfield,
                            mode=mode, n_svd_comp=[n_svd_comp], verbose=None)

    # compute sum across forward solutions for labels, append to end
    label_psf_summary = np.c_[label_psf_summary,
                              label_psf_summary.sum(axis=1)]

    # convert sub-leadfield matrix to evoked data type (a bit of a hack)
    info_fwd = forward['info']
    info_fwd['sfreq'] = info_inv['sfreq']
    info_fwd['projs'] = info_inv['projs']
    evoked_fwd = EvokedArray(label_psf_summary, info=forward['info'], tmin=0.)

    # compute PSFs by applying inverse operator to sub-leadfields
    logger.info("About to apply inverse operator for method='%s' and "
                "lambda2=%s" % (method, lambda2))

    stc_psf = apply_inverse(evoked_fwd, inverse_operator, lambda2,
                            method=method, pick_ori=pick_ori)

    return stc_psf, evoked_fwd


def _get_matrix_from_inverse_operator(inverse_operator, forward, labels=None,
                                      method='dSPM', lambda2=1. / 9.,
                                      mode='mean', n_svd_comp=1):
    """Get inverse matrix from an inverse operator

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
    if not (forward['source_ori'] == 1):
        raise RuntimeError('Forward has to be surface oriented and '
                           'force_fixed=True.')

    if labels:
        logger.info("About to process %d labels" % len(labels))
    else:
        logger.info("Computing whole inverse operator.")
    
    info_inv = _prepare_info(inverse_operator)
    inv_ch_names = info_inv['ch_names']
    ch_names = [c for c in inv_ch_names if (c not in info_inv['bads']) ]
    bad_idx = [inv_ch_names.index(cc) for cc in info_inv['bads']]

    # create identity matrix as input for inverse operator
    id_mat = np.eye(len(inv_ch_names))

    # convert identity matrix to evoked data type (pretending it's an epoch)
    ev_id = EvokedArray(id_mat, info=info_inv, tmin=0.)

    snr = 3.0
    lambda2 = 1.0 / snr ** 2

    # apply inverse operator to identity matrix in order to get inverse matrix
    # free orientation constraint not possible because apply_inverse would
    # combine components
    invmat_mat_op = apply_inverse(ev_id, inverse_operator, lambda2=lambda2,
                                  method=method, pick_ori='normal')

    logger.info("Dimension of inverse matrix: %s" % str(invmat_mat_op.shape))

    # turn source estimate into numpty array
    invmat_mat = invmat_mat_op.data
    # remove columns for bad channels (better for SVD)
    invmat_mat = np.delete(invmat_mat, bad_idx, 1)

    invmat_summary = []
    # if mode='svd', label_singvals will collect all SVD singular values for
    # labels
    label_singvals = []

    if labels: # if labels specified, get summary of inverse matrix for labels
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

                this_invmat_summary = _label_svd(invmat_lbl.T, n_svd_comp,
                                                                      ch_names)
                this_invmat_summary = this_invmat_summary.T

            invmat_summary.append(this_invmat_summary)

        invmat = np.concatenate(invmat_summary, axis=0)

        
    else:   # no labels provided: return whole matrix
        invmat = invmat_mat

    return invmat


@verbose
def cross_talk_function(inverse_operator, forward, labels,
                        method='dSPM', lambda2=1 / 9., signed=False,
                        mode='mean', n_svd_comp=1, verbose=None):
    """Compute cross-talk functions (CTFs) for linear estimators

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
    method : 'MNE' | 'dSPM' | 'sLORETA'
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
        matrix that belongs to vertices within invidual labels.
    n_svd_comp : int
        Number of SVD components for which CTFs will be computed and output
        (irrelevant for 'sum' and 'mean'). Explained variances within
        sub-inverses are shown in screen output.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc_ctf : SourceEstimate
        The CTFs for the specified labels.
        If mode='svd': n_svd_comp components per label are created
        (i.e. n_svd_comp successive time points in mne_analyze)
        The last sample is the summed CTF across all labels.
    """

    forward = convert_forward_solution(forward, force_fixed=True,
                                       surf_ori=True)    

    info_inv = _prepare_info(inverse_operator)
    inv_ch_names = info_inv['ch_names']
    ch_names = [c for c in inv_ch_names if (c not in info_inv['bads'])]

    # reduce forward to channels in invop
    forward = pick_channels_forward(forward, ch_names)

    # select those bad channels that are actually used in current operator
    # (otherwise could cause trouble later)
    ch_bads = [c for c in info_inv['bads'] if (c in inv_ch_names)]
    forward['info']['bads'] = ch_bads
    inverse_operator['info']['bads'] = ch_bads
    
    # get the inverse matrix corresponding to inverse operator
    # get summary components for labels if specified
    invmat = _get_matrix_from_inverse_operator(inverse_operator, forward,
                                            labels=labels, method=method,
                                            lambda2=lambda2, mode=mode,
                                            n_svd_comp=n_svd_comp)

    # get the leadfield matrix from forward solution
    leadfield = _pick_leadfield(forward['sol']['data'], forward, ch_names)

    # average-referencing leadfield for EEG before SVD    
    EEG_idx = [cc for cc in range(len(ch_names)) if ch_names[cc][:3]=='EEG']
    nr_eeg = len(EEG_idx)
    if nr_eeg: # if EEG present
        lfdmean = leadfield[EEG_idx,:].mean(axis=0)
        leadfield[EEG_idx,:] = leadfield[EEG_idx,:] - lfdmean[np.newaxis,:]

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
# Done

# the next three functions are from DeFleCT module,
# which is not available yet)

def _deflect_make_subleadfields(labels, forward, leadfield, mode='svd',
                                                   n_svd_comp=[1], verbose=None):
    """ Compute summaries of subleadfields for specific labels for creation of
        DeFleCT estimator

    Parameters:
    -----------
    labels : list of labels
        first label is the target for DeFleCT, remaining ones to be suppressed
    forward : forward solution object
    leadfield : 2D numpy array (n_chan x n_vert)
        Leadfield matrix from forward solution
    ch_names: list of strings
        channel names to be processed
    mode : 'mean' | 'sum' | 'svd'
        PSFs can be computed for different summary measures with labels:
        'sum' or 'mean': sum or means of sub-leadfields for labels
        This corresponds to situations where labels can be assumed to be
        homogeneously activated.
        'svd': SVD components of sub-leadfields for labels
        This is better suited for situations where activation patterns are
        assumed to be more variable.
        "sub-leadfields" are the parts of the forward solutions that belong to
        vertices within invidual labels.
    n_svd_comp : list of integers
        Number of SVD components for which PSFs will be computed and output
        (irrelevant for 'sum' and 'mean'). Explained variances within
        sub-leadfields are shown in screen output.
        Either list with one value per label, or one value for all (in which
        case only one component will be extracted for the first label as
        target component)
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns:
    --------
    label_lfd_summary : numpy array (n_chan x ((n_labels-1)*n_svd_comp)+1)
        First columns: target component for DeFleCT for first label
        Following columns: Components for remaining labels, depending on 'mode'
        if mode != 'svd': one components per label (i.e. n_svd_comp=1 above)
    """

    # check if number of labels and number of SVD components are compatible
    len_labels = len(labels)
    len_svd = len(n_svd_comp)
    if len_svd > 1:    # if numbers supposed to exist for every label
        if len_svd != len_labels:
            raise ValueError("Number of labels and SVD components do not "
                                "match: %d vs %d" % (len_labels, len_svd))
    else:    # if only one number to be applied to all labels, except first
        if len_labels > 1: # if only one label: keep n_svd_comp as is
            # as specified for non-target labels
            n_svd_comp = np.tile(n_svd_comp[0], len_labels).tolist()
            n_svd_comp[0] = 1     # one component for first label

    label_lfd_summary = np.array([])  # sub-leadfield summaries
    fwd_idx_all = []  #  source space indices in labels

    for [lli,ll] in enumerate(labels):
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
        fwd_idx_all.append(fwd_idx)

        # get sub-leadfield matrix for label vertices
        sub_leadfield = leadfield[:, fwd_idx + offset]

        # compute summary data for labels
        if mode == 'sum':  # sum across forward solutions in label
            logger.info("Computing sums within labels")
            this_label_lfd_summary = np.array(sub_leadfield.sum(axis=1))
            # for later concatenation:
            this_label_lfd_summary = this_label_lfd_summary[:,np.newaxis]
        elif mode == 'mean':
            logger.info("Computing means within labels")
            this_label_lfd_summary = np.array(sub_leadfield.mean(axis=1))
            this_label_lfd_summary = this_label_lfd_summary[:,np.newaxis]
        elif mode == 'svd':  # takes svd of forward solutions in label
            info_fwd = forward['info']
            fwd_ch_names = info_fwd['ch_names']
            ch_names = [c for c in fwd_ch_names if (c not in info_fwd['bads'])]
            this_label_lfd_summary = _label_svd(sub_leadfield, n_svd_comp[lli],
                                                                      ch_names)

        # initialise or append to existing list
        if lli == 0:
            label_lfd_summary = this_label_lfd_summary
        else:
            label_lfd_summary = np.append(label_lfd_summary,
                                           this_label_lfd_summary, 1)

    return label_lfd_summary

# Done


def _get_svd_comps(sub_leadfield, n_svd_comp):
    """ Compute SVD components of sub-leadfield for selected channels
    (all channels in one SVD)
    Parameters:
    -----------
    sub_leadfield : 2D numpy array (n_sens x n_vert)
        sub-leadfield (for n_vert vertices in labels) for which SVD is required
    n_svd_comp : scalar
        number of SVD components required per sub-leadfield

    Returns:
    --------
    u_svd : 2D numpy array (n_chan x n_svd_comp )
        scaled SVD components of subleadfield for
           selected channels
    s_svd : numpy vector (n_chan)
        corresponding singular values
    """

    u_svd, s_svd, _ = np.linalg.svd(sub_leadfield,
                                 full_matrices=False,
                                 compute_uv=True)

    # get desired first vectors of u_svd
    u_svd = u_svd[:, :n_svd_comp]
   
    # project SVD components on sub-leadfield, take sum over vertices
    u_svd_proj = u_svd.T.dot(sub_leadfield).sum(axis=1)
    # make sure overall projection has positive sign
    u_svd = u_svd.dot(np.sign(np.diag(u_svd_proj)))

    u_svd = u_svd * s_svd[:n_svd_comp][np.newaxis, :]

    logger.info("First 5 singular values (n=%d): %s" % (u_svd.shape[0],
                                                              s_svd[0:5]))
    
    # explained variance by chosen components within sub-leadfield
    my_comps = s_svd[0:n_svd_comp]

    comp_var = (100. * np.sum(my_comps * my_comps) / np.sum(s_svd * s_svd))
    logger.info("Your %d component(s) explain(s) %.1f%% "
                "variance." % (n_svd_comp, comp_var))

    return u_svd
# Done


def _label_svd(sub_leadfield, n_svd_comp, ch_names):
    """ Computes SVD of subleadfield for sensor types separately

    Parameters:
    -----------
    sub_leadfield : 2D numpy array (n_sens x n_vert)
        sub-leadfield (for n_vert vertices in labels) for which SVD is required
    n_svd_comp : scalar
        number of SVD components required for sub-leadfield
    ch_names : list of strings
        list of channel names

    Returns:
    --------
    this_label_lfd_summary : numpy array (n_chan x n_comp)
        n_svd_comp scaled SVD components of sub-leadfield
    """

    logger.info("\nComputing SVD within labels, using %d component(s)"
                        % n_svd_comp)
    

    EEG_idx = [cc for cc in range(len(ch_names)) if ch_names[cc][:3]=='EEG']
    MAG_idx = [cc for cc in range(len(ch_names)) if (ch_names[cc][:3]=='MEG'
                                                and ch_names[cc][-1:]=='1')]
    GRA_idx = [cc for cc in range(len(ch_names)) if (ch_names[cc][:3]=='MEG'
                    and (ch_names[cc][-1:]=='2' or ch_names[cc][-1:]=='3'))]

    list_idx = []
    u_idx = -1 # keep track which element of u_svd belongs to which sensor type
    if MAG_idx:
        list_idx.append(MAG_idx)
        u_idx += 1
        u_mag = u_idx
    if GRA_idx:
        list_idx.append(GRA_idx)
        u_idx += 1
        u_gra = u_idx
    if EEG_idx:
        list_idx.append(EEG_idx)
        u_idx += 1
        u_eeg = u_idx
    
    # # compute SVD of sub-leadfield for individual sensor types
    u_svd = [_get_svd_comps(sub_leadfield[ch_idx,:], n_svd_comp) for ch_idx
                                                                  in list_idx]

    # put sensor types back together
    this_label_lfd_summary = np.zeros([len(ch_names),u_svd[0].shape[1]])
    if MAG_idx:
        this_label_lfd_summary[MAG_idx,:] = u_svd[u_mag]
    if GRA_idx:
        this_label_lfd_summary[GRA_idx,:] = u_svd[u_gra]
    if EEG_idx:
        this_label_lfd_summary[EEG_idx,:] = u_svd[u_eeg]

    return this_label_lfd_summary
# Done