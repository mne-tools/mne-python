# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)

from copy import deepcopy
import numpy as np
import scipy
from mne.utils import logger, verbose
from mne import pick_types, pick_channels_forward, pick_channels_cov
from mne import SourceEstimate
from mne.cov import prepare_noise_cov
from mne.cov import regularize as cov_regularize
from mne.proj import make_eeg_average_ref_proj 
from mne.minimum_norm.inverse import _prepare_forward


def _deflect_matrix(F, P, i, t, lambda2_S=1/9.):
    """ Computes the linear DeFleCT estimator given its component matrices

    Parameters:
    -----------
    F : 2D numpy array (n_chan x n_vert)
        leadfield/forward solution matrix
    P : 2D numpy array (n_chan x n_comp)
        projection matrix
    i : numpy vector (n_comp)
        desired sensitivity to columns in P
    t : numpy vector (n_vert)
        desired CTF associated with spatial filter w
    lambda2_S : scalar
        regularization parameter for Gram matrix (S)
    (matrix notation taken from Hauk&Stenroos HBM 2014 paper)

    Returns:
    --------
    w : numpy vector (n_chan)
        Spatial filter/linear estimator for first column of P
    """

    # DeFleCT formula:
    # w = (tF^t + (i - tF^t S-1 P) (P^t S-1 P)-1 P^t) S-1  
    # S = F F^t + lambda2 C
    # in the following, variable names reflect the chain of matrix operations
    # small t: transpose; small i: (pseudo)inverse
    # brackets and +-* operations are not reflected in variable names

    [n_chan, n_comp] = P.shape
    [n_chan2, n_vert] = F.shape
    if n_chan != n_chan2:
        raise ValueError("Number of channels doesn't match between P and F: "
                            "%d vs %d" % (n_chan, n_chan2))

    n_comp2 = i.shape[0]
    if n_comp != n_comp2:
        raise ValueError("Number of components doesn't match between P and i: "
                            "%d vs %d" % (n_comp, n_comp2))

    n_vert2 = t.shape[1]
    if n_vert != n_vert2:
        raise ValueError("Number of vertices doesn't match between F and t: "
                            "%d vs %d" % (n_vert, n_vert2))

    reg_mat = np.eye( n_chan )   # for regulatisation 

    # compute regularized inverse of Gram matrix S
    S = np.dot(F,F.T)
    S_trace = np.trace(S)
    reg_trace = np.trace(reg_mat)
    S = S + lambda2_S*(S_trace/reg_trace)*reg_mat

    Si = scipy.linalg.pinv(S)

    Si_P = Si.dot(P)

    Pt_Si_P = P.T.dot(Si_P)

    Pt_Si_P_cond = np.linalg.cond( Pt_Si_P )
    logger.info("Condition number of Pt_Si_P: %f\n" % Pt_Si_P_cond)

    PtSiPi = scipy.linalg.pinv(Pt_Si_P)

    PtSiPi_Pt = PtSiPi.dot(P.T)

    Ft_Si_P = F.T.dot(Si_P)

    t_Ft = np.dot(t, F.T)
    t_Ft_Si_P = np.dot(t, Ft_Si_P)

    #i_t_F_S_P_P_S_P = np.dot((i - t_F_S_P), PSPi_Pt)
    i_t_Ft_Si_P_PtSiPi_Pt = (i - t_Ft_Si_P).dot(PtSiPi_Pt)

    w = (t_Ft + i_t_Ft_Si_P_PtSiPi_Pt).dot(Si)

    return w


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


def _deflect_make_subleadfields(labels, forward, leadfield, mode='svd',
                                                   n_svd_comp=1, verbose=None):
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
        elif mode == 'mean':
            logger.info("Computing means within labels")
            this_label_lfd_summary = np.array(sub_leadfield.mean(axis=1))
        elif mode == 'svd':  # takes svd of forward solutions in label
            ch_names = forward['info']['ch_names']

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


def deflect_make_estimator(forward, noise_cov, labels, lambda2_cov=3/10.,
                           lambda2_S=1/9., pick_meg=True, pick_eeg=False,
                           mode='svd', n_svd_comp=1, verbose=None):
    """
    Create the DeFleCT estimator for a set of labels

    Parameters:
    -----------
    forward : forward solution object
        (assumes surf_ori=True)
    noise_cov : noise covariance matrix object
    lambda2_cov : scalar
        regularisation paramter for noise covariance matrix (whitening)
    pick_meg : True | False | 'grad' | 'mag'
        which MEG channels to pick
    pick_eeg : True | False
        which EEG channels to pick
    labels : list of labels
        first one is the target for DeFleCT
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
    n_svd_comp : integer
        Number of SVD components for which PSFs will be computed and output
        (irrelevant for 'sum' and 'mean'). Explained variances within
        sub-leadfields are shown in screen output.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns:
    --------
    w : numpy array (1 x n_chan)
        spatial filter for first column of P
    F : 2D numpy array (n_chan x n_vert)
        whitened leadfield matrix
    P : 2D numpy array (n_chan x n_comp)
        whitened projection matrix
    noise_cov_mat : 2D numpy array (n_chan x n_chan)
        noise covariance matrix as used in DeFleCT
    whitener : 2D numpy array (df x n_chan)
        whitening matrix as used in DeFleCT
    """
    
    # get wanted channels
    picks = pick_types(forward['info'], meg=pick_meg, eeg=pick_eeg, eog=False,
                                                    stim=False, exclude='bads')
    
    fwd_ch_names_all = [c['ch_name'] for c in forward['info']['chs']]
    fwd_ch_names = [fwd_ch_names_all[pp] for pp in picks]
    ch_names = [c for c in fwd_ch_names
                if ((c not in noise_cov['bads']) and
                    (c not in forward['info']['bads'])) and
                    (c in noise_cov.ch_names)]

    if (not len(forward['info']['bads']) == len(noise_cov['bads']) or
         not all([b in noise_cov['bads'] for b in forward['info']['bads']])):
        logger.info('forward["info"]["bads"] and noise_cov["bads"] do not '
            'match excluding bad channels from both')

    # reduce forward to desired channels
    forward = pick_channels_forward(forward, ch_names)
    noise_cov = pick_channels_cov(noise_cov, ch_names)
    
    logger.info("Noise covariance matrix has %d channels." %
                                                    noise_cov.data.shape[0])

    info_fwd = deepcopy(forward['info'])
    info_fwd['sfreq'] = 1000.
    if pick_eeg:        
        avgproj = make_eeg_average_ref_proj(info_fwd, activate=True)
        info_fwd['projs'] = []
        info_fwd['projs'].append(avgproj)
    else:
        info_fwd['projs'] = noise_cov['projs']
    
    if lambda2_cov:  # regularize covariance matrix "old style"
        lbd = lambda2_cov
        noise_cov_reg = cov_regularize(noise_cov, info_fwd, mag=lbd['mag'],
                                    grad=lbd['gra'], eeg=lbd['eeg'], proj=True)
    else:  # use cov_mat as is
        noise_cov_reg = noise_cov

    fwd_info, leadfield, noise_cov_fwd, whitener, n_nzero = _prepare_forward(
                             forward, info_fwd, noise_cov_reg,
                             pca=False, rank=None, verbose=None)
    leadfield = leadfield[:,2::3]  # assumes surf_ori=True, (normal component)
    n_chan, n_vert = leadfield.shape
    logger.info("Leadfield has dimensions %d by %d\n" % (n_chan, n_vert))

    # if EEG present: remove mean of columns for EEG (average-reference)
    if pick_eeg:
        print "Referening EEG\n"
        EEG_idx = [cc for cc in range(len(ch_names)) if ch_names[cc][:3]=='EEG']
        nr_eeg = len(EEG_idx)
        lfdmean = leadfield[EEG_idx,:].mean(axis=0)
        leadfield[EEG_idx,:] = leadfield[EEG_idx,:] - lfdmean[np.newaxis,:]

    #### CREATE SUBLEADFIELDs FOR LABELS
    # extract SUBLEADFIELDS for labels
    label_lfd_summary = _deflect_make_subleadfields(labels, forward, leadfield,
                            mode='svd', n_svd_comp=n_svd_comp, verbose=None)

    #### COMPUTE DEFLECT ESTIMATOR
    # rename variables to match paper
    F = np.dot( whitener, leadfield )
    P = np.dot( whitener, label_lfd_summary )
    nr_comp = P.shape[1] # number of forward solutions as columns of P

    i = np.eye( nr_comp )[0,:].T          # desired sensitivity to columns in P
    t = np.zeros(n_vert).T[np.newaxis,:]  # desired CTF for spatial filter w

    # Compute DeFleCT ESTIMATOR
    w = _deflect_matrix(F, P, i, t, lambda2_S)

    # add whitener on the right (i.e. input should be unwhitened)
    w = w.dot(whitener)

    return w, ch_names, leadfield, label_lfd_summary, noise_cov_fwd, whitener

# Done


def apply_spatial_filters_epochs(spatial_filters, epochs):
    """
    Apply spatial filter(s) (e.g. DeFleCT) to epochs

    Parameters:
    -----------
    spatial_filters : list of numpy arrays
        each array a spatial filter (e.g. from DeFleCT)
    epochs : epoch object
        from mne.Epochs
    Number of channels in spatial filter(s) and epochs must match

    Returns:
    --------
    label_tc : list of source estimate objects
        time courses of spatial filter outputs for labels
    """

    # check for EOG channel(s)
    eog_idx = pick_types(epochs.info, meg=False, eeg=False, eog=True)

    # remove EOG channel(s) if necessary
    if eog_idx:
        epochs_use = epochs.drop_channels( [epochs.info['ch_names'][eog_idx]],
                                                                    copy=True)
    else:
        epochs_use = epochs

    epoch_data = epochs_use.get_data()
    n_epochs = epoch_data.shape[0]

    label_tc_list = [this_filter.dot(epoch_data[:,:,:]) for this_filter in
                                                              spatial_filters]
    label_tc_np = np.squeeze(label_tc_list)

    n_filters = len(spatial_filters)
    #vertno = [ [np.array([ii+1]) for ii in np.arange(len(spatial_filters))]
    vertno = [np.arange(n_filters), np.array([])]

    label_tc = [SourceEstimate(label_tc_np[:,ee,:], vertno, tmin=epochs.tmin,
                    tstep=1./epochs.info['sfreq']) for ee in
                                                           np.arange(n_epochs)]

    return label_tc


# Done