"""
================================================
Get inverse operator matrix
================================================

Get inverse matrix from an inverse operator for specific parameter settings

"""
# Author: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, read_inverse_operator


def get_matrix_from_inverse_operator(inverse_operator, forward, labels, method='dSPM', lambda2=3, mode='mean', svd_comp=1):
    """

    Get inverse matrix from an inverse operator for specific parameter settings
    Currently works only for fixed/loose orientation constraints
    For loose orientation constraint, the CTFs are computed for the radial component (pick_ori='normal')

    Parameters
    ----------
    inverse_operator: dict
        Inverse operator read with mne.read_inverse_operator.
    forward: Forward Solution
         Forward solution.
    method: 'MNE' | 'dSPM' | 'sLORETA'
        Inverse methods (for apply_inverse).
    labels: list of Label | None
        Labels for which CTFs shall be computed. If None, inverse matrix for all vertices will be returned
    lambda2 : float
        The regularization parameter (for apply_inverse).
    pick_ori : None | "normal"
        pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations (for apply_inverse).
        Determines whether whole inverse matrix G will have one or three rows per vertex.
        This will also affect summary measures for labels.
    mode: 'mean' | 'sum' | 'svd' |
        CTFs can be computed for different summary measures with labels:
        'sum' or 'mean': sum or means of sub-inverse for labels
        This corresponds to situations where labels can be assumed to be homogeneously activated.
        'svd': SVD components of sub-inverse for labels
        This is better suited for situations where activation patterns are assumed to be more variable.
        "sub-inverse" is the part of the inverse matrix that belongs to vertices within invidual labels
    svd_comp: integer
        Number of SVD components for which CTFs will be computed and output (irrelevant for 'sum' and 'mean')
        Explained variances within sub-inverses are shown in screen output


    Returns
    -------
    G: list numpy arrays
        Inverse matrix associated with inverse operator and specified parameters
    label_singvals: list of numpy arrays
        Singular values of svd for sub-inverses
        Provides information about how well labels are represented by chosen components
        Explained variances within sub-inverses are shown in screen output
    """
    if labels:
        print "\nAbout to process %d labels" % len(labels)
    else:
        print "\nComputing whole inverse operator."


    # in order to convert sub-leadfield matrix to evoked data type (pretending it's an epoch, see in loop below)
    # uses 'info' from forward solution, need to add 'sfreq' and 'proj'
    info = forward['info']
    info['sfreq'] = 1000. # add sfreq or it won't work
    info['projs'] = [] # add projs

    # create identity matrix as input for inverse operator
    nr_chan = forward['nchan']
    id_mat = np.eye(nr_chan)

    # convert identity matrix to evoked data type (pretending it's an epoch)
    ev_id = mne.epochs.EpochsArray(id_mat[None, :, :], info, np.zeros((1, 3), dtype=int)).average()

    snr = 3.0
    lambda2 = 1.0 / snr ** 2

    # apply inverse operator to identity matrix in order to get inverse matrix
    # free orientation constraint not possible because apply_inverse would combined components
    G_mat_op = apply_inverse(ev_id, inverse_operator, lambda2=lambda2, method=method, pick_ori='normal')
    print "\nDimension of inverse matrix:"
    print G_mat_op.shape

    # turn source estimate into numpty array
    G_mat = G_mat_op.data

    G_summary = np.array(0)
    label_singvals = []     # if mode='svd', this will collect all SVD singular values for labels

    if labels:
        for ll in labels:
            print ll
            if ll.hemi == 'rh':
                # for RH labels, add number of LH vertices
                offset = forward['src'][0]['vertno'].shape[0]
                # remember whether we are in the LH or RH
                this_hemi = 1
            elif ll.hemi == 'lh':
                offset = 0
                this_hemi = 0
            else:
                print "Cannot determine hemisphere of label.\n"

            # get vertices on cortical surface inside label
            idx = np.intersect1d(ll.vertices, forward['src'][this_hemi]['vertno'])

            # get vertices in source space inside label
            fwd_idx = np.searchsorted(forward['src'][this_hemi]['vertno'], idx)

            # get sub-inverse for label vertices, one row per vertex
            G_lbl = G_mat[fwd_idx + offset, :]

            # compute summary data for labels
            if mode.lower() == 'sum':  # takes sum across estimators in label
                print "Computing sums within labels"
                this_G_summary = G_lbl.sum(axis=0)

            elif mode.lower() == 'mean':
                print "Computing means within labels"
                this_G_summary = G_lbl.mean(axis=0)

            elif mode.lower() == 'svd':  # takes svd of sub-inverse in label
                print "Computing SVD within labels, using %d component(s)" % svd_comp

                # compute SVD of sub-inverse
                U_svd, s_svd, V_svd = np.linalg.svd(G_lbl.T, full_matrices=False, compute_uv=True)

                # keep singular values (might be useful to some people)
                label_singvals.append(s_svd)

                # get first svd_comp components, weighted with their corresponding singular values
                print "first 5 singular values:"
                print s_svd[0:5]
                print "(This tells you something about variability of estimators in sub-inverse for label)"
                # explained variance by chosen components within sub-inverse
                my_comps = s_svd[0:svd_comp]
                comp_var = 100*np.sum( np.power(my_comps, 2) ) / np.sum( np.power(s_svd, 2))
                print "Your %d component(s) explain(s) %.1f%% variance.\n" % (svd_comp, comp_var)
                this_G_summary = np.dot( U_svd[:,0:svd_comp], np.diag(s_svd[0:svd_comp]) )
                this_G_summary = this_G_summary.T

            if G_summary.shape == ():
                G_summary = this_G_summary
            else:
                G_summary = np.vstack((G_summary, this_G_summary))

        G = G_summary

    else:   # no labels provided: return whole matrix
        G = G_mat

    return G, label_singvals



def plot_mne_cross_talk_function(inverse_operator, forward, labels, method='dSPM',
                              lambda2=1 / 9., mode='mean', svd_comp=1):
    """Compute cross-talk functions (CTFs) for linear estimators

    Compute cross-talk functions (CTF) in labels for a combination of inverse operator and forward solution
    CTFs are computed for test sources that are perpendicular to cortical surface

    Parameters
    ----------
    inverse_operator: dict
        Inverse operator read with mne.read_inverse_operator.
    forward: dict
         Forward solution, created with "force_fixed=True"
         Note: (Bad) channels not included in forward solution will not be used in CTF computation.
    method: 'MNE' | 'dSPM' | 'sLORETA'
        Inverse method for which CTFs shall be computed.
    labels: list of Label
        Labels for which CTFs shall be computed.
    lambda2 : float
        The regularization parameter.
    mode: 'mean' | 'sum' | 'svd' |
        CTFs can be computed for different summary measures with labels:
        'sum' or 'mean': sum or means of sub-inverses for labels
        This corresponds to situations where labels can be assumed to be homogeneously activated.
        'svd': SVD components of sub-inverses for labels
        This is better suited for situations where activation patterns are assumed to be more variable.
        "sub-inverse" is the part of the inverse matrix that belongs to vertices within invidual labels
    svd_comp: integer
        Number of SVD components for which CTFs will be computed and output (irrelevant for 'sum' and 'mean')
        Explained variances within sub-inverses are shown in screen output

    Returns
    -------
    stc_ctf : SourceEstimate
        The CTFs for the specified labels
        If mode='svd': svd_comp components per label are created
        (i.e. svd_comp successive time points in mne_analyze)
        The last sample is the summed CTF across all labels
    label_singvals: list of numpy arrays
        Singular values of svd for sub-inverses
        Provides information about how well labels are represented by chosen components
        Explained variances within sub-inverses are shown in screen output
    """

    # get the inverse matrix corresponding to inverse operator
    G, label_singvals = get_matrix_from_inverse_operator(inverse_operator, forward, labels=labels,
                                                method=method, lambda2=lambda2, mode=mode, svd_comp=svd_comp)

    # get the leadfield matrix from forward solution
    leadfield = forward['sol']['data']

    # compute cross-talk functions (CTFs)
    ctfs = np.dot(G, leadfield)

    # compute sum across forward solutions for labels, append to end
    ctfs = np.vstack((ctfs, ctfs.sum(axis=0)))

    # create a dummy source estimate and put in the CTFs, in order to write them to STC file

    # in order to convert sub-leadfield matrix to evoked data type (pretending it's an epoch, see in loop below)
    # uses 'info' from forward solution, need to add 'sfreq' and 'proj'
    info = forward['info']
    info['sfreq'] = 1000. # add sfreq or it won't work
    info['projs'] = [] # add projs

    # create identity matrix as input for inverse operator
    nr_chan = forward['nchan']
    id_mat = np.eye(nr_chan)

    # convert identity matrix to evoked data type (pretending it's an epoch)
    ev_id = mne.epochs.EpochsArray(id_mat[None, :, 0:nr_labels+1], info, np.zeros((1, 3), dtype=int)).average()

    # apply inverse operator to dummy data to create dummy source estimate, in this case fixed orientation constraint
    stc_ctf = apply_inverse(ev_id, inverse_operator, lambda2=3, method='MNE')

    # insert CTF into source estimate object
    stc_ctf._data = ctfs.T

    return stc_ctf, label_singvals




## Example how to compute CTFs

data_path = sample.data_path()
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'

fname_label = [data_path + '/MEG/sample/labels/Vis-rh.label',
               data_path + '/MEG/sample/labels/Vis-lh.label',
               data_path + '/MEG/sample/labels/Aud-rh.label',
               data_path + '/MEG/sample/labels/Aud-lh.label']

# In order to get leadfield with fixed source orientation, read forward solution again
forward = mne.read_forward_solution(fname_fwd, force_fixed=True, surf_ori=True)

# read label(s)
labels = [mne.read_label(ss) for ss in fname_label]
nr_labels = len(labels)

inverse_operator = read_inverse_operator(fname_inv)

fname_stem = 'ctf_3'
for method in ('MNE', 'dSPM', 'sLORETA'):
    stc_ctf, label_singvals = plot_mne_cross_talk_function(inverse_operator, forward, labels, method=method,
                                  lambda2=1 / 9., mode='svd', svd_comp=3)

    fname_out = fname_stem + '_' + method
    print "Writing CTFs to files %s" % fname_out
    # signed
    stc_ctf.save(fname_out)

    # unsigned
    stc_ctf._data = np.abs( stc_ctf.data )
    stc_ctf.save(fname_out+'_abs')
