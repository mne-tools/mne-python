# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk> and Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

# To do:
# add bad channels?
# other ~linear estimators, beamformers etc.?

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, read_inverse_operator



def mne_point_spread_function(inverse_operator, forward, labels, method='dSPM',
                              lambda2=1 / 9., pick_ori=None, mode='mean', svd_comp=1):
    """Compute point-spread functions (PSFs) for linear estimators

    Compute point-spread functions (PSF) in labels for a combination of inverse operator and forward solution
    PSFs are computed for test sources that are perpendicular to cortical surface

    Parameters
    ----------
    inverse_operator: dict
        Inverse operator read with mne.read_inverse_operator.
    forward: dict
         Forward solution, created with "surf_ori=True" and "force_fixed=False"
         Note: (Bad) channels not included in forward solution will not be used in PSF computation.
    method: 'MNE' | 'dSPM' | 'sLORETA'
        Inverse method for which PSFs shall be computed (for apply_inverse).
    labels: list of Label
        Labels for which PSFs shall be computed.
    lambda2 : float
        The regularization parameter (for apply_inverse).
    pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations (for apply_inverse).
    mode: 'mean' | 'sum' | 'svd' |
        PSFs can be computed for different summary measures with labels:
        'sum' or 'mean': sum or means of sub-leadfields for labels
        This corresponds to situations where labels can be assumed to be homogeneously activated.
        'svd': SVD components of sub-leadfields for labels
        This is better suited for situations where activation patterns are assumed to be more variable.
        "sub-leadfields" are the parts of the forward solutions that belong to vertices within invidual labels
    svd_comp: integer
        Number of SVD components for which PSFs will be computed and output (irrelevant for 'sum' and 'mean')
        Explained variances within sub-leadfields are shown in screen output

    Returns
    -------
    stc_psf : SourceEstimate
        The PSFs for the specified labels
        If mode='svd': svd_comp components per label are created
        (i.e. svd_comp successive time points in mne_analyze)
        The last sample is the summed PSF across all labels
        Scaling of PSFs is arbitrary, and may differ greatly among methods
        (especially for MNE compared to noise-normalized estimates)
    evoked_fwd: Evoked
        Forward solutions corresponding to PSFs in stc_psf
        If mode='svd': svd_comp components per label are created
        (i.e. svd_comp successive time points in mne_analyze)
        The last sample is the summed forward solution across all labels
        (sum is taken across summary measures)
    label_singvals: list of numpy arrays
        Singular values of svd for sub-leadfields
        Provides information about how well labels are represented by chosen components
        Explained variances within sub-leadfields are shown in screen output



    """
    if mode.lower() not in ['mean', 'svd']:
        raise ValueError('mode must be ''svd'' or ''mean''. Got %s.' % mode.lower())

    print "\nAbout to process %d labels" % len(labels)

    # get whole leadfield matrix with normal dipole components
    Lfd = forward['sol']['data'][:, 2::3]
    # n_channels, n_sources = Lfd.shape

    # in order to convert sub-leadfield matrix to evoked data type (pretending it's an epoch, see in loop below)
    # uses 'info' from forward solution, need to add 'sfreq' and 'proj'
    info = forward['info']
    info['sfreq'] = 1000. # add sfreq or it won't work
    info['projs'] = [] # add projs

    # loop over labels
    label_psf_summary = np.array(0)  # will contain means of subleadfields for all labels
    label_singvals = []     # if mode='svd', this will collect all SVD singular values for labels

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

        # get sub-leadfield matrix for label vertices
        Lfd_lbl = Lfd[:, fwd_idx + offset]

        # compute summary data for labels
        if mode.lower() == 'sum':  # takes sum across forward solutions in label
            print "Computing sums within labels"
            this_label_psf_summary = Lfd_lbl.sum(axis=1)

        elif mode.lower() == 'mean':
            print "Computing means within labels"
            this_label_psf_summary = Lfd_lbl.mean(axis=1)

        elif mode.lower() == 'svd':  # takes svd of forward solutions in label
            print "Computing SVD within labels, using %d component(s)" % svd_comp

            # compute SVD of sub-leadfield
            U_svd, s_svd, V_svd = np.linalg.svd(Lfd_lbl, full_matrices=False, compute_uv=True)

            # keep singular values (might be useful to some people)
            label_singvals.append(s_svd)

            # get first svd_comp components, weighted with their corresponding singular values
            print "first 5 singular values:"
            print s_svd[0:5]
            print "(This tells you something about variability of forward solutions in sub-leadfield for label)"
            # explained variance by chosen components within sub-leadfield
            my_comps = s_svd[0:svd_comp]
            comp_var = 100*np.sum( np.power(my_comps, 2) ) / np.sum( np.power(s_svd, 2))
            print "Your %d component(s) explain(s) %.1f%% variance.\n" % (svd_comp, comp_var)
            this_label_psf_summary = np.dot( U_svd[:,0:svd_comp], np.diag(s_svd[0:svd_comp]) )
            # transpose required for conversion to "evoked"
            this_label_psf_summary = this_label_psf_summary.T

        # initialise or append to existing collection
        if label_psf_summary.shape == ():
            label_psf_summary = this_label_psf_summary
        else:
            label_psf_summary = np.vstack((label_psf_summary, this_label_psf_summary))

    # compute sum across forward solutions for labels, append to end
    label_psf_summary = np.vstack((label_psf_summary, label_psf_summary.sum(axis=0)))
    # transpose required for conversion to "evoked"
    label_psf_summary = label_psf_summary.T

    # convert sub-leadfield matrix to evoked data type (a bit of a hack)
    evoked_fwd = mne.epochs.EpochsArray(label_psf_summary[None, :, :], info,
                                        np.zeros((1, 3), dtype=int)).average()

    # compute PSFs by applying inverse operator to sub-leadfields
    print "About to apply inverse operator for method='%s' and lambda2=%f\n" % (method, lambda2)
    stc_psf = apply_inverse(evoked_fwd, inverse_operator, lambda2, method=method, pick_ori=pick_ori)

    return stc_psf, evoked_fwd, label_singvals


## Compute PSFs for labels in MNE sample data set

data_path = sample.data_path()
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_label = [data_path + '/MEG/sample/labels/Vis-rh.label',
               data_path + '/MEG/sample/labels/Vis-lh.label',
               data_path + '/MEG/sample/labels/Aud-rh.label',
               data_path + '/MEG/sample/labels/Aud-lh.label']


# read forward solution (sources in surface-based coordinates)
forward = mne.read_forward_solution(fname_fwd, surf_ori=True)
# read inverse operator
inverse_operator = read_inverse_operator(fname_inv)
# read label(s)
labels = [mne.read_label(ss) for ss in fname_label]

# regularisation parameter
snr = 3.0
lambda2 = 1.0 / snr ** 2

for method in ('MNE', 'dSPM', 'sLORETA'):
    stc_psf, evoked_fwd, label_eigval = mne_point_spread_function(inverse_operator, forward, method=method,
                                     labels=labels, lambda2=lambda2, mode='svd', svd_comp=2)
    fname_out = method + '_3'
    print "Writing STC to file: %s" % fname_out
    stc_psf.save(fname_out)
