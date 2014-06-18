"""
=======================================
Compute spatial filter based on DeFleCT
=======================================

Compute spatial filter obeying several constraints.
See http://www.ncbi.nlm.nih.gov/pubmed/23616402 and
http://imaging.mrc-cbu.cam.ac.uk/meg/AnalyzingData/DeFleCT_SpatialFiltering_Tools.

The constraints that can be implemented are:
- Suppress cross-talk within a small set of labels
- Minimize cross-talk from other locations anywhere in the brain
- Minimize the effect of noise (represented by noise covariance matrix).
"""

# Author: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)

print(__doc__)

from mne.utils import logger

import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator
from mne.source_estimate import SourceEstimate
import numpy as np


## Example how to compute CTFs

data_path = sample.data_path()
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'

fname_label = [data_path + '/MEG/sample/labels/Aud-rh.label',
               data_path + '/MEG/sample/labels/Aud-lh.label',
               data_path + '/MEG/sample/labels/Vis-rh.label',
               data_path + '/MEG/sample/labels/Vis-lh.label']

# w = (tF^t + (i - tF^t S-1 P) (P^t S-1 P)-1 P^t) S-1  
# S = F F^t + lambda2 C

# i: desired discrete projections (usually 0s or 1s)
# t: target vector in source space
# F: leadfield matrix
# P: horizontally concatenated forward solutions for discrete source constraint
# C: noise covariance matrix

# read forward solution (sources in surface-based coordinates)
forward = mne.read_forward_solution(fname_fwd, surf_ori=True)

# get leadfield matrix from forward solution
leadfield = forward['sol']['data'][:, 2::3]

# average reference for the poor
for cc in np.arange(306,366,1):
    leadfield[:,cc] = leadfield[:,cc] - leadfield[:,cc].mean()

# MEG only for now, to avoid scaling/referencing problems with EEG
# leadfield = leadfield[:306,:]

# read label(s)
labels = [mne.read_label(ss) for ss in fname_label]

# to create source estimate, get rid of it later
inverse_operator = read_inverse_operator(fname_inv)

# read noise covariance matrix
noise_cov = mne.read_cov(fname_cov)
noise_cov_mat = noise_cov.data

# MEG only for now, to avoid scaling/referencing problems with EEG
# noise_cov_mat = noise_cov_mat[:306,:306]

# average reference for the poor
for cc in np.arange(306,366,1):
    noise_cov_mat[:,cc] = noise_cov_mat[:,cc] - noise_cov_mat[:,cc].mean()

# get whitener from noise_cov_mat via SVD
u_svd, s_svd, v_svd = np.linalg.svd(noise_cov_mat,
                                     full_matrices=False,
                                     compute_uv=True)

s_svd = np.sqrt(s_svd)
s_svd = np.power(s_svd, -1)
s_svd[80:] = 0  # regularization for the simple minded

whitener = np.dot(np.diag(s_svd), u_svd.T)
whitener = np.dot(v_svd.T, whitener)
whitener = np.vstack(whitener).T

mode = 'svd'
n_svd_comp = 1

label_lfd_summary = []  # sub-leadfield summaries
label_singvals = []  # singular values for sub-leadfields in labels
fwd_idx_all = []  #  source space indices in labels
# extract sub-leadfields for labels
# get first PCA component for each label
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
    fwd_idx_all.append(fwd_idx)

    # get sub-leadfield matrix for label vertices
    sub_leadfield = leadfield[:, fwd_idx + offset]

    # compute summary data for labels
    if mode == 'sum':  # sum across forward solutions in label
        logger.info("Computing sums within labels")
        this_label_lfd_summary = np.array(sub_leadfield.sum(axis=1))
        this_label_lfd_summary = np.vstack(this_label_lfd_summary).T
    elif mode == 'mean':
        logger.info("Computing means within labels")
        this_label_lfd_summary = np.array(sub_leadfield.mean(axis=1))
        this_label_lfd_summary = np.vstack(this_label_lfd_summary).T
    elif mode == 'svd':  # takes svd of forward solutions in label
        logger.info("Computing SVD within labels, using %d component(s)"
                    % n_svd_comp)

        # compute SVD of sub-leadfield
        u_svd, s_svd, _ = np.linalg.svd(sub_leadfield,
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
        my_comps = s_svd[0:n_svd_comp]
        comp_var = (100. * np.sum(my_comps * my_comps) /
                    np.sum(s_svd * s_svd))
        logger.info("Your %d component(s) explain(s) %.1f%% "
                    "variance." % (n_svd_comp, comp_var))
        this_label_lfd_summary = (u_svd[:, :n_svd_comp]
                                  * s_svd[:n_svd_comp][np.newaxis, :])
        # transpose required for conversion to "evoked"
        this_label_lfd_summary = this_label_lfd_summary.T

    # initialise or append to existing collection
    label_lfd_summary.append(this_label_lfd_summary)

label_lfd_summary = np.concatenate(label_lfd_summary, axis=0).T
# compute sum across forward solutions for labels, append to end

# rename variables to match paper
F = np.dot(whitener, leadfield)
P = np.dot(whitener, label_lfd_summary)

P_pinv = np.linalg.pinv(P)

# w = (tF^t + (i - tF^t S-1 P) (P^t S-1 P)-1 P^t) S-1  
# S = F F^t + lambda2 C

# without t
# w = ((i) (P^t S-1 P)-1 P^t) S-1 

lambda2 = 1 / 9.
pinv_rcond = 1 / 100.  # for np.linalg.pinv regularisation

n_chan, n_vert = leadfield.shape

# Let's do the monster matrix

S1 = np.dot(F,F.T)
S1trace = np.trace(S1)
cov_trace = np.trace(noise_cov_mat)
S = S1 + lambda2*(S1trace/cov_trace)*noise_cov_mat

S_pinv = np.linalg.pinv(S, pinv_rcond)

print "S_pinv"
print S_pinv.shape

S_P = np.dot(S_pinv, P)

P_S_P = np.dot(P.T, S_P)
print "S_pinv"
print S_pinv.shape

P_S_P = np.linalg.pinv(P_S_P, pinv_rcond)
print "S_pinv"
print S_pinv.shape

P_S_P_P = np.dot(P_S_P, P.T)
print "P_S_P_P"
print P_S_P_P.shape

P_S_P_P_S = np.dot(P_S_P_P, S_pinv)
print "P_S_P_P_S"
print P_S_P_P_S.shape

F_S_P = np.dot(F.T, S_P)
print "F_S_P"
print F_S_P.shape

# get spatial filters for each label
# w = (tF^t + (i - tF^t S-1 P) (P^t S-1 P)-1 P^t) S-1 
w = []
for ii in np.arange(4):
    i = np.eye(4)[ii,:]
    i = np.vstack(i).T
    t = np.zeros(n_vert)
    t = np.vstack(t).T
    t_F = np.dot(t, F.T)
    t_F_S_P = np.dot(t, F_S_P)
    i_t_F_S_P_P_S_P = np.dot((i - t_F_S_P), P_S_P_P)
    w_tmp = np.dot(t_F + i_t_F_S_P_P_S_P, S_pinv)

    w.append(w_tmp)

w = np.concatenate(w, axis=0)

print "w"
print w.shape

ctfs = np.dot(w, F)

ctfs = ctfs / ctfs.max()

if len(ctfs)==1:  # mne_analyze gets confused by a single data point
    ctfs = np.vstack((ctfs, ctfs))

# create source estimate object
vertno = [ss['vertno'] for ss in forward['src']]
stc_ctf = SourceEstimate(ctfs.T, vertno, tmin=0., tstep=1.)

fname_out = 'deflect_8'
stc_ctf.save(fname_out)
# assemble convenient sub-matrices

# assemble whole inverse matrix (add whitening)

# compute CTF

# write STC




