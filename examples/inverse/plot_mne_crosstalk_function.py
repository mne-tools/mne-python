"""
===================================================================
Compute cross-talk functions (CTFs) for labels for MNE/dSPM/sLORETA
===================================================================

CTFs are computed for four labels in the MNE sample data set
for linear inverse operators (MNE, dSPM, sLORETA).
CTFs describe the sensitivity of a linear estimator (e.g. for
one label) to sources across the cortical surface. Sensitivity
to sources outside the label is undesirable, and referred to as
"leakage" or "cross-talk".
"""
# Author: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)

from mayavi import mlab

import mne
from mne.datasets import sample
from mne.minimum_norm import cross_talk_function, read_inverse_operator

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects/'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
fname_label = [data_path + '/MEG/sample/labels/Aud-rh.label',
               data_path + '/MEG/sample/labels/Aud-lh.label',
               data_path + '/MEG/sample/labels/Vis-rh.label',
               data_path + '/MEG/sample/labels/Vis-lh.label']

# read forward solution
forward = mne.read_forward_solution(fname_fwd)

# read label(s)
labels = [mne.read_label(ss) for ss in fname_label]

inverse_operator = read_inverse_operator(fname_inv)

# regularisation parameter
snr = 3.0
lambda2 = 1.0 / snr ** 2
mode = 'svd'
n_svd_comp = 1

method = 'MNE'  # can be 'MNE', 'dSPM', or 'sLORETA'
stc_ctf_mne = cross_talk_function(
    inverse_operator, forward, labels, method=method, lambda2=lambda2,
    signed=False, mode=mode, n_svd_comp=n_svd_comp)

method = 'dSPM'
stc_ctf_dspm = cross_talk_function(
    inverse_operator, forward, labels, method=method, lambda2=lambda2,
    signed=False, mode=mode, n_svd_comp=n_svd_comp)

time_label = "MNE %d"
brain_mne = stc_ctf_mne.plot(hemi='rh', subjects_dir=subjects_dir,
                             time_label=time_label,
                             figure=mlab.figure(size=(500, 500)))

time_label = "dSPM %d"
brain_dspm = stc_ctf_dspm.plot(hemi='rh', subjects_dir=subjects_dir,
                               time_label=time_label,
                               figure=mlab.figure(size=(500, 500)))

# Cross-talk functions for MNE and dSPM (and sLORETA) have the same shapes
# (they may still differ in overall amplitude).
# Point-spread functions (PSfs) usually differ significantly.
