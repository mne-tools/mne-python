"""
===================================================================
Compute and plot PSF and CTF for labels using MNE
CTFs should look similar, as should PSF and CTF for MNE
===================================================================
"""

# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)

import numpy as np

from mayavi import mlab

from mne import read_forward_solution, read_cov, read_evokeds, read_label
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, write_inverse_operator
from mne.minimum_norm.psf_ctf import cross_talk_function, point_spread_function

# only effective with EEG/MEG forward solution
pick_meg = True
pick_eeg = True

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects/'

fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
# fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-oct-6-fwd.fif'

# For OUTPUT of inverse operator created below
fname_inv = data_path + 'test_PSF_CTF_plot_1June17-inv.fif'

# covariance matrix for inverse operator
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'

fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'

fname_label = [data_path + '/MEG/sample/labels/Aud-rh.label',
               data_path + '/MEG/sample/labels/Aud-lh.label',
               data_path + '/MEG/sample/labels/Vis-rh.label',
               data_path + '/MEG/sample/labels/Vis-lh.label']


# read forward solution
forward = read_forward_solution(fname_fwd, force_fixed=True, surf_ori=True)

# read covariance matrix
noise_cov = read_cov(fname_cov)

# read evoked data
evoked = read_evokeds(fname_evoked, condition=0)

evoked.pick_types(meg=pick_meg, eeg=pick_eeg)

info = evoked.info

# read label(s)
labels = [read_label(ss) for ss in fname_label]

# make inverse operator based on specified forward solution inverse_operator
inverse_operator = make_inverse_operator(info, forward,
                                    noise_cov=noise_cov, fixed=True, depth=None)

write_inverse_operator(fname_inv, inverse_operator)

# regularisation parameter
snr = 3.
lambda2 = 1.0 / snr ** 2
### mode = 'svd'
mode = 'svd'
n_svd_comp = 1

# PSF/CTF for MNE
method = 'MNE'  # can be 'MNE', 'dSPM', or 'sLORETA'

stc_ctf_mne, ctf_s_mne = cross_talk_function(
    inverse_operator, forward, labels, method=method, lambda2=lambda2,
    signed=True, mode=mode, n_svd_comp=n_svd_comp)

stc_psf_mne, psf_s_mne = point_spread_function(
    inverse_operator, forward, method=method, labels=labels,
    lambda2=lambda2, pick_ori=None, mode=mode, n_svd_comp=n_svd_comp)

# PSF/CTF for sLORETA
method = 'sLORETA'  # can be 'MNE', 'dSPM', or 'sLORETA'

stc_ctf_lor, ctf_s_lor = cross_talk_function(
    inverse_operator, forward, labels, method=method, lambda2=lambda2,
    signed=True, mode=mode, n_svd_comp=n_svd_comp)

stc_psf_lor, psf_s_lor = point_spread_function(
    inverse_operator, forward, method=method, labels=labels,
    lambda2=lambda2, pick_ori=None, mode=mode, n_svd_comp=n_svd_comp)

stc_corr1 = np.corrcoef(stc_ctf_mne.data[:, 0], stc_psf_mne.data[:, 0])
stc_corr2 = np.corrcoef(stc_ctf_mne.data[:, 0], stc_ctf_lor.data[:, 0])

# Plot
time_label = "MNE CTF %d"
brain_ctf_mne = stc_ctf_mne.plot(hemi='rh', subjects_dir=subjects_dir,
                                 time_label=time_label, time_viewer=False,
                                 figure=mlab.figure(size=(500, 500)))

time_label = "MNE PSF %d"
brain_psf_mne = stc_psf_mne.plot(hemi='rh', subjects_dir=subjects_dir,
                                 time_label=time_label, time_viewer=False,
                                 figure=mlab.figure(size=(500, 500)))

time_label = "sLor CTF %d"
brain_ctf_mne = stc_ctf_lor.plot(hemi='rh', subjects_dir=subjects_dir,
                                 time_label=time_label, time_viewer=False,
                                 figure=mlab.figure(size=(500, 500)))

time_label = "sLor PSF%d"
brain_psf_mne = stc_psf_lor.plot(hemi='rh', subjects_dir=subjects_dir,
                                 time_label=time_label, time_viewer=False,
                                 figure=mlab.figure(size=(500, 500)))
