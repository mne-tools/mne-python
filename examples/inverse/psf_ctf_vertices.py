# -*- coding: utf-8 -*-
"""
.. _ex-psd-ctf:

==================================================================
Plot point-spread functions (PSFs) and cross-talk functions (CTFs)
==================================================================

Visualise PSF and CTF at one vertex for sLORETA.
"""
# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

# %%

import numpy as np

import mne
from mne.datasets import sample
from mne.minimum_norm import (make_inverse_resolution_matrix, get_cross_talk,
                              get_point_spread)

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path / 'subjects'
meg_path = data_path / 'MEG' / 'sample'
fname_fwd = meg_path / 'sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_fwd_vol = meg_path / 'sample_audvis-meg-vol-7-fwd.fif'
fname_cov = meg_path / 'sample_audvis-cov.fif'
fname_evo = meg_path / 'sample_audvis-ave.fif'

# read forward solution
forward = mne.read_forward_solution(fname_fwd)
# forward operator with fixed source orientations
mne.convert_forward_solution(forward, surf_ori=True,
                             force_fixed=True, copy=False)
# read a volumetric forward solution, too
forward_vol = mne.read_forward_solution(fname_fwd_vol)

# noise covariance matrix
noise_cov = mne.read_cov(fname_cov)

# evoked data for info
evoked = mne.read_evokeds(fname_evo, 0)

# make inverse operator from forward solution
# free source orientation
inverse_operator = mne.minimum_norm.make_inverse_operator(
    info=evoked.info, forward=forward, noise_cov=noise_cov, loose=0.,
    depth=None)
inverse_operator_vol = mne.minimum_norm.make_inverse_operator(
    info=evoked.info, forward=forward_vol, noise_cov=noise_cov)

# regularisation parameter
snr = 3.0
lambda2 = 1.0 / snr ** 2

# compute resolution matrix for sLORETA
rm_lor = make_inverse_resolution_matrix(forward, inverse_operator,
                                        method='sLORETA', lambda2=lambda2)

# compute resolution matrix for sLORETA for the volume, too
rm_lor_vol = make_inverse_resolution_matrix(
    forward_vol, inverse_operator_vol, method='sLORETA', lambda2=lambda2)

# get PSF and CTF for sLORETA at one vertex
sources = [1000]

stc_psf = get_point_spread(rm_lor, forward['src'], sources, norm=True)

stc_ctf = get_cross_talk(rm_lor, forward['src'], sources, norm=True)
del rm_lor

# for the volume, pick a source that was close to the surface location and
# compute the PSF at that source
sources_vol = [448]
stc_psf_vol = get_point_spread(
    rm_lor_vol, forward_vol['src'], sources_vol, norm=True)
del rm_lor_vol

##############################################################################
# Visualize
# ---------
# PSF:

# Which vertex corresponds to selected source
vertno_lh = forward['src'][0]['vertno']
verttrue = [vertno_lh[sources[0]]]  # just one vertex

# find vertices with maxima in PSF and CTF
vert_max_psf = vertno_lh[stc_psf.data.argmax()]
vert_max_ctf = vertno_lh[stc_ctf.data.argmax()]

brain_psf = stc_psf.plot(
    'sample', 'inflated', 'lh', views='ven', subjects_dir=subjects_dir)
brain_psf.add_text(0.1, 0.9, 'sLORETA PSF', 'title', font_size=16)

# True source location for PSF
brain_psf.add_foci(verttrue, coords_as_verts=True, scale_factor=1., hemi='lh',
                   color='green')

# Maximum of PSF
brain_psf.add_foci(vert_max_psf, coords_as_verts=True, scale_factor=1.,
                   hemi='lh', color='black')

# %%
# CTF:

brain_ctf = stc_ctf.plot(
    'sample', 'inflated', 'lh', views='ven', subjects_dir=subjects_dir)
brain_ctf.add_text(0.1, 0.9, 'sLORETA CTF', 'title', font_size=16)
brain_ctf.add_foci(verttrue, coords_as_verts=True, scale_factor=1., hemi='lh',
                   color='green')

# Maximum of CTF
brain_ctf.add_foci(vert_max_ctf, coords_as_verts=True, scale_factor=1.,
                   hemi='lh', color='black')

# %%
# The green spheres indicate the true source location, and the black
# spheres the maximum of the distribution.
#
# Volumetric source estimates
# ---------------------------
# We can do these same operations for volumetric source estimates:

# Which vertex corresponds to selected source
src_vol = forward_vol['src']
verttrue_vol = src_vol[0]['vertno'][sources_vol]

# find vertex with maximum in PSF
max_vert_idx, _ = np.unravel_index(
    stc_psf_vol.data.argmax(), stc_psf_vol.data.shape)
vert_max_ctf_vol = src_vol[0]['vertno'][[max_vert_idx]]

# plot them
brain_psf_vol = stc_psf_vol.plot_3d(
    'sample', src=forward_vol['src'], views='ven', subjects_dir=subjects_dir,
    volume_options=dict(alpha=0.5))
brain_psf_vol.add_text(
    0.1, 0.9, 'Volumetric sLORETA PSF', 'title', font_size=16)
brain_psf_vol.add_foci(
    verttrue_vol, coords_as_verts=True,
    scale_factor=1, hemi='vol', color='green')
brain_psf_vol.add_foci(
    vert_max_ctf_vol, coords_as_verts=True,
    scale_factor=1.25, hemi='vol', color='black', alpha=0.3)
