# -*- coding: utf-8 -*-
"""
.. _ex-psd-vol:

===============================================
Plot point-spread functions (PSFs) for a volume
===============================================

Visualise PSF at one volume vertex for sLORETA.
"""
# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

# %%

import numpy as np

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_resolution_matrix, get_point_spread

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path / 'subjects'
meg_path = data_path / 'MEG' / 'sample'
fname_cov = meg_path / 'sample_audvis-cov.fif'
fname_evo = meg_path / 'sample_audvis-ave.fif'
fname_trans = meg_path / 'sample_audvis_raw-trans.fif'
fname_bem = (
    subjects_dir / 'sample' / 'bem' / 'sample-5120-bem-sol.fif')

# %%
# For the volume, create a coarse source space for speed (don't do this in
# real code!), then compute the forward using this source space.

# read noise cov and and evoked
noise_cov = mne.read_cov(fname_cov)
evoked = mne.read_evokeds(fname_evo, 0)

# create a coarse source space
src_vol = mne.setup_volume_source_space(  # this is a very course resolution!
    'sample', pos=15., subjects_dir=subjects_dir,
    add_interpolator=False)  # usually you want True, this is just for speed

# compute the forward
forward_vol = mne.make_forward_solution(  # MEG-only for speed
    evoked.info, fname_trans, src_vol, fname_bem, eeg=False)
del src_vol

# %%
# Now make an inverse operator and compute the PSF at a source.
inverse_operator_vol = mne.minimum_norm.make_inverse_operator(
    info=evoked.info, forward=forward_vol, noise_cov=noise_cov)

# compute resolution matrix for sLORETA
rm_lor_vol = make_inverse_resolution_matrix(
    forward_vol, inverse_operator_vol, method='sLORETA', lambda2=1. / 9.)

# get PSF and CTF for sLORETA at one vertex
sources_vol = [100]
stc_psf_vol = get_point_spread(
    rm_lor_vol, forward_vol['src'], sources_vol, norm=True)
del rm_lor_vol

##############################################################################
# Visualize
# ---------
# PSF:

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
