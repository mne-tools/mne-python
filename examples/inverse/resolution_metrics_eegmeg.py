# -*- coding: utf-8 -*-
"""
.. _ex-res-metrics-meeg:

==============================================================
Compute spatial resolution metrics to compare MEG with EEG+MEG
==============================================================

Compute peak localisation error and spatial deviation for the point-spread
functions of dSPM and MNE. Plot their distributions and difference of
distributions. This example mimics some results from :footcite:`HaukEtAl2019`,
namely Figure 3 (peak localisation error for PSFs, L2-MNE vs dSPM) and Figure 4
(spatial deviation for PSFs, L2-MNE vs dSPM). It shows that combining MEG with
EEG reduces the point-spread function and increases the spatial resolution of
source imaging, especially for deeper sources.
"""
# Author: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD-3-Clause

# %%

import mne
from mne.datasets import sample
from mne.minimum_norm.resolution_matrix import make_inverse_resolution_matrix
from mne.minimum_norm.spatial_resolution import resolution_metrics

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path / 'subjects/'
meg_path = data_path / 'MEG' / 'sample'
fname_fwd_emeg = meg_path / 'sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_cov = meg_path / 'sample_audvis-cov.fif'
fname_evo = meg_path / 'sample_audvis-ave.fif'

# read forward solution with EEG and MEG
forward_emeg = mne.read_forward_solution(fname_fwd_emeg)
# forward operator with fixed source orientations
forward_emeg = mne.convert_forward_solution(forward_emeg, surf_ori=True,
                                            force_fixed=True)

# create a forward solution with MEG only
forward_meg = mne.pick_types_forward(forward_emeg, meg=True, eeg=False)

# noise covariance matrix
noise_cov = mne.read_cov(fname_cov)

# evoked data for info
evoked = mne.read_evokeds(fname_evo, 0)

# make inverse operator from forward solution for MEG and EEGMEG
inv_emeg = mne.minimum_norm.make_inverse_operator(
    info=evoked.info, forward=forward_emeg, noise_cov=noise_cov, loose=0.,
    depth=None)

inv_meg = mne.minimum_norm.make_inverse_operator(
    info=evoked.info, forward=forward_meg, noise_cov=noise_cov, loose=0.,
    depth=None)

# regularisation parameter
snr = 3.0
lambda2 = 1.0 / snr ** 2

# %%
# EEGMEG
# ------
# Compute resolution matrices, localization error, and spatial deviations
# for MNE:

rm_emeg = make_inverse_resolution_matrix(forward_emeg, inv_emeg,
                                         method='MNE', lambda2=lambda2)
ple_psf_emeg = resolution_metrics(rm_emeg, inv_emeg['src'],
                                  function='psf', metric='peak_err')
sd_psf_emeg = resolution_metrics(rm_emeg, inv_emeg['src'],
                                 function='psf', metric='sd_ext')
del rm_emeg

# %%
# MEG
# ---
# Do the same for MEG:

rm_meg = make_inverse_resolution_matrix(forward_meg, inv_meg,
                                        method='MNE', lambda2=lambda2)
ple_psf_meg = resolution_metrics(rm_meg, inv_meg['src'],
                                 function='psf', metric='peak_err')
sd_psf_meg = resolution_metrics(rm_meg, inv_meg['src'],
                                function='psf', metric='sd_ext')
del rm_meg

# %%
# Visualization
# -------------
# Look at peak localisation error (PLE) across the whole cortex for PSF:

brain_ple_emeg = ple_psf_emeg.plot('sample', 'inflated', 'lh',
                                   subjects_dir=subjects_dir, figure=1,
                                   clim=dict(kind='value', lims=(0, 2, 4)))

brain_ple_emeg.add_text(0.1, 0.9, 'PLE PSF EMEG', 'title', font_size=16)

# %%
# For MEG only:

brain_ple_meg = ple_psf_meg.plot('sample', 'inflated', 'lh',
                                 subjects_dir=subjects_dir, figure=2,
                                 clim=dict(kind='value', lims=(0, 2, 4)))

brain_ple_meg.add_text(0.1, 0.9, 'PLE PSF MEG', 'title', font_size=16)

# %%
# Subtract the two distributions and plot this difference:

diff_ple = ple_psf_emeg - ple_psf_meg

brain_ple_diff = diff_ple.plot('sample', 'inflated', 'lh',
                               subjects_dir=subjects_dir, figure=3,
                               clim=dict(kind='value', pos_lims=(0., .5, 1.)),
                               smoothing_steps=20)

brain_ple_diff.add_text(0.1, 0.9, 'PLE EMEG-MEG', 'title', font_size=16)

# %%
# These plots show that with respect to peak localization error, adding EEG to
# MEG does not bring much benefit. Next let's visualise spatial deviation (SD)
# across the whole cortex for PSF:

brain_sd_emeg = sd_psf_emeg.plot('sample', 'inflated', 'lh',
                                 subjects_dir=subjects_dir, figure=4,
                                 clim=dict(kind='value', lims=(0, 2, 4)))

brain_sd_emeg.add_text(0.1, 0.9, 'SD PSF EMEG', 'title', font_size=16)

# %%
# For MEG only:

brain_sd_meg = sd_psf_meg.plot('sample', 'inflated', 'lh',
                               subjects_dir=subjects_dir, figure=5,
                               clim=dict(kind='value', lims=(0, 2, 4)))

brain_sd_meg.add_text(0.1, 0.9, 'SD PSF MEG', 'title', font_size=16)

# %%
# Subtract the two distributions and plot this difference:

diff_sd = sd_psf_emeg - sd_psf_meg

brain_sd_diff = diff_sd.plot('sample', 'inflated', 'lh',
                             subjects_dir=subjects_dir, figure=6,
                             clim=dict(kind='value', pos_lims=(0., .5, 1.)),
                             smoothing_steps=20)

brain_sd_diff.add_text(0.1, 0.9, 'SD EMEG-MEG', 'title', font_size=16)

# %%
# Adding EEG to MEG decreases the spatial extent of point-spread
# functions (lower spatial deviation, blue colors), thus increasing
# resolution, especially for deeper source locations.
#
# References
# ----------
# .. footbibliography::
