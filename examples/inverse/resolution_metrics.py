# -*- coding: utf-8 -*-
"""
.. _ex-res-metrics:

==================================================
Compute spatial resolution metrics in source space
==================================================

Compute peak localisation error and spatial deviation for the point-spread
functions of dSPM and MNE. Plot their distributions and difference of
distributions. This example mimics some results from :footcite:`HaukEtAl2019`,
namely Figure 3 (peak localisation error for PSFs, L2-MNE vs dSPM) and Figure 4
(spatial deviation for PSFs, L2-MNE vs dSPM).
"""
# Author: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD-3-Clause

# %%

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_resolution_matrix
from mne.minimum_norm import resolution_metrics

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path / 'subjects'
meg_path = data_path / 'MEG' / 'sample'
fname_fwd = meg_path / 'sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_cov = meg_path / 'sample_audvis-cov.fif'
fname_evo = meg_path / 'sample_audvis-ave.fif'

# read forward solution
forward = mne.read_forward_solution(fname_fwd)
# forward operator with fixed source orientations
mne.convert_forward_solution(forward, surf_ori=True,
                             force_fixed=True, copy=False)

# noise covariance matrix
noise_cov = mne.read_cov(fname_cov)

# evoked data for info
evoked = mne.read_evokeds(fname_evo, 0)

# make inverse operator from forward solution
# free source orientation
inverse_operator = mne.minimum_norm.make_inverse_operator(
    info=evoked.info, forward=forward, noise_cov=noise_cov, loose=0.,
    depth=None)

# regularisation parameter
snr = 3.0
lambda2 = 1.0 / snr ** 2

# %%
# MNE
# ---
# Compute resolution matrices, peak localisation error (PLE) for point spread
# functions (PSFs), spatial deviation (SD) for PSFs:

rm_mne = make_inverse_resolution_matrix(forward, inverse_operator,
                                        method='MNE', lambda2=lambda2)
ple_mne_psf = resolution_metrics(rm_mne, inverse_operator['src'],
                                 function='psf', metric='peak_err')
sd_mne_psf = resolution_metrics(rm_mne, inverse_operator['src'],
                                function='psf', metric='sd_ext')
del rm_mne

# %%
# dSPM
# ----
# Do the same for dSPM:

rm_dspm = make_inverse_resolution_matrix(forward, inverse_operator,
                                         method='dSPM', lambda2=lambda2)
ple_dspm_psf = resolution_metrics(rm_dspm, inverse_operator['src'],
                                  function='psf', metric='peak_err')
sd_dspm_psf = resolution_metrics(rm_dspm, inverse_operator['src'],
                                 function='psf', metric='sd_ext')
del rm_dspm, forward

# %%
# Visualize results
# -----------------
# Visualise peak localisation error (PLE) across the whole cortex for MNE PSF:
brain_ple_mne = ple_mne_psf.plot('sample', 'inflated', 'lh',
                                 subjects_dir=subjects_dir, figure=1,
                                 clim=dict(kind='value', lims=(0, 2, 4)))
brain_ple_mne.add_text(0.1, 0.9, 'PLE MNE', 'title', font_size=16)

# %%
# And dSPM:

brain_ple_dspm = ple_dspm_psf.plot('sample', 'inflated', 'lh',
                                   subjects_dir=subjects_dir, figure=2,
                                   clim=dict(kind='value', lims=(0, 2, 4)))
brain_ple_dspm.add_text(0.1, 0.9, 'PLE dSPM', 'title', font_size=16)

# %%
# Subtract the two distributions and plot this difference
diff_ple = ple_mne_psf - ple_dspm_psf

brain_ple_diff = diff_ple.plot('sample', 'inflated', 'lh',
                               subjects_dir=subjects_dir, figure=3,
                               clim=dict(kind='value', pos_lims=(0., 1., 2.)))
brain_ple_diff.add_text(0.1, 0.9, 'PLE MNE-dSPM', 'title', font_size=16)

# %%
# These plots show that  dSPM has generally lower peak localization error (red
# color) than MNE in deeper brain areas, but higher error (blue color) in more
# superficial areas.
#
# Next we'll visualise spatial deviation (SD) across the whole cortex for MNE
# PSF:

brain_sd_mne = sd_mne_psf.plot('sample', 'inflated', 'lh',
                               subjects_dir=subjects_dir, figure=4,
                               clim=dict(kind='value', lims=(0, 2, 4)))
brain_sd_mne.add_text(0.1, 0.9, 'SD MNE', 'title', font_size=16)

# %%
# And dSPM:

brain_sd_dspm = sd_dspm_psf.plot('sample', 'inflated', 'lh',
                                 subjects_dir=subjects_dir, figure=5,
                                 clim=dict(kind='value', lims=(0, 2, 4)))
brain_sd_dspm.add_text(0.1, 0.9, 'SD dSPM', 'title', font_size=16)

# %%
# Subtract the two distributions and plot this difference:

diff_sd = sd_mne_psf - sd_dspm_psf

brain_sd_diff = diff_sd.plot('sample', 'inflated', 'lh',
                             subjects_dir=subjects_dir, figure=6,
                             clim=dict(kind='value', pos_lims=(0., 1., 2.)))
brain_sd_diff.add_text(0.1, 0.9, 'SD MNE-dSPM', 'title', font_size=16)

# %%
# These plots show that dSPM has generally higher spatial deviation than MNE
# (blue color), i.e. worse performance to distinguish different sources.
#
# References
# ----------
# .. footbibliography::
