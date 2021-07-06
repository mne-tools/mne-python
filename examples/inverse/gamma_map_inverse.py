"""
===============================================================================
Compute a sparse inverse solution using the Gamma-MAP empirical Bayesian method
===============================================================================

See :footcite:`WipfNagarajan2009` for details.
"""
# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#         Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne.datasets import sample
from mne.inverse_sparse import gamma_map, make_stc_from_dipoles
from mne.viz import (plot_sparse_source_estimates,
                     plot_dipole_locations, plot_dipole_amplitudes)

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

# Read the evoked response and crop it
condition = 'Left visual'
evoked = mne.read_evokeds(evoked_fname, condition=condition,
                          baseline=(None, 0))
evoked.crop(tmin=-50e-3, tmax=300e-3)

# Read the forward solution
forward = mne.read_forward_solution(fwd_fname)

# Read noise noise covariance matrix and regularize it
cov = mne.read_cov(cov_fname)
cov = mne.cov.regularize(cov, evoked.info, rank=None)

# Run the Gamma-MAP method with dipole output
alpha = 0.5
dipoles, residual = gamma_map(
    evoked, forward, cov, alpha, xyz_same_gamma=True, return_residual=True,
    return_as_dipoles=True)

###############################################################################
# Plot dipole activations
plot_dipole_amplitudes(dipoles)

# Plot dipole location of the strongest dipole with MRI slices
idx = np.argmax([np.max(np.abs(dip.amplitude)) for dip in dipoles])
plot_dipole_locations(dipoles[idx], forward['mri_head_t'], 'sample',
                      subjects_dir=subjects_dir, mode='orthoview',
                      idx='amplitude')

# # Plot dipole locations of all dipoles with MRI slices
# for dip in dipoles:
#     plot_dipole_locations(dip, forward['mri_head_t'], 'sample',
#                           subjects_dir=subjects_dir, mode='orthoview',
#                           idx='amplitude')

###############################################################################
# Show the evoked response and the residual for gradiometers
ylim = dict(grad=[-120, 120])
evoked.pick_types(meg='grad', exclude='bads')
evoked.plot(titles=dict(grad='Evoked Response Gradiometers'), ylim=ylim,
            proj=True, time_unit='s')

residual.pick_types(meg='grad', exclude='bads')
residual.plot(titles=dict(grad='Residuals Gradiometers'), ylim=ylim,
              proj=True, time_unit='s')

###############################################################################
# Generate stc from dipoles
stc = make_stc_from_dipoles(dipoles, forward['src'])

###############################################################################
# View in 2D and 3D ("glass" brain like 3D plot)
# Show the sources as spheres scaled by their strength
scale_factors = np.max(np.abs(stc.data), axis=1)
scale_factors = 0.5 * (1 + scale_factors / np.max(scale_factors))

plot_sparse_source_estimates(
    forward['src'], stc, bgcolor=(1, 1, 1),
    modes=['sphere'], opacity=0.1, scale_factors=(scale_factors, None),
    fig_name="Gamma-MAP")

###############################################################################
# References
# ----------
# .. footbibliography::
