"""
===============================================================================
Compute a sparse inverse solution using the Gamma-Map empirical Bayesian method
===============================================================================

See Wipf et al. "A unified Bayesian framework for MEG/EEG source imaging."
NeuroImage, vol. 44, no. 3, pp. 947?66, Mar. 2009.
"""
# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne.datasets import sample
from mne.inverse_sparse import gamma_map
from mne.viz import plot_sparse_source_estimates

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
forward = mne.read_forward_solution(fwd_fname, surf_ori=True,
                                    force_fixed=False)

# Read noise noise covariance matrix and regularize it
cov = mne.read_cov(cov_fname)
cov = mne.cov.regularize(cov, evoked.info)

# Run the Gamma-MAP method
alpha = 0.5
stc, residual = gamma_map(evoked, forward, cov, alpha, xyz_same_gamma=True,
                          return_residual=True)

# View in 2D and 3D ("glass" brain like 3D plot)

# Show the sources as spheres scaled by their strength
scale_factors = np.max(np.abs(stc.data), axis=1)
scale_factors = 0.5 * (1 + scale_factors / np.max(scale_factors))

plot_sparse_source_estimates(
    forward['src'], stc, bgcolor=(1, 1, 1),
    modes=['sphere'], opacity=0.1, scale_factors=(scale_factors, None),
    fig_name="Gamma-MAP")

# Show the evoked response and the residual for gradiometers
ylim = dict(grad=[-120, 120])
evoked.pick_types(meg='grad', exclude='bads')
evoked.plot(titles=dict(grad='Evoked Response Gradiometers'), ylim=ylim,
            proj=True)

residual.pick_types(meg='grad', exclude='bads')
residual.plot(titles=dict(grad='Residuals Gradiometers'), ylim=ylim,
              proj=True)
