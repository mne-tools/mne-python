"""
================================================================
Compute sparse inverse solution with mixed norm: MxNE and irMxNE
================================================================

Runs an (ir)MxNE (L1/L2 :footcite:`GramfortEtAl2012` or L0.5/L2
:footcite:`StrohmeierEtAl2014` mixed norm) inverse solver.
L0.5/L2 is done with irMxNE which allows for sparser source estimates with less
amplitude bias due to the non-convexity of the L0.5/L2 mixed norm penalty.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD (3-clause)

# sphinx_gallery_thumbnail_number = 2

import numpy as np

import mne
from mne.datasets import sample
from mne.inverse_sparse import mixed_norm, make_stc_from_dipoles
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.viz import (plot_sparse_source_estimates,
                     plot_dipole_locations, plot_dipole_amplitudes)

print(__doc__)

data_path = sample.data_path()
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'
subjects_dir = data_path + '/subjects'

# Read noise covariance matrix
cov = mne.read_cov(cov_fname)
# Handling average file
condition = 'Left Auditory'
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
evoked.crop(tmin=0, tmax=0.3)
# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)

###############################################################################
# Run solver
alpha = 55  # regularization parameter between 0 and 100 (100 is high)
loose, depth = 0.2, 0.9  # loose orientation & depth weighting
n_mxne_iter = 10  # if > 1 use L0.5/L2 reweighted mixed norm solver
# if n_mxne_iter > 1 dSPM weighting can be avoided.

# Compute dSPM solution to be used as weights in MxNE
inverse_operator = make_inverse_operator(evoked.info, forward, cov,
                                         depth=depth, fixed=True,
                                         use_cps=True)
stc_dspm = apply_inverse(evoked, inverse_operator, lambda2=1. / 9.,
                         method='dSPM')

# Compute (ir)MxNE inverse solution with dipole output
dipoles, residual = mixed_norm(
    evoked, forward, cov, alpha, loose=loose, depth=depth, maxit=3000,
    tol=1e-4, active_set_size=10, debias=True, weights=stc_dspm,
    weights_min=8., n_mxne_iter=n_mxne_iter, return_residual=True,
    return_as_dipoles=True, verbose=True)

t = 0.083
tidx = evoked.time_as_index(t)
for di, dip in enumerate(dipoles, 1):
    print(f'Dipole #{di} GOF at {1000 * t:0.1f} ms: '
          f'{float(dip.gof[tidx]):0.1f}%')

###############################################################################
# Plot dipole activations
plot_dipole_amplitudes(dipoles)

# Plot dipole location of the strongest dipole with MRI slices
idx = np.argmax([np.max(np.abs(dip.amplitude)) for dip in dipoles])
plot_dipole_locations(dipoles[idx], forward['mri_head_t'], 'sample',
                      subjects_dir=subjects_dir, mode='orthoview',
                      idx='amplitude')

# Plot dipole locations of all dipoles with MRI slices
for dip in dipoles:
    plot_dipole_locations(dip, forward['mri_head_t'], 'sample',
                          subjects_dir=subjects_dir, mode='orthoview',
                          idx='amplitude')

###############################################################################
# Plot residual
ylim = dict(eeg=[-10, 10], grad=[-400, 400], mag=[-600, 600])
evoked.pick_types(meg=True, eeg=True, exclude='bads')
evoked.plot(ylim=ylim, proj=True, time_unit='s')
residual.pick_types(meg=True, eeg=True, exclude='bads')
residual.plot(ylim=ylim, proj=True, time_unit='s')

###############################################################################
# Generate stc from dipoles
stc = make_stc_from_dipoles(dipoles, forward['src'])

###############################################################################
# View in 2D and 3D ("glass" brain like 3D plot)
solver = "MxNE" if n_mxne_iter == 1 else "irMxNE"
plot_sparse_source_estimates(forward['src'], stc, bgcolor=(1, 1, 1),
                             fig_name="%s (cond %s)" % (solver, condition),
                             opacity=0.1)

###############################################################################
# Morph onto fsaverage brain and view
morph = mne.compute_source_morph(stc, subject_from='sample',
                                 subject_to='fsaverage', spacing=None,
                                 sparse=True, subjects_dir=subjects_dir)
stc_fsaverage = morph.apply(stc)
src_fsaverage_fname = subjects_dir + '/fsaverage/bem/fsaverage-ico-5-src.fif'
src_fsaverage = mne.read_source_spaces(src_fsaverage_fname)

plot_sparse_source_estimates(src_fsaverage, stc_fsaverage, bgcolor=(1, 1, 1),
                             fig_name="Morphed %s (cond %s)" % (solver,
                             condition), opacity=0.1)

###############################################################################
# References
# ----------
# .. footbibliography::
