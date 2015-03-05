"""
=========================================
Compute Rap-Music on evoked data
=========================================

Compute a Recursively Applied and Projected MUltiple Signal Classification
(RAP-MUSIC) on evoked dataset.

The reference for Rap-Music is:
J.C. Mosher and R.M. Leahy. 1999. Source localization using recursively
applied and projected (RAP) MUSIC. Trans. Sig. Proc. 47, 2
(February 1999), 332-340.
DOI=10.1109/78.740118 http://dx.doi.org/10.1109/78.740118
"""

# Author: Yousra Bekhti <yousra.bekhti@gmail.com>
#
# License: BSD (3-clause)

import mne

from mayavi import mlab

from mne.datasets import sample
from mne.beamformer import rap_music
from mne.io.pick import pick_types_evoked


data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

# Read the evoked response and crop it
condition = 'Left Auditory'
evoked = mne.read_evokeds(evoked_fname, condition=condition,
                          baseline=(None, 0))
evoked.crop(tmin=-50e-3, tmax=300e-3)

evoked = pick_types_evoked(evoked, meg=True, eeg=False)

# Read the forward solution
forward = mne.read_forward_solution(fwd_fname, surf_ori=True,
                                    force_fixed=False)

# Read noise covariance matrix and regularize it
noise_cov = mne.read_cov(cov_fname)

dipole, residual = rap_music(evoked, forward, noise_cov, n_sources=2,
                             return_residual=True, verbose=True)

# Plot the evoked data and the residual.
evoked.plot()
residual.plot()

src = forward['src']
pos, ori = dipole['pos'], dipole['ori']
# Add a viz function for dipoles instead of this
lh_points = src[0]['rr']
lh_faces = src[0]['use_tris']
mlab.figure(size=(600, 600), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

# show one cortical surface
mlab.triangular_mesh(lh_points[:, 0], lh_points[:, 1], lh_points[:, 2],
                     lh_faces, color=(0.7, ) * 3)

rh_points = src[1]['rr']
rh_faces = src[1]['use_tris']

# show one cortical surface
mlab.triangular_mesh(rh_points[:, 0], rh_points[:, 1], rh_points[:, 2],
                     rh_faces, color=(0.7, ) * 3)

# show dipole as small cones
mlab.quiver3d(pos[:, 0], pos[:, 1], pos[:, 2],
              ori[:, 0], ori[:, 1], ori[:, 2],
              opacity=1., mode='cone')

# # With fixed orientation
# forward_fixed = mne.convert_forward_solution(forward, force_fixed=True)
# dipole_fixed, residual_fixed = rap_music(evoked, forward_fixed, noise_cov,
#                                          n_sources=2, return_residual=True)

# residual_fixed.plot()

# mne.viz.tight_layout()
# plt.show()
