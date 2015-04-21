"""
================================
Compute Rap-Music on evoked data
================================

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

from mne.datasets import sample
from mne.beamformer import rap_music
from mne.viz import plot_dipole_locations, plot_dipole_amplitudes

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

# Read the evoked response and crop it
condition = 'Right Auditory'
evoked = mne.read_evokeds(evoked_fname, condition=condition,
                          baseline=(None, 0))
evoked.crop(tmin=0.05, tmax=0.15)  # select N100

evoked.pick_types(meg=True, eeg=False)

# Read the forward solution
forward = mne.read_forward_solution(fwd_fname, surf_ori=True,
                                    force_fixed=False)

# Read noise covariance matrix
noise_cov = mne.read_cov(cov_fname)

dipoles, residual = rap_music(evoked, forward, noise_cov, n_dipoles=2,
                              return_residual=True, verbose=True)
trans = forward['mri_head_t']
plot_dipole_locations(dipoles, trans, 'sample', subjects_dir=subjects_dir)
plot_dipole_amplitudes(dipoles)

# Plot the evoked data and the residual.
evoked.plot(ylim=dict(grad=[-300, 300], mag=[-800, 800], eeg=[-6, 8]))
residual.plot(ylim=dict(grad=[-300, 300], mag=[-800, 800], eeg=[-6, 8]))
