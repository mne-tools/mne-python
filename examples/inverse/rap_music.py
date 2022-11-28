# -*- coding: utf-8 -*-
"""
.. _ex-rap-music:

================================
Compute Rap-Music on evoked data
================================

Compute a Recursively Applied and Projected MUltiple Signal Classification
(RAP-MUSIC) :footcite:`MosherLeahy1999` on evoked data.
"""

# Author: Yousra Bekhti <yousra.bekhti@gmail.com>
#
# License: BSD-3-Clause

# %%

import mne

from mne.datasets import sample
from mne.beamformer import rap_music
from mne.viz import plot_dipole_locations, plot_dipole_amplitudes

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path / 'subjects'
meg_path = data_path / 'MEG' / 'sample'
fwd_fname = meg_path / 'sample_audvis-meg-eeg-oct-6-fwd.fif'
evoked_fname = meg_path / 'sample_audvis-ave.fif'
cov_fname = meg_path / 'sample_audvis-cov.fif'

# Read the evoked response and crop it
condition = 'Right Auditory'
evoked = mne.read_evokeds(evoked_fname, condition=condition,
                          baseline=(None, 0))
# select N100
evoked.crop(tmin=0.05, tmax=0.15)

evoked.pick_types(meg=True, eeg=False)

# Read the forward solution
forward = mne.read_forward_solution(fwd_fname)

# Read noise covariance matrix
noise_cov = mne.read_cov(cov_fname)

dipoles, residual = rap_music(evoked, forward, noise_cov, n_dipoles=2,
                              return_residual=True, verbose=True)
trans = forward['mri_head_t']
plot_dipole_locations(dipoles, trans, 'sample', subjects_dir=subjects_dir)
plot_dipole_amplitudes(dipoles)

# Plot the evoked data and the residual.
evoked.plot(ylim=dict(grad=[-300, 300], mag=[-800, 800], eeg=[-6, 8]),
            time_unit='s')
residual.plot(ylim=dict(grad=[-300, 300], mag=[-800, 800], eeg=[-6, 8]),
              time_unit='s')

# %%
# References
# ----------
# .. footbibliography::
