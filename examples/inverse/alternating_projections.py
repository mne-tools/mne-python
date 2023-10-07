# -*- coding: utf-8 -*-
"""
.. _ex-alternating-projections:

==============================================
Compute Alternating Projections on evoked data
==============================================

Compute an Alternating Projections (AP) on :class:`~mne.Evoked` data,
on both free-oriented dipoles and fixed-oriented dipoles.

The AP method addresses the problem of multiple dipole localization
in MEG/EEG with a sequential and iterative solution,
based on minimizing the least-squares (LS) criterion.
:footcite:t:`AdlerEtAl2019,AdlerEtAl2022`
"""

# Author: Yuval Realpe <yuval.realpe@gmail.com>
#
# License: BSD-3-Clause

# %%

import mne

from mne.datasets import sample
from mne.beamformer import alternating_projections
from mne.viz import plot_dipole_locations, plot_dipole_amplitudes

print(__doc__)


# %% Setup paths

data_path = sample.data_path()
subjects_dir = data_path / "subjects"
meg_path = data_path / "MEG" / "sample"
fwd_fname = meg_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"
evoked_fname = meg_path / "sample_audvis-ave.fif"
cov_fname = meg_path / "sample_audvis-cov.fif"

# %% Load data

# Auditory samples, such as the one used on this example,
# are characterized by 2 symmetrically opposed activation zones,
# one on each lobe.
# Thus we will be looking for 2 sources,
# representing each of the activations occurring throughout the sample.
# The extent to which the estimated dipoles
# (and their estimated orientations) are able to explain the evoked data
# is represented by the var_exp variable.


# %%
# Read the evoked response and crop it
condition = "Right Auditory"
evoked = mne.read_evokeds(evoked_fname, condition=condition, baseline=(None, 0))
# select N100
evoked.crop(tmin=0.05, tmax=0.15)

evoked.pick_types(meg=True, eeg=False)

# Read the forward solution
forward = mne.read_forward_solution(fwd_fname)

# Read noise covariance matrix
noise_cov = mne.read_cov(cov_fname)

# %% Applied on free-oriented dipoles

dipoles, residual, _, var_exp = alternating_projections(
    evoked,
    forward,
    n_sources=2,
    noise_cov=noise_cov,
    return_residual=True,
    verbose=True,
)

trans = forward["mri_head_t"]
plot_dipole_locations(dipoles, trans, "sample", subjects_dir=subjects_dir)
plot_dipole_amplitudes(dipoles)

# Plot the evoked data and the residual.
ylim = dict(grad=[-300, 300], mag=[-800, 800])
evoked.plot(ylim=ylim)
residual.plot(ylim=ylim)


# %% Applied on fixed-oriented dipoles

forward = mne.convert_forward_solution(forward, force_fixed=True)
dipoles, residual, _, var_exp = alternating_projections(
    evoked,
    forward,
    n_sources=2,
    noise_cov=noise_cov,
    return_residual=True,
    verbose=True,
)

trans = forward["mri_head_t"]
plot_dipole_locations(dipoles, trans, "sample", subjects_dir=subjects_dir)
plot_dipole_amplitudes(dipoles)

# Plot the evoked data and the residual.
evoked.plot(ylim=dict(grad=[-300, 300], mag=[-800, 800], eeg=[-6, 8]), time_unit="s")
residual.plot(ylim=dict(grad=[-300, 300], mag=[-800, 800], eeg=[-6, 8]), time_unit="s")

# %%
# References
# ----------
# .. footbibliography::
