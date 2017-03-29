"""
.. _point_spread:

Corrupt known signal with point spread
======================================

The aim of this tutorial is to demonstrate how to put a known signal at a
desired location(s) in a :class:`mne.SourceEstimate` and then corrupt the
signal with point-spread by applying a forward and inverse solution.
"""

import os.path as op

import numpy as np
from mayavi import mlab

import mne
from mne.datasets import sample

from mne.minimum_norm import read_inverse_operator, apply_inverse
from mne.simulation import simulate_stc, simulate_evoked

###############################################################################
# First, we set some parameters.

seed = 42

# parameters for inverse method
method = 'sLORETA'
snr = 3.
lambda2 = 1.0 / snr ** 2

# signal simulation parameters
# do not add extra noise to the known signals
evoked_snr = np.inf
T = 100
times = np.linspace(0, 1, T)
dt = times[1] - times[0]

# Paths to MEG data
data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-meg-oct-6-fwd.fif')
fname_inv = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-meg-oct-6-meg-fixed-inv.fif')

fname_evoked = op.join(data_path, 'MEG', 'sample',
                       'sample_audvis-ave.fif')

###############################################################################
# Load the MEG data
# -----------------

fwd = mne.read_forward_solution(fname_fwd, force_fixed=True,
                                surf_ori=True)
fwd['info']['bads'] = []
inv_op = read_inverse_operator(fname_inv)

raw = mne.io.RawFIF(op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_raw.fif'))
events = mne.find_events(raw)
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2}
epochs = mne.Epochs(raw, events, event_id, baseline=(None, 0), preload=True)
epochs.info['bads'] = []
evoked = epochs.average()

labels = mne.read_labels_from_annot('sample', subjects_dir=subjects_dir)
label_names = [l.name for l in labels]
n_labels = len(labels)

###############################################################################
# Estimate the background noise covariance from the baseline period
# -----------------------------------------------------------------

cov = mne.compute_covariance(epochs, tmin=None, tmax=0.)

###############################################################################
# Generate sinusoids in two spatially distant labels
# --------------------------------------------------

# The known signal is all zero-s off of the two labels of interest
signal = np.zeros((n_labels, T))
idx = label_names.index('inferiorparietal-lh')
signal[idx, :] = 1e-7 * np.sin(5 * 2 * np.pi * times)
idx = label_names.index('rostralmiddlefrontal-rh')
signal[idx, :] = 1e-7 * np.sin(7 * 2 * np.pi * times)

###############################################################################
# Find the center vertices in source space of each label
# ------------------------------------------------------
#
# We want the known signal in each label to only be active at the center. We
# create a mask for each label that is 1 at the center vertex and 0 at all
# other vertices in the label. This mask is then used when simulating
# source-space data.

hemi_to_ind = {'lh': 0, 'rh': 1}
for i, label in enumerate(labels):
    # The `center_of_mass` function needs labels to have values.
    labels[i].values.fill(1.)

    # Restrict the eligible vertices to be those on the surface under
    # consideration and within the label.
    surf_vertices = fwd['src'][hemi_to_ind[label.hemi]]['vertno']
    restrict_verts = np.intersect1d(surf_vertices, label.vertices)
    com = labels[i].center_of_mass(subject='sample',
                                   subjects_dir=subjects_dir,
                                   restrict_vertices=restrict_verts,
                                   surf='white')

    # Convert the center of vertex index from surface vertex list to Label's
    # vertex list.
    cent_idx = np.where(label.vertices == com)[0][0]

    # Create a mask with 1 at center vertex and zeros elsewhere.
    labels[i].values.fill(0.)
    labels[i].values[cent_idx] = 1.

###############################################################################
# Create source-space data with known signals
# -------------------------------------------
#
# Put known signals onto surface vertices using the array of signals and
# the label masks (stored in labels[i].values).
stc_gen = simulate_stc(fwd['src'], labels, signal, times[0], dt,
                       value_fun=lambda x: x)

###############################################################################
# Plot original signals
# ---------------------
#
# Note that the original signals are highly concentrated (point) sources.
#
kwargs = dict(subjects_dir=subjects_dir, hemi='split', smoothing_steps=4,
              time_unit='s', initial_time=0.05, size=1200,
              views=['lat', 'med'])
clim = dict(kind='value', pos_lims=[1e-9, 1e-8, 1e-7])
figs = [mlab.figure(1), mlab.figure(2), mlab.figure(3), mlab.figure(4)]
brain_gen = stc_gen.plot(clim=clim, figure=figs, **kwargs)

###############################################################################
# Simulate sensor-space signals
# -----------------------------
#
# Use the forward solution and add Gaussian noise to simulate sensor-space
# (evoked) data from the known source-space signals. The amount of noise is
# controlled by `evoked_snr` (higher values imply less noise).
#
evoked_gen = simulate_evoked(fwd, stc_gen, evoked.info, cov, evoked_snr,
                             tmin=0., tmax=1., random_state=seed)

# Map the simulated sensor-space data to source-space using the inverse
# operator.
stc_inv = apply_inverse(evoked_gen, inv_op, lambda2, method=method)

###############################################################################
# Plot the point-spread of corrupted signal
# -----------------------------------------
#
# Notice that after applying the forward- and inverse-operators to the known
# point sources that the point sources have spread across the source-space.
# This spread is due to the minimum norm solution so that the signal leaks to
# nearby vertices with similar orientations so that signal ends up crossing the
# sulci and gyri.
figs = [mlab.figure(5), mlab.figure(6), mlab.figure(7), mlab.figure(8)]
brain_inv = stc_inv.plot(figure=figs, **kwargs)

###############################################################################
# Exercises
# ---------
#    - Change the `method` parameter to either `dSPM` or `MNE` to explore the
#      effect of the inverse method.
#    - Try setting `evoked_snr` to a small, finite value, e.g. 3., to see the
#      effect of noise.
