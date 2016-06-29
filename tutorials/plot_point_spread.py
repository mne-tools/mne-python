"""
.. _point_spread:

Corrupt known signal with point spread
======================================

The aim of this tutorial is to demonstrate how to put a known signal at a
desired location(s) in a :class:`SourceEstimate` and then corrupt the signal
with point-spread by applying a forward and inverse solution.
"""

import os.path as op

import numpy as np

import mne
from mne.datasets import sample

from mne.minimum_norm import read_inverse_operator, apply_inverse
from mne.simulation import simulate_stc, simulate_evoked

###############################################################################
# Parameters

seed = 42

# regularization parameter for inverse method
method = 'sLORETA'
snr = 3.
lambda2 = 1.0 / snr ** 2

# signal simulation parameters
# do not add extra noise to our know signals
evoked_snr = np.inf
T = 100
times = np.linspace(0, 1, T)
dt = times[1] - times[0]

# Paths to MEG data
data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects/')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-meg-oct-6-fwd.fif')
fname_inv = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-meg-oct-6-meg-fixed-inv.fif')

fname_evoked = op.join(data_path, 'MEG', 'sample',
                       'sample_audvis-ave.fif')

subject_dir = op.join(data_path, 'subjects/')

###############################################################################
# Load and process MEG data
fwd = mne.read_forward_solution(fname_fwd, force_fixed=True,
                                surf_ori=True)
inv_op = read_inverse_operator(fname_inv)

raw = mne.io.RawFIF(op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_raw.fif'))
events = mne.find_events(raw)
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2}
epochs = mne.Epochs(raw, events, event_id,
                    baseline=(None, 0),
                    preload=True)
evoked = epochs.average()

labels = mne.read_labels_from_annot('sample', subjects_dir=subject_dir)
label_names = [l.name for l in labels]
n_labels = len(labels)

###############################################################################
# Estimate noise covariance from baseline
cov = mne.compute_covariance(epochs, tmin=None, tmax=0.)

###############################################################################
# Simulate known signals at two regions

# Generate a known signal to corrupt with point spread
idx = label_names.index('inferiorparietal-lh')
signal = np.zeros((n_labels, T))
signal[idx, :] = 1e-7 * np.sin(5 * 2 * np.pi * times)
idx = label_names.index('rostralmiddlefrontal-lh')
signal[idx, :] = 1e-7 * np.sin(7 * 2 * np.pi * times)

###############################################################################
# Create mask for each label so that the signal is only set at the center
# vertex.
hemi_to_ind = {'lh': 0, 'rh': 1}
for i, label in enumerate(labels):
    # Labels need values to use center_of_mass function
    labels[i].values.fill(1.)
    surf_vertices = fwd['src'][hemi_to_ind[label.hemi]]['vertno']
    restrict_verts = np.intersect1d(surf_vertices, label.vertices)
    com = labels[i].center_of_mass(subject='sample',
                                   subjects_dir=subject_dir,
                                   restrict_vertices=restrict_verts,
                                   surf='white')
    # Get center of vertex index in Label's vertex list
    cent_idx = np.where(label.vertices == com)[0][0]

    # Mask out all vertices except the center.
    labels[i].values.fill(0.)
    labels[i].values[cent_idx] = 1.

###############################################################################
# Create SourceEstimate object with known signal at center vertices of two
# labels.
stc_gen = simulate_stc(fwd['src'], labels, signal, times[0], dt,
                       value_fun=lambda x: x)

# Use forward solution to generate sensor space data
evoked_gen = simulate_evoked(fwd, stc_gen, evoked.info, cov, evoked_snr,
                             tmin=0., tmax=1., random_state=seed)

# Apply inverse to project sensor space signal back into source space
stc_inv = apply_inverse(evoked_gen, inv_op, lambda2, method=method)

###############################################################################
# Plot original signals
lims = (0., np.mean(stc_gen.data[stc_gen.data > 0]), np.max(stc_gen.data))
tlabel = 'Initial point-sources'
brain_gen = stc_gen.copy().crop(0.05, None).plot(subjects_dir=subjects_dir,
                                                 hemi='lh',
                                                 surface='inflated',
                                                 clim=dict(kind='value',
                                                           pos_lims=lims),
                                                 time_label=tlabel,
                                                 figure=1)

# Plot point-spread of corrupted signal
tlabel = 'Corrupted with point-spread'
brain_inv = stc_inv.copy().crop(0.05, None).plot(subjects_dir=subjects_dir,
                                                 hemi='lh',
                                                 surface='inflated',
                                                 time_label=tlabel,
                                                 figure=2)

###############################################################################
# Exercises
# ---------
#    - Change the `method` parameter to either `dSPM` or `MNE` to explore the
#      effect of the inverse method.
#    - Try setting `noise_snr` to a small, finite value, e.g. 3., to see the
#      effect of noise.
