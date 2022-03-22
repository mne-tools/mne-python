# -*- coding: utf-8 -*-
"""
=================================================================
Computing source timecourses with an XFit-like multi-dipole model
=================================================================

MEGIN's XFit program offers a "guided ECD modeling" interface, where multiple
dipoles can be fitted by manually selecting subsets of sensors and time ranges.
Then, source timecourses can be computed using a multi-dipole model. The
advantage of using a multi-dipole model over fitting each dipole in isolation,
is that when a source is picked up by more than one dipole, a multi-dipole can
prevent spatial leakage. This example shows how to build a multi-dipole model
for estimating source timecourses for evokeds or single epochs.

The XFit program is the recommended approach for obtaining dipole fits because
it offers a convenient interface for it. These dipoles can then be imported
into MNE-Python by using the :func:`mne.read_dipole` function for building and
applying the multi-dipole model. However, this example will also demonstrate
how to perform guided ECD modeling using only MNE-Python functionality.
"""
# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD-3-Clause

###############################################################################
# Importing everything and setting up the data paths for the MNE-Sample
# dataset.
import mne
from mne.datasets import sample
from mne.channels import read_vectorview_selection
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              apply_inverse_epochs)
import matplotlib.pyplot as plt
import numpy as np

data_path = sample.data_path()
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_raw.fif'
cov_fname = meg_path / 'sample_audvis-shrunk-cov.fif'
bem_dir = data_path / 'subjects' / 'sample' / 'bem'
bem_fname =  bem_dir / 'sample-5120-5120-5120-bem-sol.fif'

###############################################################################
# Read the MEG data from the audvis experiment. Make epochs and evokeds for the
# left and right auditory conditions.
raw = mne.io.read_raw_fif(raw_fname)
raw = raw.pick_types(meg=True, eog=True, stim=True)
info = raw.info

# Create epochs for auditory events
events = mne.find_events(raw)
event_id = dict(right=1, left=2)
epochs = mne.Epochs(raw, events, event_id,
                    tmin=-0.1, tmax=0.3, baseline=(None, 0),
                    reject=dict(mag=4e-12, grad=4000e-13, eog=150e-6))

# Create evokeds for left and right auditory stimulation
evoked_left = epochs['left'].average()
evoked_right = epochs['right'].average()

###############################################################################
# Guided dipole modeling, meaning fitting dipoles to a manually selected subset
# of sensors as a manually chosen time, can now be performed in MEGINs XFit on
# the evokeds we computed above. However, it is possible to do it completely
# in MNE-Python.

# Setup conductor model
cov = mne.read_cov(cov_fname)
bem = mne.read_bem_solution(bem_fname)

# Fit two dipoles at t=80ms. The first dipole is fitted using only the sensors
# on the right side of the helmet. The second dipole is fitted using only the
# sensors on the left side of the helmet.
picks_left = read_vectorview_selection('Left', info=info)
evoked_fit_left = evoked_left.copy().crop(0.08, 0.08)
evoked_fit_left.pick_channels(picks_left)
dip_left, _ = mne.fit_dipole(evoked_fit_left, cov, bem)

picks_right = read_vectorview_selection('Right', info=info)
evoked_fit_right = evoked_right.copy().crop(0.08, 0.08)
evoked_fit_right.pick_channels(picks_right)
dip_right, _ = mne.fit_dipole(evoked_fit_right, cov, bem)

###############################################################################
# Now that we have the location and orientations of the dipoles, compute the
# full timecourses using MNE, assigning activity to both dipoles at the same
# time while preventing leakage between the two. We use a very low ``lambda``
# value to ensure both dipoles are fully used.

# Collect the dipoles in a single `Dipole` object and compute a forward model
dips = mne.concatenate_dipoles([dip_left, dip_right])
fwd, _ = mne.make_forward_dipole(dips, bem, info)

# Apply MNE inverse
inv = make_inverse_operator(info, fwd, cov, fixed=True, depth=0)
stc_left = apply_inverse(evoked_left, inv, method='MNE', lambda2=1E-6)
stc_right = apply_inverse(evoked_right, inv, method='MNE', lambda2=1E-6)

# Plot the timecourses of the resulting source estimate
fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
axes[0].plot(stc_left.times, stc_left.data.T)
axes[0].set_title('Left auditory stimulation')
axes[0].legend(['Dipole 1', 'Dipole 2'])
axes[1].plot(stc_right.times, stc_right.data.T)
axes[1].set_title('Right auditory stimulation')
axes[1].set_xlabel('Time (s)')
fig.supylabel('Dipole amplitude')

# We can also fit the timecourses to single epochs. Here, we do it for each
# experimental condition separately.
stcs_left = apply_inverse_epochs(epochs['left'], inv, lambda2=1E-6,
                                 method='MNE')
stcs_right = apply_inverse_epochs(epochs['right'], inv, lambda2=1E-6,
                                  method='MNE')

###############################################################################
# To summarize and visualize the single-epoch dipole amplitudes, we will create
# a detailed plot of the mean amplitude of the dipoles during different
# experimental conditions.

# Summarize the single epoch timecourses by computing the mean amplitude from
# 60-90ms.
amplitudes_left = []
amplitudes_right = []
for stc in stcs_left:
    amplitudes_left.append(stc.crop(0.06, 0.09).mean().data)
for stc in stcs_right:
    amplitudes_right.append(stc.crop(0.06, 0.09).mean().data)
amplitudes = np.vstack([amplitudes_left, amplitudes_right])

# Visualize the epoch-by-epoch dipole ampltudes in a detailed figure.
n = len(amplitudes)
n_left = len(amplitudes_left)
mean_left = np.mean(amplitudes_left, axis=0)
mean_right = np.mean(amplitudes_right, axis=0)

fig = plt.figure()
plt.scatter(np.arange(n), amplitudes[:, 0], label='Dipole 1')
plt.scatter(np.arange(n), amplitudes[:, 1], label='Dipole 2')
transition_point = n_left - 0.5
plt.plot([0, transition_point], [mean_left[0], mean_left[0]], color='C0')
plt.plot([0, transition_point], [mean_left[1], mean_left[1]], color='C1')
plt.plot([transition_point, n], [mean_right[0], mean_right[0]], color='C0')
plt.plot([transition_point, n], [mean_right[1], mean_right[1]], color='C1')
plt.axvline(transition_point, color='black')
plt.xlabel('Epochs')
plt.ylabel('Dipole amplitude')
plt.suptitle('Single epoch dipole amplitudes')
fig.text(0.30, 0.9, 'Left auditory stimulation', ha='center')
fig.text(0.70, 0.9, 'Right auditory stimulation', ha='center')
plt.legend()
