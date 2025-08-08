"""
.. _ex-multi-dipole:

======================
Guided dipole modeling
======================

Inspired by MEGIN's XFit program, MNE-Python offers a guided dipole modeling interface.
Through this interface, you can fit dipoles at specific timings using selected subsets
of sensors to gradually build up a multi dipole source estimate. This is the manual
alternative to using automated algorithms such as TRAP-MUSIC and MxNE.
"""
# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

########################################################################################
# Read the MEG data from the audvis experiment. Make epochs for the left auditory
# condition.
import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.datasets import sample

data_path = sample.data_path()
meg_path = data_path / "MEG" / "sample"
raw_fname = meg_path / "sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(raw_fname)
info = raw.info

# Create epochs for auditory events
events = mne.find_events(raw)
event_id = dict(left=2)
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin=-0.1,
    tmax=0.3,
    baseline=(None, 0),
    reject=dict(mag=4e-12, grad=4000e-13, eog=150e-6),
)

# Create evokeds for left and right auditory stimulation
evoked = epochs["left"].average()

########################################################################################
# Guided dipole modeling, meaning fitting dipoles to a manually selected subset of
# sensors as a manually chosen time, can now be performed using the
# :class:`mne.gui.DipoleFitting` GUI. By specifying only ``evokeds``, a default noise
# covariance matrix and 1-layer spherical head model will be used.
mne.gui.dipolefit(evoked)

###############################################################################
# The source modeling can be made more precise by using a noise covariance matrix and
# MRI-based 3-layer BEM head model:
cov_fname = meg_path / "sample_audvis-shrunk-cov.fif"
subjects_dir = data_path / "subjects"
bem_dir = subjects_dir / "sample" / "bem"
bem_fname = bem_dir / "sample-5120-5120-5120-bem-sol.fif"
trans_fname = meg_path / "sample_audvis_raw-trans.fif"

cov = mne.read_cov(cov_fname)  # bad channels were already excluded here
bem = mne.read_bem_solution(bem_fname)
trans = mne.read_trans(trans_fname)

mne.gui.dipolefit(
    evoked,
    cov=cov.as_diag(),
    bem=bem,
    trans=trans,
    subject="sample",
    subjects_dir=subjects_dir,
)

# Fit two dipoles at t=80ms. The first dipole is fitted using only the sensors
# on the left side of the helmet. The second dipole is fitted using only the
# sensors on the right side of the helmet.
picks_left = read_vectorview_selection("Left", info=info)
evoked_fit_left = evoked_left.copy().crop(0.08, 0.08)
evoked_fit_left.pick(picks_left)
cov_fit_left = cov.copy().pick_channels(picks_left, ordered=True)

picks_right = read_vectorview_selection("Right", info=info)
picks_right = list(set(picks_right) - set(info["bads"]))
evoked_fit_right = evoked_right.copy().crop(0.08, 0.08)
evoked_fit_right.pick(picks_right)
cov_fit_right = cov.copy().pick_channels(picks_right, ordered=True)

# Any SSS projections that are active on this data need to be re-normalized
# after picking channels.
evoked_fit_left.info.normalize_proj()
evoked_fit_right.info.normalize_proj()
cov_fit_left["projs"] = evoked_fit_left.info["projs"]
cov_fit_right["projs"] = evoked_fit_right.info["projs"]

# Fit the dipoles with the subset of sensors.
dip_left, _ = mne.fit_dipole(evoked_fit_left, cov_fit_left, bem)
dip_right, _ = mne.fit_dipole(evoked_fit_right, cov_fit_right, bem)

###############################################################################
# Now that we have the location and orientations of the dipoles, compute the
# full timecourses using MNE, assigning activity to both dipoles at the same
# time while preventing leakage between the two. We use a very low ``lambda``
# value to ensure both dipoles are fully used.

fwd, _ = mne.make_forward_dipole([dip_left, dip_right], bem, info)

# Apply MNE inverse
inv = make_inverse_operator(info, fwd, cov, fixed=True, depth=0)
stc_left = apply_inverse(evoked_left, inv, method="MNE", lambda2=1e-6)
stc_right = apply_inverse(evoked_right, inv, method="MNE", lambda2=1e-6)

# Plot the timecourses of the resulting source estimate
fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
axes[0].plot(stc_left.times, stc_left.data.T)
axes[0].set_title("Left auditory stimulation")
axes[0].legend(["Dipole 1", "Dipole 2"])
axes[1].plot(stc_right.times, stc_right.data.T)
axes[1].set_title("Right auditory stimulation")
axes[1].set_xlabel("Time (s)")
fig.supylabel("Dipole amplitude")

###############################################################################
# We can also fit the timecourses to single epochs. Here, we do it for each
# experimental condition separately.

stcs_left = apply_inverse_epochs(epochs["left"], inv, lambda2=1e-6, method="MNE")
stcs_right = apply_inverse_epochs(epochs["right"], inv, lambda2=1e-6, method="MNE")

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

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(np.arange(n), amplitudes[:, 0], label="Dipole 1")
ax.scatter(np.arange(n), amplitudes[:, 1], label="Dipole 2")
transition_point = n_left - 0.5
ax.plot([0, transition_point], [mean_left[0], mean_left[0]], color="C0")
ax.plot([0, transition_point], [mean_left[1], mean_left[1]], color="C1")
ax.plot([transition_point, n], [mean_right[0], mean_right[0]], color="C0")
ax.plot([transition_point, n], [mean_right[1], mean_right[1]], color="C1")
ax.axvline(transition_point, color="black")
ax.set_xlabel("Epochs")
ax.set_ylabel("Dipole amplitude")
ax.legend()
fig.suptitle("Single epoch dipole amplitudes")
fig.text(0.30, 0.9, "Left auditory stimulation", ha="center")
fig.text(0.70, 0.9, "Right auditory stimulation", ha="center")
