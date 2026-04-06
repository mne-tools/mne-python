"""
.. _tut-brainstorm-elekta-phantom:

==========================================
Brainstorm Elekta phantom dataset tutorial
==========================================

This tutorial provides a step-by-step guide to
importing and processing Elekta-Neuromag current phantom recordings.

A phantom recording is a measurement obtained using a device (phantom)
that generates known magnetic signals,
allowing validation and benchmarking of MEG system accuracy and analysis methods.

The aim of this tutorial is to demonstrate how phantom recordings can be used to
evaluate source localisation methods by comparing estimated and true dipole positions.

For comparison, see :footcite:`TadelEtAl2011` and
`the original Brainstorm tutorial
<https://neuroimage.usc.edu/brainstorm/Tutorials/PhantomElekta>`__.
"""
# sphinx_gallery_thumbnail_number = 9

# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Carina Forster <carinaforster0611@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import mne
from mne import find_events, fit_dipole
from mne.datasets import fetch_phantom
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif

# %%
# Load and prepare the data
# -------------------------------

# The data were collected with an Elekta Neuromag VectorView system
# at 1000 Hz and low-pass filtered at 330 Hz.
#
# The dataset contains recordings at three current amplitudes (20, 200, and 2000 nAm).
# Here we load the medium-amplitude condition.
data_path = bst_phantom_elekta.data_path(verbose=True)
raw_fname = data_path / "kojak_all_200nAm_pp_no_chpi_no_ms_raw.fif"
raw = read_raw_fif(raw_fname)

# Mark known bad channels
raw.info["bads"] = ["MEG1933", "MEG2421"]

events = find_events(raw, "STI201")

# The first 32 events correspond to dipole activations.

# %%
# Epoch the data and plot evokeds
# ---------------
#
# We epoch the data around dipole events and apply baseline correction.

bmax = -0.05
tmin, tmax = -0.1, 0.8
event_id = list(range(1, 33))

epochs = mne.Epochs(
    raw, events, event_id, tmin, tmax, baseline=(None, bmax), preload=False
)

# Plot evoked response for the first clean dipole
epochs["1"][1:-1].average().plot(time_unit="s")
# %%
# In this data the phantom was set to produce 20 Hz sinusoidal bursts of current.
# You can see that the burst envelope repeats at approximately 3 Hz.
#
# Determine peak activation using Global Field Power (GFP)
# --------------------------------------------------------

# GFP is the standard deviation across sensors at each time
# point, providing a reference-independent measure of signal strength.
evoked_tmp = epochs["1"][1:-1].average()
gfp = np.std(evoked_tmp.data, axis=0)
times = evoked_tmp.times

# Restrict to first burst window
time_mask = (times > 0) & (times <= 0.1)

peaks, _ = find_peaks(gfp[time_mask])
peak_indices = np.where(time_mask)[0][peaks]

# Select the strongest peak
strongest_peak_idx = peak_indices[np.argmax(gfp[peak_indices])]
t_peak = times[strongest_peak_idx]

print(f"Strongest peak at {t_peak * 1000:.1f} ms")
# %%
# Here we store the evoked data for each dipole at the peak amplitude.
evokeds = []
for ii in event_id:
    evoked = epochs[str(ii)][1:-1].average().crop(t_peak, t_peak)
    evoked = mne.EvokedArray(np.array(evoked.data), evoked.info, tmin=0.0)
    evokeds.append(evoked)
# %%
# Next, we need to compute the noise covariance to capture the sensor noise structure.
#
# We use the baseline window to estimate covariance.
#
# You can explore the covariance tutorial for details: :ref:`tut-compute-covariance`.

cov = mne.compute_covariance(epochs, tmax=bmax)

del epochs  # delete to save memory
# %%
# We use a :ref:`sphere head geometry model <eeg_sphere_model>`
# to fit our phantom head model.
subjects_dir = data_path
fetch_phantom("otaniemi", subjects_dir=subjects_dir)
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.08)

# %%
# Fit dipoles
# -----------

# We fit dipoles for each phantom and store them in a list.
dip_all, residuals_all = [], []

for evoked in evokeds:
    dip, residual = fit_dipole(evoked, cov, sphere, n_jobs=1)
    dip_all.append(dip)
    residuals_all.append(residual)
# %%
# Evaluate goodness of fit
# -----------------------

# The dipole object stores the goodness of fit (GOF) for each dipole.
gof = [dip.gof[0] for dip in dip_all]
colors = ["#E69F00" if val < 60 else "#0072B2" for val in gof]
plt.bar(event_id, gof, color=colors)
plt.xlabel("Phantom dipole estimation")
plt.ylabel("Goodness of fit (%)")
plt.show()

# %%
# We can see that GOF varies between 50 % and up to 95 %.
#
# Compare estimated and true dipoles
# --------------------------------

actual_pos, actual_ori = mne.dipole.get_phantom_dipoles()
actual_amp = 200.0  # nAm

# estimated dipoles
dip_pos = [dip.pos[0] for dip in dip_all]
dip_ori = [dip.ori[0] for dip in dip_all]
dip_amplitude = [dip.amplitude[0] for dip in dip_all]

fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=3, ncols=1, figsize=(6, 7), layout="constrained"
)

# Here we calculate the euclidean distance between estimated and true positions.
# We multiply by 1000 to convert from meter to millimeter.
diffs = 1000 * np.sqrt(np.sum((dip_pos - actual_pos) ** 2, axis=-1))
print(f"mean(position error) = {np.mean(diffs):0.1f} mm")
ax1.bar(event_id, diffs)
ax1.set_xlabel("Dipole index")
ax1.set_ylabel("Loc. error (mm)")

# Next we calculate the angle between estimated and true orientation.
# We convert radians to degrees.
angles = np.rad2deg(np.arccos(np.abs(np.sum(dip_ori * actual_ori, axis=1))))
print(f"mean(angle error) = {np.mean(angles):0.1f}°")
ax2.bar(event_id, angles)
ax2.set_xlabel("Dipole index")
ax2.set_ylabel("Angle error (°)")

# Here we compare amplitudes by subtracting estimated from true amplitude.
amps = actual_amp - np.array(dip_amplitude) / 1e-9
print(f"mean(abs amplitude error) = {np.mean(np.abs(amps)):0.1f} nAm")
ax3.bar(event_id, amps)
ax3.set_xlabel("Dipole index")
ax3.set_ylabel("Amplitude error (nAm)")
# %%
# The dipole fits closely match the true phantom data,
# achieving sub-centimeter accuracy (mean position error 2.7mm).
#
# Visualise estimated and true dipoles
# -----------------------

actual_amp = np.ones(len(dip))  # fake amp, needed to create Dipole instance
actual_gof = np.ones(len(dip))  # fake goodness-of-fit (GOF)
# setup dipole objects for true and estimated dipoles
dip_true = mne.Dipole(dip.times, actual_pos, actual_amp, actual_ori, actual_gof)
dip_estimated = mne.Dipole(dip.times, dip_pos, dip_amplitude, dip_ori, actual_gof)

subject = "phantom_otaniemi"
trans = mne.transforms.Transform("head", "mri", np.eye(4))

fig = mne.viz.plot_alignment(
    evoked.info,
    trans,
    subject,
    bem=sphere,
    surfaces={"head-dense": 0.2},
    coord_frame="head",
    meg="helmet",
    show_axes=True,
    subjects_dir=subjects_dir,
)

# Plot the position and the orientation of the true dipole in black
fig = mne.viz.plot_dipole_locations(
    dipoles=dip_true, mode="arrow", subject=subject, color=(0.0, 0.0, 0.0), fig=fig
)

# Plot the position and the orientation of the estimated dipole in green
fig = mne.viz.plot_dipole_locations(
    dipoles=dip_estimated, mode="arrow", subject=subject, color=(0.2, 1.0, 0.5), fig=fig
)
mne.viz.set_3d_view(figure=fig, azimuth=70, elevation=80, distance=0.5)
# %%
# We can see that the dipoles overlap, have approximately the same magnitude
# and point in the same direction.

# %%
# References
# ----------
# .. footbibliography::
