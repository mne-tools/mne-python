"""
.. _tut-brainstorm-elekta-phantom:

==========================================
Brainstorm Elekta phantom dataset tutorial
==========================================

This tutorial provides a step-by-step guide to
importing and processing Elekta-Neuromag current phantom recordings.
The aim of this tutorial is to show the user how to use phantom recordings to
evaluate source localisation methods by comparing estimated vs real dipole positions.

For comparison, see :footcite:`TadelEtAl2011` and
`the original Brainstorm tutorial with an explanation of phantom recordings
<https://neuroimage.usc.edu/brainstorm/Tutorials/PhantomElekta>`__.
"""
# sphinx_gallery_thumbnail_number = 9

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne import find_events, fit_dipole
from mne.datasets import fetch_phantom
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif

print(__doc__)

# %%
# The data were collected with an Elekta Neuromag VectorView system
# at 1000 Hz and low-pass filtered at 330 Hz.
# Here the medium-amplitude (200 nAm, amplitudes can be seen in raw data)
# data are accessed to construct instances of :class:`mne.io.Raw`.
data_path = bst_phantom_elekta.data_path(verbose=True)
raw_fname = data_path / "kojak_all_200nAm_pp_no_chpi_no_ms_raw.fif"
raw = read_raw_fif(raw_fname)

# %%
# Let's first get an idea of the data structure we are working with:

raw.info

# %%
# The data consists of 204 MEG planor gradiometers,
# 102 axial magnetometers, and 3 stimulus channels.

# Next, let's look at the events in the phantom data for one stimulus channel.
events = find_events(raw, "STI201")
raw.info["bads"] = ["MEG1933", "MEG2421"]  # known bad channels

# we have 32 artificial dipoles stored as events
# the remaining event IDs are not relevant for this tutorial (256, 768 and so on)
# %%
# Let's explore the phantom data by plotting power spectral density for each sensor.
# Here, we use only the first 30 seconds to save memory

raw.compute_psd(tmax=30).plot(
    average=False, amplitude=False, picks="data", exclude="bads"
)

# We can see that the data has strong line frequency (60 Hz, 120 Hz ...) noise
# and cHPI (continuous head position indicator) coil noise (peaks around 300 Hz).
# %%
# Next we plot the dipole events

raw.plot(events=events, n_channels=10)

# We can see that the simulated dipoles produce sinusoidal bursts at 20 Hz
# %%
# Next, we epoch the the data based on the dipoles events (1:32).

# We select 100 ms before and after the event trigger
# and baseline correct the epochs from -100 ms to -0.05 ms before stimulus onset.

tmin, tmax = -0.1, 0.1
bmax = -0.05  # Avoid capture filter ringing into baseline
event_id = list(range(1, 33))
epochs = mne.Epochs(
    raw, events, event_id, tmin, tmax, baseline=(None, bmax), preload=False
)

# Here we average the epochs for the first simulated dipole
# and plot the evoked signal
epochs["1"].average().plot(time_unit="s")
# %%

# We averaged over 640 simulated events for the first dipole.
# The first peak in the data appears close to the trigger onset
# at around 3ms, with a peak repeating every 3ms.
# Thus the burst repetition rate is 3Hz.

# %%
# .. _plt_brainstorm_phantom_elekta_eeg_sphere_geometry:
#
# Finally, we source reconstruct the evoked simulated dipole.
# To do this we use a :ref:`sphere head geometry model <eeg_sphere_model>`
# and visualise the coordinate alignment and the sphere location. The phantom
# is properly modeled by a single-shell sphere with origin (0., 0., 0.).
#
# Even though this is a VectorView/TRIUX phantom, we can use the Otaniemi
# phantom subject as a surrogate because the "head" surface (hemisphere outer
# shell) has the same geometry for both phantoms, even though the internal
# dipole locations differ. The phantom_otaniemi scan was aligned to the
# phantom's head coordinate frame, so an identity ``trans`` is appropriate
# here.

subjects_dir = data_path
fetch_phantom("otaniemi", subjects_dir=subjects_dir)
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.08)
subject = "phantom_otaniemi"
trans = mne.transforms.Transform("head", "mri", np.eye(4))
mne.viz.plot_alignment(
    epochs.info,
    subject=subject,
    show_axes=True,
    bem=sphere,
    dig=True,
    surfaces=("head-dense", "inner_skull"),
    trans=trans,
    mri_fiducials=True,
    subjects_dir=subjects_dir,
)
# %%
# We can see that our head model aligns with the phantom head model.
# %%
# Let's do some dipole fits.

# First we compute the noise covariance for the baseline window.
# See covariance/whitening tutorial for details :ref:`tut-compute-covariance`.
# %%
# The covariance captures the sensor noise structure.
cov = mne.compute_covariance(epochs, tmax=bmax)

# The plot shows the evoked signal divided by the estimated noise standard deviation.
mne.viz.plot_evoked_white(epochs["1"].average(), cov)

# Next, we fit the dipoles for the evoked data.
# We choose the timepoint which maximises global field power
# We have seen in the evoked plot that this is around 3 ms after dipole onset.
data = []
t_peak = 0.036  # true for Elekta phantom
for ii in event_id:
    # Avoid the first and last trials -- can contain dipole-switching artifacts
    evoked = epochs[str(ii)][1:-1].average().crop(t_peak, t_peak)
    data.append(evoked.data[:, 0])
evoked = mne.EvokedArray(np.array(data).T, evoked.info, tmin=0.0)
del epochs  # save memory
dip, residual = fit_dipole(evoked, cov, sphere, n_jobs=None)
# %%
# Whitened global field power (GFP):

# most baseline activity should fall roughly within ±1 (unit variance).

# %%
# Let's visualize the explained variance.

# To do this, we need to make sure that the
# data and the residuals are on the same scale
# (here the "time points" are the 32 dipole peak values that we fit).

fig, axes = plt.subplots(2, 1)
evoked.plot(axes=axes)
for ax in axes:
    for text in list(ax.texts):
        text.remove()
    for line in ax.lines:
        line.set_color("#98df81")
residual.plot(axes=axes)
# %%
# Here we visualise how well the dipole explains the evoked response (green line).
# The red lines represent the residuals, the leftover noise after dipole fitting.
# A good fit: green lines are strong and residuals are small and roughly flat.

# Finally, we compare the estimated to the true dipole locations.
actual_pos, actual_ori = mne.dipole.get_phantom_dipoles()
actual_amp = 100.0  # nAm

fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=3, ncols=1, figsize=(6, 7), layout="constrained"
)

# Here we calculate the euclidean distance between estimated and true positions.
# We multiply by 1000 to convert from meter to millimeter.
diffs = 1000 * np.sqrt(np.sum((dip.pos - actual_pos) ** 2, axis=-1))
print(f"mean(position error) = {np.mean(diffs):0.1f} mm")
ax1.bar(event_id, diffs)
ax1.set_xlabel("Dipole index")
ax1.set_ylabel("Loc. error (mm)")

# Next we calculate the angle between estimated and true orientation.
# We convert radians to degrees.
angles = np.rad2deg(np.arccos(np.abs(np.sum(dip.ori * actual_ori, axis=1))))
print(f"mean(angle error) = {np.mean(angles):0.1f}°")
ax2.bar(event_id, angles)
ax2.set_xlabel("Dipole index")
ax2.set_ylabel("Angle error (°)")

# Finally we compare amplitudes by subtracting estimated from true amplitude.
amps = actual_amp - dip.amplitude / 1e-9
print(f"mean(abs amplitude error) = {np.mean(np.abs(amps)):0.1f} nAm")
ax3.bar(event_id, amps)
ax3.set_xlabel("Dipole index")
ax3.set_ylabel("Amplitude error (nAm)")
# %%
# The dipole fits closely match the true phantom data.
# We can achieve sub-centimeter accuracy with a mean position error of 2.6 mm.
# This demonstrates that the fitting procedure is accurate.

# Finally, we can plot the positions and the orientations
# of the estimated and true dipoles.

actual_amp = np.ones(len(dip))  # fake amp, needed to create Dipole instance
actual_gof = np.ones(len(dip))  # fake goodness-of-fit (GOF)
dip_true = mne.Dipole(dip.times, actual_pos, actual_amp, actual_ori, actual_gof)

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
    dipoles=dip, mode="arrow", subject=subject, color=(0.2, 1.0, 0.5), fig=fig
)

mne.viz.set_3d_view(figure=fig, azimuth=70, elevation=80, distance=0.5)
# %%
# The dipoles overlap and point in the same direction.
# %%
# References
# ----------
# .. footbibliography::
