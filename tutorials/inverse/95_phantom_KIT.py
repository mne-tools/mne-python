"""
.. _tut-phantom-kit:

============================
KIT phantom dataset tutorial
============================

Here we read KIT data obtained from a phantom with 49 dipoles sequentially activated
with 2-cycle 11 Hz sinusoidal bursts to verify source localization accuracy.
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%
import mne_bids
import numpy as np

import mne

data_path = mne.datasets.phantom_kit.data_path()
actual_pos, actual_ori = mne.dipole.get_phantom_dipoles("oyama")
actual_pos, actual_ori = actual_pos[:49], actual_ori[:49]  # only 49 of 50 dipoles

bids_path = mne_bids.BIDSPath(
    root=data_path,
    subject="01",
    task="phantom",
    run="01",
    datatype="meg",
)
# ignore warning about misc units
raw = mne_bids.read_raw_bids(bids_path).load_data()

# Let's apply a little bit of preprocessing (temporal filtering and reference
# regression)
picks_artifact = ["MISC 001", "MISC 002", "MISC 003"]
picks = np.r_[
    mne.pick_types(raw.info, meg=True),
    mne.pick_channels(raw.info["ch_names"], picks_artifact),
]
raw.filter(None, 40, picks=picks)
mne.preprocessing.regress_artifact(
    raw, picks="meg", picks_artifact=picks_artifact, copy=False, proj=False
)
plot_scalings = dict(mag=5e-12)  # large-amplitude sinusoids
raw_plot_kwargs = dict(duration=15, n_channels=50, scalings=plot_scalings)
events, event_id = mne.events_from_annotations(raw)
raw.plot(events=events, **raw_plot_kwargs)
n_dip = len(event_id)
assert n_dip == 49  # sanity check

# %%
# We can also look at the power spectral density to see the phantom oscillations at
# 11 Hz plus the expected frequency-domain sinc-like oscillations due to the time-domain
# boxcar windowing of the 11 Hz sinusoid.

spectrum = raw.copy().crop(0, 60).compute_psd(n_fft=10000)
fig = spectrum.plot(amplitude=False)
fig.axes[0].set_xlim(0, 50)
dip_freq = 11.0
fig.axes[0].axvline(dip_freq, color="r", ls="--", lw=2, zorder=4)

# %%
# Now  we can figure out our epoching parameters and epoch the data, sanity checking
# some values along the way knowing how the stimulation was done.

tmin, tmax = -0.08, 0.18
epochs = mne.Epochs(raw, tmin=tmin, tmax=tmax, decim=10, picks="data", preload=True)
del raw
epochs.plot(scalings=plot_scalings)

# %%
# Now we can average the epochs for each dipole, get the activation at the peak time,
# and create an :class:`mne.EvokedArray` from the result.

t_peak = 1.0 / dip_freq / 4.0
data = np.zeros((len(epochs.ch_names), n_dip))
for di in range(n_dip):
    data[:, [di]] = epochs[f"dip{di + 1:02d}"].average().crop(t_peak, t_peak).data
evoked = mne.EvokedArray(data, epochs.info, tmin=0, comment="KIT phantom activations")
evoked.plot_joint()

# %%
# Let's fit dipoles at each dipole's peak activation time.

trans = mne.transforms.Transform("head", "mri", np.eye(4))
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.08)
cov = mne.compute_covariance(epochs, tmax=0, method="empirical")
dip, residual = mne.fit_dipole(evoked, cov, sphere, n_jobs=None)

# %%
# Finally let's look at the results.

# sphinx_gallery_thumbnail_number = 5

print(f"Average amplitude: {np.mean(dip.amplitude) * 1e9:0.1f} nAm")
print(f"Average GOF:       {np.mean(dip.gof):0.1f}%")
diffs = 1000 * np.sqrt(np.sum((dip.pos - actual_pos) ** 2, axis=-1))
print(f"Average loc error: {np.mean(diffs):0.1f} mm")
angles = np.rad2deg(np.arccos(np.abs(np.sum(dip.ori * actual_ori, axis=1))))
print(f"Average ori error: {np.mean(angles):0.1f}°")

fig = mne.viz.plot_alignment(
    evoked.info,
    trans,
    bem=sphere,
    coord_frame="head",
    meg="helmet",
    show_axes=True,
)
fig = mne.viz.plot_dipole_locations(
    dipoles=dip, mode="arrow", color=(0.2, 1.0, 0.5), fig=fig
)

actual_amp = np.ones(len(dip))  # misc amp to create Dipole instance
actual_gof = np.ones(len(dip))  # misc GOF to create Dipole instance
dip_true = mne.Dipole(dip.times, actual_pos, actual_amp, actual_ori, actual_gof)
fig = mne.viz.plot_dipole_locations(
    dipoles=dip_true, mode="arrow", color=(0.0, 0.0, 0.0), fig=fig
)

mne.viz.set_3d_view(figure=fig, azimuth=90, elevation=90, distance=0.5)
