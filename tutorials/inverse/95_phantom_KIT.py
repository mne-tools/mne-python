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
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import mne

data_path = mne.datasets.phantom_kit.data_path()
actual_pos, actual_ori = mne.dipole.get_phantom_dipoles("oyama")
actual_pos, actual_ori = actual_pos[:49], actual_ori[:49]  # only 49 of 50 dipoles

raw = mne.io.read_raw_kit(data_path / "002_phantom_11Hz_100uA.con")
# cut from ~800 to ~300s for speed, and also at convenient dip stim boundaries
# chosen by examining MISC 017 by eye.
raw.crop(11.5, 302.9).load_data()
# 11 Hz stimulation, no need to keep higher freqs
picks_artifact = ["MISC 001", "MISC 002", "MISC 003"]
picks = np.r_[
    mne.pick_types(raw.info, meg=True),
    mne.pick_channels(raw.info["ch_names"], picks_artifact),
]
raw.filter(None, 40, picks=picks)
# Apply reference regression
mne.preprocessing.regress_artifact(
    raw, picks="meg", picks_artifact=picks_artifact, copy=False, proj=False
)
plot_scalings = dict(mag=5e-12)  # large-amplitude sinusoids
raw_plot_kwargs = dict(duration=15, n_channels=50, scalings=plot_scalings)
raw.plot(**raw_plot_kwargs)

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
# To find the events, we can look at the MISC channel that recorded the activations.
# Here we use a very simple thresholding approach to find the events.
# The MISC 017 channel holds the dipole activations, which are 2-cycle 11 Hz sinusoidal
# bursts with the initial sinusoidal deflection downward, so we do a little bit of
# signal manipulation to help :func:`~scipy.signal.find_peaks`.

# Figure out events
dip_act, dip_t = raw["MISC 017"]
dip_act = dip_act[0]  # 2D to 1D array
dip_act -= dip_act.mean()  # remove DC offset
dip_act *= -1  # invert so first deflection is positive
thresh = np.percentile(dip_act, 90)
min_dist = raw.info["sfreq"] / dip_freq * 0.9  # 90% of period, to be safe
peaks = find_peaks(dip_act, height=thresh, distance=min_dist)[0]
assert len(peaks) % 2 == 0  # 2-cycle modulations
peaks = peaks[::2]  # take only first peaks of each 2-cycle burst

fig, ax = plt.subplots(layout="constrained", figsize=(12, 4))
stop = int(15 * raw.info["sfreq"])  # 15 sec
ax.plot(dip_t[:stop], dip_act[:stop], color="k", lw=1)
ax.axhline(thresh, color="r", ls="--", lw=1)
peak_idx = peaks[peaks < stop]
ax.plot(dip_t[peak_idx], dip_act[peak_idx], "ro", zorder=5, ms=5)
ax.set(xlabel="Time (s)", ylabel="Dipole activation (AU)\n(MISC 017 adjusted)")
ax.set(xlim=dip_t[[0, stop - 1]])

# We know that there are 32 dipoles, so mark the first ones as well
n_dip = 49
assert len(peaks) % n_dip == 0  # we found them all (hopefully)
ax.plot(dip_t[peak_idx[::n_dip]], dip_act[peak_idx[::n_dip]], "bo", zorder=4, ms=10)

# Knowing we've caught the top of the first cycle of a 11 Hz sinusoid, plot onsets
# with red X's.
onsets = peaks - np.round(raw.info["sfreq"] / dip_freq / 4.0).astype(
    int
)  # shift to start
onset_idx = onsets[onsets < stop]
ax.plot(dip_t[onset_idx], dip_act[onset_idx], "rx", zorder=5, ms=5)

# %%
# Given the onsets are now stored in ``peaks``, we can create our events array and plot
# on our raw data.

n_rep = len(peaks) // n_dip
events = np.zeros((len(peaks), 3), int)
events[:, 0] = onsets + raw.first_samp
events[:, 2] = np.tile(np.arange(1, n_dip + 1), n_rep)
raw.plot(events=events, **raw_plot_kwargs)

# %%
# Now  we can figure out our epoching parameters and epoch the data, sanity checking
# some values along the way knowing how the stimulation was done.

# Sanity check and determine epoching params
deltas = np.diff(events[:, 0], axis=0)
group_deltas = deltas[n_dip - 1 :: n_dip] / raw.info["sfreq"]  # gap between 49 and 1
assert (group_deltas > 0.8).all()
assert (group_deltas < 0.9).all()
others = np.delete(deltas, np.arange(n_dip - 1, len(deltas), n_dip))  # remove 49->1
others = others / raw.info["sfreq"]
assert (others > 0.25).all()
assert (others < 0.3).all()
tmax = 1 / dip_freq * 2.0  # 2 cycles
tmin = tmax - others.min()
assert tmin < 0
epochs = mne.Epochs(
    raw,
    events,
    tmin=tmin,
    tmax=tmax,
    baseline=(None, 0),
    decim=10,
    picks="data",
    preload=True,
)
del raw
epochs.plot(scalings=plot_scalings)

# %%
# Now we can average the epochs for each dipole, get the activation at the peak time,
# and create an :class:`mne.EvokedArray` from the result.

t_peak = 1.0 / dip_freq / 4.0
data = np.zeros((len(epochs.ch_names), n_dip))
for di in range(n_dip):
    data[:, [di]] = epochs[str(di + 1)].average().crop(t_peak, t_peak).data
evoked = mne.EvokedArray(data, epochs.info, tmin=0, comment="KIT phantom activations")
evoked.plot_joint()

# %%
# Let's fit dipoles at each dipole's peak activation time.

trans = mne.transforms.Transform("head", "mri", np.eye(4))
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.08)
cov = mne.compute_covariance(epochs, tmax=0, method="empirical")
# We need to correct the ``dev_head_t`` because it's incorrect for these data!
# relative to the helmet: hleft, forward, up
translation = mne.transforms.translation(x=0.01, y=-0.015, z=-0.088)
# pitch down (rot about x/R), roll left (rot about y/A), yaw left (rot about z/S)
rotation = mne.transforms.rotation(
    x=np.deg2rad(5),
    y=np.deg2rad(-1),
    z=np.deg2rad(-3),
)
evoked.info["dev_head_t"]["trans"][:] = translation @ rotation
dip, residual = mne.fit_dipole(evoked, cov, sphere, n_jobs=None)

# %%
# Finally let's look at the results.

# sphinx_gallery_thumbnail_number = 7

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
