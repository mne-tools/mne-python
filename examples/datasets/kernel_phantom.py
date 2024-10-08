"""
.. _ex-kernel-opm-phantom:

Kernel OPM phantom data
=======================

In this dataset, a Neuromag phantom was placed inside the Kernel OPM helmet and
stimulated with 7 modules active (121 channels). Here we show some example traces.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

import mne

data_path = mne.datasets.phantom_kernel.data_path()
fname = data_path / "phantom_32_100nam_raw.fif"
raw = mne.io.read_raw_fif(fname).load_data()
events = mne.find_events(raw, stim_channel="STI101")

# Bads identified by inspecting averages
raw.info["bads"] = [
    "RC2.bx.ave",
    "LC3.bx.ave",
    "RC2.by.7",
    "RC2.bz.7",
    "RC2.bx.4",
    "RC2.by.4",
    "LC3.bx.5",
]
# Drop the module-average channels
raw.drop_channels([ch_name for ch_name in raw.ch_names if ".ave" in ch_name])
# Add field correction projectors
raw.add_proj(mne.preprocessing.compute_proj_hfc(raw.info, order=2))
raw.pick("meg", exclude="bads")
raw.filter(0.5, 40)
epochs = mne.Epochs(
    raw,
    events,
    tmin=-0.1,
    tmax=0.25,
    decim=5,
    preload=True,
    baseline=(None, 0),
)
evoked = epochs["17"].average()  # a high-SNR dipole for these data
fig = evoked.plot()
t_peak = 0.016  # based on visual inspection of evoked
fig.axes[0].axvline(t_peak, color="k", ls=":", lw=3, zorder=2)

# %%
# The data covariance has an interesting structure because of densely packed sensors:

cov = mne.compute_covariance(epochs, tmax=-0.01)
mne.viz.plot_cov(cov, raw.info)

# %%
# So let's be careful and compute rank ahead of time and regularize:

rank = mne.compute_rank(epochs, tol=1e-3, tol_kind="relative")
cov = mne.compute_covariance(epochs, tmax=-0.01, rank=rank, method="shrunk")
mne.viz.plot_cov(cov, raw.info)

# %%
# Look at our alignment:

sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.08)
trans = mne.transforms.Transform("head", "mri", np.eye(4))
align_kwargs = dict(
    trans=trans,
    bem=sphere,
    surfaces={"outer_skin": 0.2},
    show_axes=True,
)
mne.viz.plot_alignment(
    raw.info,
    coord_frame="meg",
    meg=dict(sensors=1.0, helmet=0.05),
    **align_kwargs,
)

# %%
# Let's do dipole fits, which are not great because the dev_head_t is approximate and
# the sensor coverage is sparse:

data = list()
for ii in range(1, 33):
    evoked = epochs[str(ii)][1:-1].average().crop(t_peak, t_peak)
    data.append(evoked.data[:, 0])
evoked = mne.EvokedArray(np.array(data).T, evoked.info, tmin=0.0)
del epochs
dip, residual = mne.fit_dipole(evoked, cov, sphere, n_jobs=None)
actual_pos, actual_ori = mne.dipole.get_phantom_dipoles()
actual_amp = np.ones(len(dip))  # fake amp, needed to create Dipole instance
actual_gof = np.ones(len(dip))  # fake GOF, needed to create Dipole instance
dip_true = mne.Dipole(dip.times, actual_pos, actual_amp, actual_ori, actual_gof)

fig = mne.viz.plot_alignment(
    evoked.info, coord_frame="head", meg="sensors", **align_kwargs
)
mne.viz.plot_dipole_locations(
    dipoles=dip_true, mode="arrow", color=(0.0, 0.0, 0.0), fig=fig
)
mne.viz.plot_dipole_locations(dipoles=dip, mode="arrow", color=(0.2, 1.0, 0.5), fig=fig)
mne.viz.set_3d_view(figure=fig, azimuth=30, elevation=70, distance=0.4)
