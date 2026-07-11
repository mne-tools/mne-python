"""
.. _tut-inverse-methods:

========================================================
Source localization with MNE, dSPM, sLORETA, and eLORETA
========================================================

The aim of this tutorial is to teach you how to compute and apply a linear
minimum-norm inverse method on evoked/raw/epochs data.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, make_inverse_operator

# %%
# Process MEG data

data_path = sample.data_path()
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"

raw = mne.io.read_raw_fif(raw_fname)  # already has an average reference
events = mne.find_events(raw, stim_channel="STI 014")

event_id = dict(aud_l=1)  # event trigger and conditions
tmin = -0.2  # start of each epoch (200ms before the trigger)
tmax = 0.5  # end of each epoch (500ms after the trigger)
raw.info["bads"] = ["MEG 2443", "EEG 053"]  # mark known bad channels
baseline = (None, 0)  # means from the first instant to t = 0
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    proj=True,
    picks=("meg", "eog"),
    baseline=baseline,
    reject=reject,
)

# %%
# Compute regularized noise covariance
# ------------------------------------
# For more details see :ref:`tut-compute-covariance`.

noise_cov = mne.compute_covariance(
    epochs, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=True
)

fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw.info)

# %%
# Compute the evoked response
# ---------------------------
# Let's just use the MEG channels for simplicity.

evoked = epochs.average().pick("meg")
evoked.plot(time_unit="s")
evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type="mag")

# %%
# It's also a good idea to look at whitened data:

evoked.plot_white(noise_cov, time_unit="s")
del epochs, raw  # to save memory

# %%
# Inverse modeling: MNE/dSPM on evoked and raw data
# -------------------------------------------------
# Here we first read the forward solution. You will likely need to compute
# one for your own data -- see :ref:`tut-forward` for information on how
# to do it.

fname_fwd = data_path / "MEG" / "sample" / "sample_audvis-meg-oct-6-fwd.fif"
fwd = mne.read_forward_solution(fname_fwd)

# %%
# Next, we make an MEG inverse operator.

inverse_operator = make_inverse_operator(
    evoked.info, fwd, noise_cov, loose=0.2, depth=0.8
)
del fwd

# %%
# .. note::
#
#     You can write the inverse operator to disk with:
#
#     .. code-block::
#
#         from mne.minimum_norm import write_inverse_operator
#         write_inverse_operator(
#             "sample_audvis-meg-oct-6-inv.fif", inverse_operator
#         )
#
# Compute inverse solution
# ------------------------
# We can use this to compute the inverse solution and obtain source time
# courses:

method = "dSPM"  # could choose MNE, sLORETA, or eLORETA instead
snr = 3.0
lambda2 = 1.0 / snr**2
stc, residual = apply_inverse(
    evoked,
    inverse_operator,
    lambda2,
    method=method,
    pick_ori=None,
    return_residual=True,
    verbose=True,
)

# %%
# Visualization
# -------------
# We can look at different dipole activations:

fig, ax = plt.subplots()
ax.plot(1e3 * stc.times, stc.data[::100, :].T)
ax.set(xlabel="time (ms)", ylabel=f"{method} value")

# %%
# Examine the original data and the residual after fitting:

fig, axes = plt.subplots(2, 1)
evoked.plot(axes=axes)
for ax in axes:
    for text in list(ax.texts):
        text.remove()
    for line in ax.lines:
        line.set_color("#98df81")
residual.plot(axes=axes)

# %%
# Here we use peak getter to move visualization to the time point of the peak
# and draw a marker at the maximum peak vertex.

# sphinx_gallery_thumbnail_number = 9

vertno_max, time_max = stc.get_peak(hemi="rh")

subjects_dir = data_path / "subjects"
surfer_kwargs = dict(
    hemi="rh",
    subjects_dir=subjects_dir,
    clim=dict(kind="value", lims=[8, 12, 15]),
    views="lateral",
    initial_time=time_max,
    time_unit="s",
    size=(800, 800),
    smoothing_steps=10,
)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(
    vertno_max,
    coords_as_verts=True,
    hemi="rh",
    color="blue",
    scale_factor=0.6,
    alpha=0.5,
)
brain.add_text(
    0.1, 0.9, "dSPM (plus location of maximal activation)", "title", font_size=14
)

# The documentation website's movie is generated with:
# brain.save_movie(..., tmin=0.05, tmax=0.15, interpolation='linear',
#                  time_dilation=20, framerate=10, time_viewer=True)

# %%
# Overlay all four methods simultaneously
# ----------------------------------------
#
# We compute all four inverse solutions and overlay them on the same
# :class:`~mne.viz.Brain`.  Because the four methods operate on very different
# amplitude scales (MNE/eLORETA in pA·m, dSPM/sLORETA as pseudo-Z scores),
# we normalize each method's data to ``[0, 1]`` using its own 5th–95th
# percentile range before overlaying so the colormaps are comparable.
#
# :func:`~mne.viz.Brain.add_data` defaults to ``key="data"`` when no key is
# supplied, which is what :func:`~mne.minimum_norm.apply_inverse` uses
# internally.  We pass an explicit key for every layer so the "Overlay"
# dropdown in the *Color Limits* panel shows meaningful names.


overlay_configs = [
    dict(method="dSPM", colormap="Reds", alpha=1.0),
    dict(method="MNE", colormap="Greens", alpha=0.7),
    dict(method="sLORETA", colormap="Blues", alpha=0.4),
    dict(method="eLORETA", colormap="RdPu", alpha=0.1),
]

surfer_kwargs_overlay = {k: v for k, v in surfer_kwargs.items() if k != "clim"}

brain2 = None
for i, cfg in enumerate(overlay_configs):
    stc_i, _ = apply_inverse(
        evoked,
        inverse_operator,
        lambda2,
        method=cfg["method"],
        pick_ori=None,
        return_residual=True,
        verbose=True,
    )

    data_rh = stc_i.rh_data
    p5, p95 = np.percentile(data_rh, [5, 95])
    data_rh_norm = np.clip((data_rh - p5) / (p95 - p5), 0, 1)

    if brain2 is None:
        brain2 = stc_i.plot(
            **surfer_kwargs_overlay,
            clim=dict(kind="value", lims=[0.5, 0.75, 1.0]),
            colormap=cfg["colormap"],
            alpha=cfg["alpha"],
        )

    brain2.add_data(
        data_rh_norm,
        vertices=stc_i.rh_vertno,
        hemi="rh",
        fmin=0.5,
        fmid=0.75,
        fmax=1.0,
        colormap=cfg["colormap"],
        alpha=cfg["alpha"],
        smoothing_steps=surfer_kwargs["smoothing_steps"],
        time=stc_i.times,
        initial_time=surfer_kwargs["initial_time"],
        key=cfg["method"].lower(),
        remove_existing=(i == 0),  # first pass removes the default "data" layer
    )

brain2.add_text(
    0.1, 0.9, "MNE · dSPM · sLORETA · eLORETA overlay", "title", font_size=14
)

# %%
# There are many other ways to visualize and work with source data, see
# for example:
#
# - :ref:`tut-viz-stcs`
# - :ref:`ex-morph-surface`
# - :ref:`ex-morph-volume`
# - :ref:`ex-vector-mne-solution`
# - :ref:`tut-dipole-orientations`
# - :ref:`tut-mne-fixed-free`
# - :ref:`examples using apply_inverse
#   <sphx_glr_backreferences_mne.minimum_norm.apply_inverse>`.
