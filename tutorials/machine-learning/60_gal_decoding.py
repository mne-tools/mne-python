"""
.. _tut-gal-decoding:

====================================================
Generalizing motor-imagery decoding across locations
====================================================

The :ref:`temporal generalization example <tut-mvpa>` evaluates a decoder
estimated at one time point at every other time point. Its matrix shows whether
a multichannel activity pattern remains decodable over time.

The same operation can be applied across locations. Generalization Across
Location (GAL) uses the time course at one sensor as the feature vector, fits a
decoder at that sensor, and tests it at every other sensor. The matrix then
shows where a decodable temporal pattern transfers across the sensor array.

For epochs with shape ``trials x sensors x time``, temporal generalization uses
time as the task axis and sensors as features. GAL reverses those roles: sensor
locations are tasks and time samples are features. Both analyses use
:class:`mne.decoding.GeneralizingEstimator`.

This tutorial uses the hand and foot motor-imagery runs from the EEGBCI dataset,
as in the :ref:`ERDS example <ex-tfr-erds>`. It applies the
sensor-generalization construction described in the `Time-GAL paper
<https://doi.org/10.1002/hbm.70152>`_. The paper describes this
location-to-location classifier as a backward model.

An off-diagonal score measures transfer of decodable information. It does not
estimate anatomical or directed connectivity.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%
# Download motor-imagery EEG data
# --------------------------------

import matplotlib.pyplot as plt
import numpy as np
from mne_connectivity.viz import plot_connectivity_circle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mne
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import GeneralizingEstimator, SlidingEstimator, cross_val_multiscore
from mne.io import concatenate_raws, read_raw_edf

# %%
# The ERDS example uses runs 6, 10, and 14 from participant 1. These runs
# contain imagined hand and foot movements. We retain the complete EEG array:
# each electrode is a possible training and test location.

raw_fnames = eegbci.load_data(subjects=1, runs=(6, 10, 14))
raw = concatenate_raws([read_raw_edf(fname, preload=True) for fname in raw_fnames])
eegbci.standardize(raw)
raw.set_montage(make_standard_montage("spherical_1005"))
raw.annotations.rename(dict(T1="hands", T2="feet"))
raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
raw.set_eeg_reference(projection=True).apply_proj()
raw = mne.preprocessing.compute_current_source_density(raw)

# %%
# Prepare hand and foot motor-imagery epochs
# -------------------------------------------
# The surface Laplacian is a reference-free current-source-density (CSD)
# transform. It suppresses broad, volume-conducted activity and sharpens local
# sensor patterns, which can make the mu- and beta-band ERD/ERS contrast easier
# to see. For GAL, the 1 to 3 s waveform becomes the feature vector at each
# location.

event_id = dict(hands=2, feet=3)
epochs = mne.Epochs(
    raw,
    event_id=event_id,
    tmin=-1,
    tmax=4,
    baseline=None,
    proj=True,
    preload=True,
    verbose=False,
)
X = epochs.copy().crop(1, 3).get_data()
y = epochs.events[:, 2] == event_id["hands"]
print(
    "Sensor-generalization input: "
    f"{X.shape[0]} trials, {X.shape[1]} sensors, {X.shape[2]} times"
)

# %%
# Classical time-resolved decoding uses sensors as features
# ---------------------------------------------------------
# This is the temporal slice from the :ref:`temporal generalization example
# <tut-mvpa>`: a model is fitted at every time point using the full EEG array
# as features. GAL below swaps these two roles.

classifier = make_pipeline(
    StandardScaler(), LogisticRegression(solver="liblinear", random_state=0)
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
time_decoder = SlidingEstimator(classifier, scoring="roc_auc", n_jobs=1, verbose=False)
time_scores = cross_val_multiscore(time_decoder, X, y, cv=cv, n_jobs=1).mean(axis=0)

fig, ax = plt.subplots(layout="constrained")
ax.plot(np.linspace(1, 3, len(time_scores)), time_scores)
ax.axhline(0.5, color="k", linestyle=":", linewidth=1)
ax.set(
    title="Time-resolved decoding: hands versus feet",
    xlabel="Time (s)",
    ylabel="Cross-validated AUC",
)

# %%
# Inspect the condition waveforms
# -------------------------------
# The averages are a diagnostic view of the 1 to 3 s feature window. GAL fits
# separate models at every sensor. C3 is used only as a familiar central-site
# illustration; it is not selected for the analysis.

fig, ax = plt.subplots(layout="constrained")
time_mask = (epochs.times >= 1) & (epochs.times <= 3)
for selection, label in ((y, "Hands"), (~y, "Feet")):
    ax.plot(
        epochs.times[time_mask],
        epochs.get_data()[selection, epochs.ch_names.index("C3")][:, time_mask].mean(
            axis=0
        ),
        label=label,
    )
ax.axvline(0, color="k", linestyle=":", linewidth=1)
ax.set(
    title="Motor-imagery averages at C3",
    xlabel="Time (s)",
    ylabel="CSD (V/m²)",
)
ax.legend()

# %%
# The condition contrast is spatially structured
# ----------------------------------------------
# These maps show the hand-minus-foot voltage at three times in the decoding
# window. They describe evoked responses, not decoder weights or sources.

hands_minus_feet = mne.combine_evoked(
    [epochs[y].average(), epochs[~y].average()], weights=[1, -1]
)
fig, axes = plt.subplots(
    1,
    4,
    figsize=(10, 3.5),
    layout="constrained",
    gridspec_kw={"width_ratios": [1, 1, 1, 0.06]},
)
fig = hands_minus_feet.plot_topomap(
    times=[1.2, 1.8, 2.5],
    ch_type="csd",
    time_unit="s",
    axes=axes,
    show=False,
)

# %%
# The temporal pattern is described independently of decoding
# ------------------------------------------------------------
# The Time-GAL paper pairs this backward model with a correlation matrix.
# Here, Pearson correlation between the binary hand-imagery label and each sensor's
# trial signal describes when the condition effect occurs.

X_centered = X - X.mean(axis=0)
y_centered = y - y.mean()
label_signal_correlation = np.einsum("n,nct->ct", y_centered, X_centered)
label_signal_correlation /= np.sqrt(
    np.sum(y_centered**2) * np.sum(X_centered**2, axis=0)
)
corr_limit = np.max(np.abs(label_signal_correlation))
fig, ax = plt.subplots(layout="constrained")
image = ax.imshow(
    label_signal_correlation,
    aspect="auto",
    cmap="RdBu_r",
    extent=(1, 3, -0.5, len(epochs.ch_names) - 0.5),
    origin="lower",
    vmin=-corr_limit,
    vmax=corr_limit,
)
ax.set(
    title="Label--signal correlation (descriptive temporal pattern)",
    xlabel="Time (s)",
    ylabel="EEG sensor",
)
fig.colorbar(image, ax=ax, label="Pearson r")

# %%
# Generalize waveform decoders across sensors
# --------------------------------------------
# :class:`~mne.decoding.GeneralizingEstimator` treats the final dimension as
# the task axis by default. ``axis=1`` selects sensors instead. The data remain
# ordered as ``trials x sensors x time``: each model is trained on one sensor's
# time samples and scored on every sensor's time samples.
#
# The same logistic-regression pipeline is used for both slices.

sensor_gen = GeneralizingEstimator(
    classifier,
    scoring="roc_auc",
    n_jobs=1,
    axis=1,
    verbose=False,
)

# %%
# Score held-out trials
# ---------------------
# The three stratified folds hold out trials from this participant.

scores = []
for train, test in cv.split(X, y):
    scores.append(sensor_gen.fit(X[train], y[train]).score(X[test], y[test]))
scores = np.array(scores)
mean_scores = scores.mean(axis=0)
off_diagonal = ~np.eye(len(epochs.ch_names), dtype=bool)
print(
    "Mean AUC: "
    f"{np.diag(mean_scores).mean():.3f} within sensor; "
    f"{mean_scores[off_diagonal].mean():.3f} across sensors"
)

# %%
# A matrix summarizes location-to-location transfer
# --------------------------------------------------
# Rows identify training sensors and columns identify test sensors. The
# diagonal is within-sensor decoding. Off-diagonal cells test the temporal
# pattern learned at one location at another location. A single cell is not a
# group-level inference.

limit = np.max(np.abs(mean_scores - 0.5))
fig, ax = plt.subplots(layout="constrained")
image = ax.imshow(
    mean_scores,
    origin="lower",
    cmap="RdBu_r",
    vmin=0.5 - limit,
    vmax=0.5 + limit,
    interpolation="nearest",
)
ax.set(
    title="Sensor generalization: hands versus feet",
    xlabel="Test sensor",
    ylabel="Training sensor",
)
sensor_tick_labels = np.array(("C3", "Cz", "C4"))
sensor_ticks = np.array([epochs.ch_names.index(name) for name in sensor_tick_labels])
ax.set_xticks(sensor_ticks, sensor_tick_labels, rotation=45, ha="right")
ax.set_yticks(sensor_ticks, sensor_tick_labels)
fig.colorbar(image, ax=ax, label="Cross-validated AUC")

# %%
# Summarize the strongest cross-sensor effects
# ---------------------------------------------
# ``mne-connectivity`` draws the three largest, symmetrised off-diagonal effects.
# This is a display of sensor-generalization scores, not a physical-connectivity
# graph.

cross_sensor_effects = mean_scores - 0.5
cross_sensor_effects = (cross_sensor_effects + cross_sensor_effects.T) / 2
np.fill_diagonal(cross_sensor_effects, 0)
fig, _ = plot_connectivity_circle(
    cross_sensor_effects,
    node_names=epochs.ch_names,
    n_lines=3,
    colormap="RdBu_r",
    title="Largest cross-decoding effects",
    show=False,
)
