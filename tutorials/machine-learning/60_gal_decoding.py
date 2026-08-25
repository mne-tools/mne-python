"""
.. _tut-gal-decoding:

=====================================================
Generalization across locations (GAL): Measuring cross-decodability between brain areas
=====================================================

The :ref:`temporal generalization example <tut-mvpa>` evaluates a decoder
estimated at one time point at every other time point. Its matrix shows whether
a multichannel activity pattern remains decodable over time.

The same operation can be applied across space or locations. Generalization Across
Location (GAL) uses the time course at one sensor (or ROI) as the feature vector, fits a
decoder at that sensor, and tests it at every other sensor. The matrix then
shows where a decodable temporal pattern transfers across the sensor array.

For epochs with shape ``trials x sensors x time``, temporal generalization uses
time as the iteration axis and sensors as features. GAL reverses those roles: sensor
locations are the iteration dimension and time samples are features. Both analyses use
:class:`mne.decoding.GeneralizingEstimator`. Importantly, such function uses by the default 
the last dimension (time) as iteration dimension. To change this behavior, here we set the parameter axis = 1 (sensors) to 
iterate in such dimension.

This tutorial applies GAL to face and non-face trials from MNE's
``visual_92_categories`` dataset. It applies the sensor-generalization
construction described in the `Time-GAL paper <https://doi.org/10.1002/hbm.70152>`_.
The paper describes this location-to-location classifier as a backward model.

An off-diagonal score measures transfer of decodable information. It does not
estimate anatomical or directed connectivity.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%
# Download two runs of the visual-object data
# --------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from mne_connectivity.viz import plot_connectivity_circle
from pandas import read_csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

import mne
from mne.datasets import visual_92_categories
from mne.decoding import GeneralizingEstimator
from mne.io import concatenate_raws, read_raw_fif

# %%
# Load the visual-object dataset
# ------------------------------

data_path = visual_92_categories.data_path()
conditions = read_csv(data_path / "visual_stimuli.csv")[:24]
event_id = {}
for condition in conditions.values:
    tags = list(condition[:2])
    tags += [
        ("not-" if value == 0 else "") + conditions.columns[index]
        for index, value in enumerate(condition[2:], 2)
    ]
    event_id["/".join(map(str, tags))] = condition[0] + 1
event_id["0/human bodypart/human/not-face/animal/natural"] = 1

raw = concatenate_raws(
    [
        read_raw_fif(
            data_path / f"sample_subject_{run}_tsss_mc.fif",
            verbose="error",
            on_split_missing="ignore",
        )
        for run in range(2)
    ]
)
events = mne.find_events(raw, min_duration=0.002)
events = events[events[:, 2] <= len(conditions)]

# %%
# Prepare face and non-face epochs
# --------------------------------
# We use 32 posterior magnetometers to keep the example quick. Each trial
# contributes its 50 to 300 ms waveform. Baseline correction precedes the
# crop.

mag_picks = mne.pick_types(raw.info, meg="mag")
mag_locations = np.array([raw.info["chs"][pick]["loc"][:3] for pick in mag_picks])
picks = mag_picks[np.argsort(mag_locations[:, 1])[:32]]
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    baseline=(None, 0),
    picks=picks,
    tmin=-0.1,
    tmax=0.5,
    preload=True,
    verbose=False,
)
X = epochs.copy().crop(0.05, 0.30).get_data()
face_triggers = conditions.loc[conditions["face"] == 1, "trigger"].to_numpy() + 1
y = np.isin(epochs.events[:, 2], face_triggers)
print(
    "Sensor-generalization input: "
    f"{X.shape[0]} trials, {X.shape[1]} sensors, {X.shape[2]} times"
)

# %%
# Inspect the sensor and time dimensions
# --------------------------------------
# The averages are a diagnostic view of the feature window. The estimator fits
# separate models at every sensor.

fig, ax = plt.subplots(layout="constrained")
time_mask = (epochs.times >= 0.05) & (epochs.times <= 0.30)
for selection, label in ((y, "Face"), (~y, "Non-face")):
    ax.plot(
        epochs.times[time_mask],
        epochs.get_data()[selection, 0][:, time_mask].mean(axis=0),
        label=label,
    )
ax.axvline(0, color="k", linestyle=":", linewidth=1)
ax.set(
    title=f"Condition averages at {epochs.ch_names[0]}",
    xlabel="Time (s)",
    ylabel="Magnetic field (T)",
)
ax.legend()

# %%
# The condition contrast is spatially structured
# ----------------------------------------------
# These maps show the face-minus-non-face field at three times in the decoding
# window. They describe evoked responses, not decoder weights or sources.

face_minus_non_face = mne.combine_evoked(
    [epochs[y].average(), epochs[~y].average()], weights=[1, -1]
)
fig, axes = plt.subplots(
    1,
    4,
    figsize=(10, 3.5),
    layout="constrained",
    gridspec_kw={"width_ratios": [1, 1, 1, 0.06]},
)
fig = face_minus_non_face.plot_topomap(
    times=[0.13, 0.17, 0.24],
    ch_type="mag",
    time_unit="s",
    axes=axes,
    show=False,
)

# %%
# The temporal pattern is described independently of decoding
# ------------------------------------------------------------
# The Time-GAL paper pairs this backward model with a correlation matrix.
# Here, Pearson correlation between the binary face label and each sensor's
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
    extent=(0.05, 0.30, -0.5, len(epochs.ch_names) - 0.5),
    origin="lower",
    vmin=-corr_limit,
    vmax=corr_limit,
)
ax.set(
    title="Label--signal correlation (descriptive forward pattern)",
    xlabel="Time (s)",
    ylabel="Posterior magnetometer",
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
# We use linear discriminant analysis, as in the paper.

sensor_gen = GeneralizingEstimator(
    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    scoring="accuracy",
    n_jobs=1,
    axis=1,
    verbose=False,
)

# %%
# Score held-out trials
# ---------------------
# The five stratified folds hold out trials from this participant. A group
# analysis should hold out participants.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = []
for train, test in cv.split(X, y):
    scores.append(sensor_gen.fit(X[train], y[train]).score(X[test], y[test]))
scores = np.array(scores)
mean_scores = scores.mean(axis=0)
off_diagonal = ~np.eye(len(epochs.ch_names), dtype=bool)
print(
    "Mean accuracy: "
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
    title="Sensor generalization: face versus non-face decoding",
    xlabel="Test sensor",
    ylabel="Training sensor",
)
sensor_ticks = np.arange(0, len(epochs.ch_names), 4)
sensor_tick_labels = np.array(epochs.ch_names)[sensor_ticks]
ax.set_xticks(sensor_ticks, sensor_tick_labels, rotation=45, ha="right")
ax.set_yticks(sensor_ticks, sensor_tick_labels)
fig.colorbar(image, ax=ax, label="Cross-validated accuracy")

# %%
# Summarize the strongest cross-sensor effects
# ---------------------------------------------
# ``mne-connectivity`` draws the 20 largest, symmetrised off-diagonal effects.
# This is a display of sensor-generalization scores, not a physical-connectivity
# graph.

cross_sensor_effects = mean_scores - 0.5
cross_sensor_effects = (cross_sensor_effects + cross_sensor_effects.T) / 2
np.fill_diagonal(cross_sensor_effects, 0)
fig, _ = plot_connectivity_circle(
    cross_sensor_effects,
    node_names=epochs.ch_names,
    n_lines=20,
    colormap="RdBu_r",
    title="Largest cross-decoding effects",
    show=False,
)
