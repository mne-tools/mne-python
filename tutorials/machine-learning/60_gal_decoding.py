"""
.. _tut-gal-decoding:

=============================================
Generalizing decoding across time and sensors
=============================================

Generalization is defined by the axis that is sliced into tasks. With epoched
data of shape ``trials x sensors x time``,
:class:`mne.decoding.GeneralizingEstimator` can ask either whether a decoder
generalizes across time or whether it generalizes across sensors:

.. list-table:: Two complementary slices of the same epochs
   :header-rows: 1

   * - Question
     - Task axis
     - Features within each task
   * - :ref:`Temporal generalization <tut-mvpa>`
     - Time
     - Sensors
   * - Sensor generalization (this tutorial)
     - Sensors
     - Time samples

Both analyses fit a classifier at every training slice and test it at every
other slice. The resulting matrix has training slices as rows and test slices
as columns.

This tutorial uses face versus non-face trials from MNE's
``visual_92_categories`` dataset. The `Time-GAL paper
<https://doi.org/10.1002/hbm.70152>`_ uses the same sensor-generalization
idea, but the name is not essential to the analysis.

An off-diagonal sensor-generalization score measures transfer of decodable
information. It is not an anatomical or directed-connectivity estimate.
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
# Choose the sensor-generalization slice
# --------------------------------------
# MNE stores the data as ``trials x sensors x time``. Setting ``axis=1`` makes
# sensors the task axis and leaves time samples as features.

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
# The contrast supplies labels, and the time window supplies features
# --------------------------------------------------------------------
# We use 32 posterior magnetometers to keep the example quick. The classifier
# receives each trial's 50 to 300 ms waveform. Baseline correction precedes the
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
# Inspect the condition waveforms before decoding
# ------------------------------------------------
# The averages are a diagnostic view of the feature window. The estimator later
# fits a separate model at every sensor.

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
# Inspect the sensor-level contrast
# ---------------------------------
# The measured face-minus-non-face field is shown at three times in the
# decoding window. It describes the evoked response, not decoder weights or
# sources.

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
# Describe the condition effect separately from the classifier
# -------------------------------------------------------------
# The paper places a forward temporal pattern beside its matrix. Here it is the
# Pearson correlation between the binary face label and each sensor's trial
# signal. It is calculated independently of the classifier.

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
# Fit one waveform decoder per sensor
# -----------------------------------
# ``GeneralizingEstimator(axis=1)`` takes sensors as tasks, leaving time as
# features. We use linear discriminant analysis, as in the paper.

sensor_gen = GeneralizingEstimator(
    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    scoring="accuracy",
    n_jobs=1,
    axis=1,
    verbose=False,
)

# %%
# The split determines the claim
# -------------------------------
# The five stratified folds hold out trials from this participant. A group
# analysis would instead hold out participants.

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
# Read the sensor-generalization matrix
# -------------------------------------
# Rows identify training sensors and columns identify test sensors. The
# diagonal is within-sensor decoding. Interpret the 32 x 32 matrix as a whole;
# a single cell is not a group-level inference.

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
