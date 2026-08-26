"""
.. _tut-gal-decoding:

======================================================
Generalization across locations (GAL): Generalizing face decoding across EEG locations
======================================================

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

This tutorial uses the N170 face-perception task from the ERP CORE dataset. We
contrast faces with scrambled faces and use every scalp electrode. This visual
ERP gives the sensor-by-sensor analysis a more interpretable structure than the
motor-imagery data used in the temporal-generalization tutorial.

The sensor-generalization construction follows the `Time-GAL paper
<https://doi.org/10.1002/hbm.70152>`_, which describes the location-to-location
classifier as a backward model. An off-diagonal score measures transfer of
decodable information; it does not estimate anatomical or directed
connectivity.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%
# Download one ERP CORE N170 recording
# ------------------------------------
# ERP CORE is available from NEMAR as a BIDS dataset. We download only the
# N170 recording from participant 1 (about 94 MB), rather than the complete
# multi-participant archive. The recording contains 80 face and 80 scrambled
# face trials.

import matplotlib.pyplot as plt
import numpy as np
from mne_connectivity.viz import plot_connectivity_circle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mne
from mne.channels import make_standard_montage
from mne.decoding import GeneralizingEstimator, SlidingEstimator, cross_val_multiscore
from mne.io import read_raw_eeglab

fetch_file = mne.datasets.erp_core.fetch_file
fetch_file("sub-001/eeg/sub-001_task-N170_eeg.fdt")
raw_fname = fetch_file("sub-001/eeg/sub-001_task-N170_eeg.set")
events_fname = fetch_file("sub-001/eeg/sub-001_task-N170_events.tsv")
raw = read_raw_eeglab(raw_fname, preload=True)

# %%
# Prepare the epochs
# ------------------
# The EEGLAB recording labels the three ocular channels as EEG. Mark them as
# EOG before re-referencing. We use the standard 10-20 positions associated
# with the recorded cap labels for consistent sensor-space plots. All 30 scalp
# electrodes enter the analysis; ``picks=\"eeg\"`` below removes only EOG.

raw.set_channel_types(
    {name: "eog" for name in ("HEOG_left", "HEOG_right", "VEOG_lower")}
)
raw.set_montage(make_standard_montage("colin27_1020"), match_case=False)
raw.filter(0.1, 30.0, fir_design="firwin")
raw.set_eeg_reference(projection=True).apply_proj()

events_tsv = np.genfromtxt(
    events_fname,
    delimiter="\t",
    names=True,
    dtype=None,
    encoding="utf-8",
)
is_target_trial = (events_tsv["trial_type"] == "stimulus") & np.isin(
    events_tsv["event_type"], ("face", "scrambled_face")
)
events_tsv = events_tsv[is_target_trial]
event_id = dict(face=1, scrambled_face=2)
events = np.column_stack(
    (
        events_tsv["sample"],
        np.zeros(len(events_tsv), dtype=int),
        np.where(events_tsv["event_type"] == "face", 1, 2),
    )
).astype(int)
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=-0.2,
    tmax=0.6,
    baseline=(None, 0),
    picks="eeg",
    preload=True,
    verbose=False,
)
feature_epochs = epochs.copy().crop(0.05, 0.25)
X = feature_epochs.get_data()
y = epochs.events[:, 2] == event_id["face"]
print(
    "Sensor-generalization input: "
    f"{X.shape[0]} trials, {X.shape[1]} scalp sensors, {X.shape[2]} times"
)

# %%
# Classical time-resolved decoding uses sensors as features
# ---------------------------------------------------------
# This is the temporal slice from the :ref:`temporal generalization example
# <tut-mvpa>`: a model is fitted at every time point using the full scalp array
# as features. GAL below swaps these two roles.

classifier = make_pipeline(
    StandardScaler(), LogisticRegression(solver="liblinear", random_state=0)
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
time_decoder = SlidingEstimator(classifier, scoring="roc_auc", n_jobs=1, verbose=False)
time_scores = cross_val_multiscore(time_decoder, X, y, cv=cv, n_jobs=1).mean(axis=0)

fig, ax = plt.subplots(layout="constrained")
ax.plot(feature_epochs.times, time_scores)
ax.axhline(0.5, color="k", linestyle=":", linewidth=1)
ax.set(
    title="Time-resolved decoding: face versus scrambled face",
    xlabel="Time (s)",
    ylabel="Cross-validated AUC",
)

# %%
# Inspect the condition waveforms
# -------------------------------
# The averages are a diagnostic view of the feature window. GAL fits separate
# models at every scalp electrode. PO8 is used only as a familiar posterior-site
# illustration; it is not selected for the analysis.

fig, ax = plt.subplots(layout="constrained")
time_mask = (epochs.times >= 0.05) & (epochs.times <= 0.25)
for selection, label in ((y, "Face"), (~y, "Scrambled face")):
    ax.plot(
        epochs.times[time_mask],
        epochs.get_data()[selection, epochs.ch_names.index("PO8")][:, time_mask].mean(
            axis=0
        ),
        label=label,
    )
ax.axvline(0, color="k", linestyle=":", linewidth=1)
ax.set(
    title="Face-perception averages at PO8",
    xlabel="Time (s)",
    ylabel="Voltage (V)",
)
ax.legend()

# %%
# The face contrast is spatially structured
# ------------------------------------------
# These maps show the face-minus-scrambled-face voltage at three times in the
# decoding window. They describe condition averages, not decoder weights or
# sources.

face_minus_scrambled = mne.combine_evoked(
    [epochs[y].average(), epochs[~y].average()], weights=[1, -1]
)
fig, axes = plt.subplots(
    1,
    4,
    figsize=(10, 3.5),
    layout="constrained",
    gridspec_kw={"width_ratios": [1, 1, 1, 0.06]},
)
fig = face_minus_scrambled.plot_topomap(
    times=[0.12, 0.17, 0.22],
    ch_type="eeg",
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
    extent=(
        feature_epochs.times[0],
        feature_epochs.times[-1],
        -0.5,
        len(epochs.ch_names) - 0.5,
    ),
    origin="lower",
    vmin=-corr_limit,
    vmax=corr_limit,
)
ax.set(
    title="Label--signal correlation (descriptive temporal pattern)",
    xlabel="Time (s)",
    ylabel="Scalp sensor",
)
fig.colorbar(image, ax=ax, label="Pearson r")

# %%
# Generalize waveform decoders across sensors
# --------------------------------------------
# :class:`~mne.decoding.GeneralizingEstimator` treats the final dimension as
# the task axis by default. ``axis=1`` selects sensors instead. The data remain
# ordered as ``trials x sensors x time``: each model is trained on one sensor's
# time samples and scored on every sensor's time samples.

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
# pattern learned at one location at another location. Below-chance transfer
# can occur when the two locations carry opposite-polarity patterns. A single
# participant's matrix is not a group-level inference.

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
    title="Sensor generalization: face versus scrambled face",
    xlabel="Test sensor",
    ylabel="Training sensor",
)
sensor_ticks = np.arange(len(epochs.ch_names))
ax.set_xticks(sensor_ticks, epochs.ch_names, rotation=90)
ax.set_yticks(sensor_ticks, epochs.ch_names)
ax.tick_params(axis="both", labelsize=7)
fig.colorbar(image, ax=ax, label="Cross-validated AUC")

# %%
# Show an individual cross-sensor effect
# --------------------------------------
# The largest off-diagonal score is useful for connecting one matrix cell to
# the data. The model is trained on the time course at the left panel's sensor
# and evaluated at the right panel's sensor. The two panels show the face and
# scrambled-face averages that produce those trial-wise feature vectors.
#
# This is an exploratory view, not a statistical selection procedure. The AUC
# was evaluated on held-out trials; inference about the population would
# require repeating the analysis across participants.

off_diagonal_scores = mean_scores.copy()
np.fill_diagonal(off_diagonal_scores, np.nan)
train_sensor, test_sensor = np.unravel_index(
    np.nanargmax(off_diagonal_scores), off_diagonal_scores.shape
)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True, layout="constrained")
for ax, sensor, role in zip(
    axes, (train_sensor, test_sensor), ("Training", "Test"), strict=True
):
    for selection, label in ((y, "Face"), (~y, "Scrambled face")):
        ax.plot(
            feature_epochs.times,
            X[selection, sensor].mean(axis=0),
            label=label,
        )
    ax.axvline(0, color="k", linestyle=":", linewidth=1)
    ax.set(title=f"{role}: {epochs.ch_names[sensor]}", xlabel="Time (s)")
axes[0].set_ylabel("Voltage (V)")
axes[1].legend(loc="best")
fig.suptitle(
    "Strongest cross-sensor cell: "
    f"{epochs.ch_names[train_sensor]} to {epochs.ch_names[test_sensor]} "
    f"(held-out AUC = {mean_scores[train_sensor, test_sensor]:.2f})"
)

# %%
# Summarize the strongest cross-sensor effects
# ---------------------------------------------
# ``mne-connectivity`` draws the largest, symmetrised off-diagonal effects.
# This is a display of sensor-generalization scores, not a physical-connectivity
# graph.

cross_sensor_effects = mean_scores - 0.5
cross_sensor_effects = (cross_sensor_effects + cross_sensor_effects.T) / 2
np.fill_diagonal(cross_sensor_effects, 0)
fig, _ = plot_connectivity_circle(
    cross_sensor_effects,
    node_names=epochs.ch_names,
    n_lines=8,
    colormap="RdBu_r",
    title="Largest cross-decoding effects",
    show=False,
)
