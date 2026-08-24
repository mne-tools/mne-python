"""
.. _tut-gal-decoding:

======================================================
Generalization across MEG locations with time features
======================================================

The `Time-GAL paper <https://doi.org/10.1002/hbm.70152>`_ fits a classifier
to the waveform at one sensor, then evaluates that classifier at every other
sensor. This tutorial implements the same analysis in MNE-Python. Each output
cell is a cross-validated accuracy for one training sensor and one test sensor.

We use face versus non-face trials from MNE's ``visual_92_categories`` data,
which also support the :ref:`RSA example <ex-rsa-noplot>`. The tutorial
demonstrates the analysis, rather than reproducing the MATLAB output.

Conventional time-resolved decoding uses sensors as features and repeats the
fit across time. GAL uses time samples as features and repeats the fit across
sensors. It asks whether the condition-related temporal pattern at one sensor
also predicts labels at another.

An off-diagonal GAL score is not anatomical or directed connectivity. It can
reflect a shared generator, correlated sensor signals, or field spread. Read
the matrix as evidence about transfer of decodable information, alongside the
sensor geometry and the preprocessing choices.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%
# Download two runs of the visual-object data
# --------------------------------------------
# This CI-supported dataset replaces the paper's external EEG archive.

import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

import mne
from mne.datasets import visual_92_categories
from mne.decoding import GeneralizingEstimator
from mne.io import concatenate_raws, read_raw_fif

# %%
# GAL changes the feature axis and the generalization axis
# --------------------------------------------------------
# Figure 1 in the paper makes this contrast visually. The two analyses differ
# only in which data dimension is treated as features and which is iterated.

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
for axis, title, x_label, y_label, output in (
    (
        axes[0],
        "Conventional time-resolved decoding",
        "Sensors are features",
        "Repeat over time",
        "Decoding over time",
    ),
    (
        axes[1],
        "GAL: temporal decoding",
        "Time samples are features",
        "Repeat over sensors",
        "Generalization across sensors",
    ),
):
    axis.imshow(np.arange(30).reshape(5, 6), cmap="Blues", aspect="auto")
    axis.set(xticks=[], yticks=[], title=title, xlabel=x_label, ylabel=y_label)
    axis.annotate(
        output,
        xy=(0.5, -0.28),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontweight="bold",
    )

# %%
# The array defines the statistical question
# ------------------------------------------
# Time-GAL uses ``sensors x time x trials``; MNE Epochs use
# ``trials x sensors x time``. The ordering changes, but the roles do not:
#
# * **observations** are trials;
# * **features** are time samples from one sensor;
# * **tasks** are the sensors at which a decoder is fitted or evaluated;
# * **target** is the experimental condition; and
# * **score** is held-out classification accuracy.
#
# Every row is a different decoder. The diagonal tests its own sensor.
# Off-diagonal cells test the same fitted decoder at a different sensor.

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
# We use 32 magnetometers to keep the documentation example short. Remove the
# slice to analyse every magnetometer.
#
# The paper contrasts pleasant with unpleasant ERP trials, then habituation
# with extinction ssVEP trials. Here the contrast is face versus non-face. In
# all three cases, the classifier receives the 50--300 ms trial waveform, not
# an average over trials. Baseline correction happens before the crop.

picks = mne.pick_types(raw.info, meg="mag")[:32]
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
print(f"GAL input: {X.shape[0]} trials, {X.shape[1]} sensors, {X.shape[2]} times")

# %%
# The backward matrix and the forward description answer different questions
# ---------------------------------------------------------------------------
# The paper separates labelled waveforms, one decoder per channel, the GAL
# matrix, and a forward temporal description. We compute the first three.
# A forward model correlates labels with signals; classifier weights do not
# directly describe activation patterns.
#
# In the paper, Pearson correlation between labels and sensor time series
# identifies when condition-related differences occur. GAL identifies where a
# predictive temporal pattern transfers. Neither result substitutes for the
# other.

fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), layout="constrained")
labels = ("A / B trials", "Temporal\nfeatures", "One decoder\nper sensor", "GAL matrix")
for index, (axis, label) in enumerate(zip(axes, labels, strict=True)):
    axis.set(title=label, xticks=[], yticks=[])
    if index == 0:
        time = np.linspace(0, 2 * np.pi, 100)
        axis.plot(time, np.sin(time), color="tab:red", label="A")
        axis.plot(time, 0.6 * np.sin(time + 0.8), color="goldenrod", label="B")
        axis.legend(frameon=False, loc="upper right")
    elif index == 1:
        axis.imshow(X[:20, 0, :].T, aspect="auto", cmap="RdBu_r")
    elif index == 2:
        axis.text(0.5, 0.5, "LDA", ha="center", va="center", fontsize=18)
    else:
        axis.imshow(np.eye(8), cmap="RdBu_r", vmin=0, vmax=1)
    if index < len(axes) - 1:
        axis.annotate(
            "",
            xy=(1.12, 0.5),
            xytext=(0.9, 0.5),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )

# %%
# Fit one temporal decoder per sensor
# -----------------------------------
# ``GeneralizingEstimator(axis=1)`` takes sensor 1 as the task axis. Each fit
# therefore uses time as features and evaluates across sensors.
#
# We use linear discriminant analysis, as in the paper. Here 251 time features
# are fitted from a limited number of trials, so automatic covariance shrinkage
# stabilizes the covariance estimate. MATLAB's ``fitcdiscr`` chooses its own
# regularization, so this choice does not target bitwise MATLAB agreement.

gal = GeneralizingEstimator(
    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    scoring="accuracy",
    n_jobs=1,
    axis=1,
    verbose=False,
)

# %%
# The split determines the claim
# -------------------------------
# This single-participant dataset uses stratified trial folds. Several
# participants require leave-one-subject-out cross-validation.
#
# The split must match the intended unit of inference. The paper holds out a
# participant because its result targets a population. Trial folds estimate
# within-participant generalization only.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = []
for train, test in cv.split(X, y):
    scores.append(gal.fit(X[train], y[train]).score(X[test], y[test]))
scores = np.array(scores)
mean_scores = scores.mean(axis=0)

# %%
# Read the matrix as a transfer map
# ---------------------------------
# Rows identify the training sensor; columns identify the test sensor. The
# diagonal is within-sensor temporal decoding.
#
# Above-chance values show transfer of condition information. A structured
# off-diagonal region matters more than one isolated cell. For group inference,
# retain every participant's matrix, test cells across participants, and adjust
# for the number of sensor pairs before drawing contours.

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
    title="GAL: face versus non-face temporal decoding",
    xlabel="Test sensor",
    ylabel="Training sensor",
)
fig.colorbar(image, ax=ax, label="Cross-validated accuracy")

# %%
# Relation to representational similarity analysis
# -------------------------------------------------
# The RSA example uses sensors as features to distinguish many image classes
# and summarizes a confusion matrix. GAL reverses those dimensions: each
# sensor's time series is the feature vector and the output describes
# generalization across sensors. Both are multivariate analyses of this MEG
# dataset.
#
# RSA asks which *stimulus classes* have similar multichannel response patterns.
# GAL asks which *sensor locations* share a decodable temporal pattern for a
# specified contrast. Both use cross-validation, but their matrices have
# different axes and scientific meanings. Keeping this distinction explicit
# prevents reading a GAL matrix as a representational dissimilarity matrix.

# %%
# References
# ----------
# .. footbibliography::
