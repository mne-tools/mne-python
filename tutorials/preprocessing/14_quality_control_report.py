"""
.. _tut-qc-report:

============================================
Quality control (QC) reports with mne.Report
============================================

Quality control (QC) is the process of systematically inspecting M/EEG data
**throughout all stages of an analysis pipeline**, including raw data,
intermediate preprocessing steps, and derived results.

While QC often begins with an initial inspection of the raw recording,
it is equally important to verify that signals continue to "look reasonable"
after operations such as filtering, artifact correction, epoching, and
averaging. Issues introduced or missed at any stage can propagate downstream
and invalidate later analyses.

This tutorial demonstrates how to create a **single, narrative QC report**
using :class:`mne.Report`, focusing on **what should be inspected and how the
results should be interpreted**, rather than exhaustively covering the API.

For clarity and reproducibility, the examples below focus on common QC checks
applied at representative stages of an analysis pipeline. The same reporting
approach can—and should—be reused whenever new processing steps are applied.

We use the MNE sample dataset for demonstration. Not all QC sections are
applicable to every dataset (e.g., continuous head-position tracking), and
this tutorial explicitly handles such cases.

.. note:: For several additional examples of complete reports, see the
   `MNE-BIDS-Pipeline QC reports <https://mne.tools/mne-bids-pipeline/stable/examples/examples.html>`_.
"""  # noqa: E501

# Authors: The MNE-Python contributors
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

from pathlib import Path

import mne
from mne.preprocessing import ICA, create_eog_epochs

# %%
# Load the sample dataset
# -----------------------
# We load a pre-filtered MEG/EEG recording from the MNE sample dataset.
# Only channels relevant for QC (MEG, EEG, EOG, stimulus) are retained.

data_path = Path(mne.datasets.sample.data_path(verbose=False))
subject = "sample"
sample_dir = data_path / "MEG" / subject
subjects_dir = data_path / "subjects"

raw_path = sample_dir / "sample_audvis_filt-0-40_raw.fif"


raw = mne.io.read_raw(raw_path)

# We will also crop the dataset for speed
raw.crop(0, 60).load_data()

# Retain only channels relevant for QC to simplify visualization and
# focus inspection on signals typically reviewed during data quality checks.
raw.pick(["meg", "eeg", "eog", "stim"])

sfreq = raw.info["sfreq"]  # Sampling Frequency (Hz)

# %%
# Create the QC report
# --------------------
# The report acts as a container that collects figures, tables, and text
# into a single HTML document.

report = mne.Report(
    title="Sample dataset - Quality Control report",
    subject=subject,
    subjects_dir=subjects_dir,
)

# %%
# Dataset overview
# ----------------
# A brief overview helps the reviewer immediately understand the scale and
# basic properties of the dataset.

html_overview = """
This report presents a quality control (QC) overview of the MNE sample dataset.<br><br>
For information about the paradigm, see
<a href="https://mne.tools/stable/documentation/datasets.html#sample">the MNE docs</a>.
"""

report.add_html(
    title="Overview",
    html=html_overview,
    tags=("overview"),
)

# %%
# Raw data inspection
# -------------------
# Visual inspection of raw data is the single most important QC step.
# Here we inspect both the time series and the power spectral density (PSD).
#
# - Look for channels with unusually large amplitudes or flat signals.
# - In the PSD, check for excessive low-frequency drift, strong line noise,
#   or abnormal spectral shapes compared to neighboring channels.

report.add_raw(
    raw,
    title="Raw data overview",
    psd=False,  # omit just for speed here
)

# %%
# Events and stimulus timing
# --------------------------
# Correct event detection is crucial for all subsequent epoch-based analyses.
#
# - Verify that the number of events matches expectations.
# - Check that event timing is plausible and evenly distributed.
# - Missing or duplicated events often indicate trigger channel issues.

events = mne.find_events(raw)

report.add_events(
    events,
    sfreq=sfreq,
    title="Detected events",
)

# %%
# Epoching and rejection statistics
# ---------------------------------
# Epoching allows inspection of data segments time-locked to events, along
# with automated rejection based on amplitude thresholds.

event_id = {
    "auditory/left": 1,
    "auditory/right": 2,
    "visual/left": 3,
    "visual/right": 4,
}

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=-0.2,
    tmax=0.5,
    baseline=(None, 0),
    reject=dict(eeg=150e-6),
    preload=True,
)

report.add_epochs(
    epochs,
    title="Epochs and rejection statistics",
)

# %%
# Evoked responses
# ----------------
# Averaged responses should show physiologically plausible waveforms and
# reasonable signal-to-noise ratios.
#
# - Check that evoked responses have the expected polarity and timing.
# - Absence of clear evoked structure may indicate poor data quality or
#   incorrect event definitions.

cov_path = sample_dir / "sample_audvis-cov.fif"
evoked = mne.read_evokeds(
    sample_dir / "sample_audvis-ave.fif",
    baseline=(None, 0),
)[0]  # just one for speed
evoked.decimate(4)  # also for speed

report.add_evokeds(
    evokeds=evoked,
    noise_cov=cov_path,
    n_time_points=5,
)


# %%
# ICA for artifact inspection
# ---------------------------
# Independent Component Analysis (ICA) can be used during QC to identify
# stereotypical artifacts such as eye blinks and eye movements.
#
# For QC purposes, ICA is typically run with a lightweight configuration
# (e.g., fewer components or temporal decimation) to provide rapid feedback
# on data quality, rather than an optimized decomposition for final analysis.
#
# - Use the topographic maps to identify spatial patterns characteristic
#   of artifacts (e.g., frontal patterns for eye blinks).
# - The component property viewer is intended for detailed inspection of
#   individual components and is most informative when combined with
#   epoched data or explicit artifact scoring.
# - Components correlated with EOG should show frontal topographies and
#   stereotyped time courses.
# - Only components clearly associated with artifacts should be excluded.

ica = ICA(
    n_components=15,
    random_state=97,
    max_iter=50,  # just for speed!
)

# Fit ICA using a decimated signal for speed
ica.fit(raw, picks=("meg", "eeg"), decim=10, verbose="error")


# Identify EOG-related components
eog_epochs = create_eog_epochs(raw)
eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
ica.exclude = eog_inds

report.add_ica(
    ica=ica,
    inst=epochs,
    eog_evoked=eog_epochs.average(),
    eog_scores=eog_scores,
    title="ICA components (artifact inspection)",
)

# %%
# MEG–MRI coregistration
# ----------------------
# Accurate coregistration is critical for source localization.
#
# - Head shape points should align well with the MRI scalp surface.
# - Systematic misalignment indicates digitization or transformation errors.


trans = sample_dir / "sample_audvis_raw-trans.fif"
report.add_trans(
    trans,
    info=raw.info,
    title="MEG–MRI-head coregistration",
    subject=subject,
    subjects_dir=subjects_dir,
)

# %%
# MRI and BEM surfaces
# --------------------
# Boundary Element Method (BEM) surfaces define the head model used for
# forward and inverse solutions.
#
# - Surfaces should be smooth, closed, and non-intersecting.
# - Poorly formed surfaces can severely degrade source estimates.

report.add_bem(
    subject,
    subjects_dir=subjects_dir,
    title="BEM surfaces",
    decim=20,  # for speed
)

# %%
# View the final report
# ---------------------
# You can set ``open_browser=True`` to have it pop open a browser tab if you want:

report.save("qc_report.html", overwrite=True, open_browser=False)
