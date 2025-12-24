"""
.. _tut-qc-report:

=============================================
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

Note:
The generated HTML report is intended to be opened directly in a browser.
Some interactive elements (e.g., sliders) may not function correctly
when the file is served via a local HTTP server.


"""

# Authors: The MNE-Python contributors
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

from pathlib import Path

import mne
from mne.preprocessing import ICA, create_eog_epochs

# %%
# Load the sample dataset
# ----------------------
# We load a pre-filtered MEG/EEG recording from the MNE sample dataset.
# Only channels relevant for QC (MEG, EEG, EOG, stimulus) are retained.

data_path = Path(mne.datasets.sample.data_path(verbose=False))
sample_dir = data_path / "MEG" / "sample"
subjects_dir = data_path / "subjects"

raw_path = sample_dir / "sample_audvis_filt-0-40_raw.fif"


raw = mne.io.read_raw(raw_path, preload=True)

# Retain only channels relevant for QC to simplify visualization and
# focus inspection on signals typically reviewed during data quality checks.
raw.pick(["meg", "eeg", "eog", "stim"])

sfreq = raw.info["sfreq"]  # Sampling Frequency (Hz)

# %%
# Create the QC report
# -------------------
# The report acts as a container that collects figures, tables, and text
# into a single HTML document.

report = mne.Report(
    title="Sample dataset – Quality Control report",
    subject="sample",
    subjects_dir=subjects_dir,
)

# %%
# Dataset overview
# ----------------
# A brief overview helps the reviewer immediately understand the scale and
# basic properties of the dataset.

html_overview = f"""

<ul>
  <li><b>Sampling frequency:</b> {sfreq:.1f} Hz</li>
  <li><b>Duration:</b> {raw.times[-1]:.1f} s</li>
  <li><b>Number of channels:</b> {len(raw.ch_names)}</li>
</ul>
<p>
These values should be checked for consistency with the experimental design.
Unexpected sampling rates, unusually short recordings, or missing channel
classes often indicate acquisition or conversion problems.
</p>
"""

report.add_html(
    title="Overview",
    html=html_overview,
    tags=("qc", "overview"),
)

# %%
# Raw data inspection
# -------------------
# Visual inspection of raw data is the single most important QC step.
# Here we inspect both the time series and the power spectral density (PSD).

report.add_raw(
    raw,
    title="Raw data overview",
    psd=True,
    tags=("qc", "raw"),
)

# Interpretation:
# - Look for channels with unusually large amplitudes or flat signals.
# - In the PSD, check for excessive low-frequency drift, strong line noise,
# or abnormal spectral shapes compared to neighboring channels.

# %%
# Events and stimulus timing
# --------------------------
# Correct event detection is crucial for all subsequent epoch-based analyses.

events = mne.find_events(raw)

report.add_events(
    events,
    sfreq=sfreq,
    title="Detected events",
    tags=("qc", "events"),
)

# Interpretation:
# - Verify that the number of events matches expectations.
# - Check that event timing is plausible and evenly distributed.
# - Missing or duplicated events often indicate trigger channel issues.
# %%

# Epoching and rejection statistics
# --------------------------------
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
    tags=("qc", "epochs"),
)

# Interpretation:
# - Excessive rejection rates suggest noisy data or overly strict thresholds.
# - Rejected epochs should be visually inspected to confirm true artifacts.

# %%
# Evoked responses
# ----------------
# Averaged responses should show physiologically plausible waveforms and
# reasonable signal-to-noise ratios.

cov_path = sample_dir / "sample_audvis-cov.fif"
evokeds = mne.read_evokeds(
    sample_dir / "sample_audvis-ave.fif",
    baseline=(None, 0),
)

report.add_evokeds(
    evokeds=evokeds[:2],
    noise_cov=cov_path,
    n_time_points=5,
    tags=("qc", "evoked"),
)

# Interpretation:
# - Check that evoked responses have the expected polarity and timing.
# - Absence of clear evoked structure may indicate poor data quality or
# incorrect event definitions.


# %%
# ICA for artifact inspection
# ---------------------------
# Independent Component Analysis (ICA) can be used during QC to identify
# stereotypical artifacts such as eye blinks and eye movements.
#
# For QC purposes, ICA is typically run with a lightweight configuration
# (e.g., fewer components or temporal decimation) to provide rapid feedback
# on data quality, rather than an optimized decomposition for final analysis.

ica = ICA(
    n_components=15,
    random_state=97,
    max_iter="auto",
)

# Fit ICA using a decimated signal for speed
ica.fit(raw, picks=("meg", "eeg"), decim=3)


# Identify EOG-related components
eog_epochs = create_eog_epochs(raw)
eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
ica.exclude = eog_inds

report.add_ica(
    ica=ica,
    inst=epochs,
    title="ICA components (artifact inspection)",
    tags=("qc", "ica"),
)

# Interpretation:
# - Use the topographic maps to identify spatial patterns characteristic
#   of artifacts (e.g., frontal patterns for eye blinks).
# - The component property viewer is intended for detailed inspection of
#   individual components and is most informative when combined with
# qe epoched data or explicit artifact scoring.
# - Components correlated with EOG should show frontal topographies and
# stereotyped time courses.
# - Only components clearly associated with artifacts should be excluded.

# %%
# Head position / HPI quality control
# ----------------------------------
# Continuous head-position tracking (cHPI) allows monitoring subject movement
# during MEG acquisition. Not all datasets contain usable cHPI information.
# This sample dataset does not contain usable cHPI information.

report.add_html(
    title="Head position / HPI (run locally)",
    html="""
<p>
Continuous head-position tracking (cHPI) estimation can be computationally
expensive and is therefore typically run selectively during QC.
If your dataset contains cHPI information,
you can run the following code locally:
</p>

<pre><code class="python">
head_pos = mne.chpi.compute_head_pos(raw.info, raw)
fig = mne.viz.plot_head_positions(
    head_pos,
    mode="traces",
    show=True,
)
</code></pre>

<p>
Stable traces indicate minimal head movement. Large drifts suggest
movement-related artifacts.
</p>
""",
    tags=("qc", "hpi"),
)


# Interpretation:
# - Stable traces indicate minimal head movement.
# - Large translations or rotations suggest movement-related artifacts and
# may motivate movement compensation or data exclusion.

# %%
# MEG–MRI coregistration
# ---------------------
# Accurate coregistration is critical for source localization.

report.add_html(
    title="MEG–MRI coregistration (run locally)",
    html="""
<p>
Coregistration visualization requires access to MRI surfaces and interactive
rendering, which require MRI surfaces and interactive 3D visualization.
Run the following code locally to inspect coregistration quality:
</p>

<pre><code class="python">
trans_path = sample_dir / "sample_audvis_raw-trans.fif"
report.add_trans(
    trans=trans_path,
    info=raw_path,
    subject="sample",
    subjects_dir=subjects_dir,
)
</code></pre>
""",
    tags=("qc", "coreg"),
)


# Interpretation:
# - Head shape points should align well with the MRI scalp surface.
# - Systematic misalignment indicates digitization or transformation errors.

# %%
# MRI and BEM surfaces
# -------------------
# Boundary Element Method (BEM) surfaces define the head model used for
# forward and inverse solutions.

report.add_html(
    title="MRI and BEM surfaces (run locally)",
    html="""
<p>
BEM surface visualization is typically performed interactively when preparing
source-space analyses.
To inspect BEM surfaces locally, run:
</p>

<pre><code class="python">
report.add_bem(
    subject="sample",
    subjects_dir=subjects_dir,
    decim=40,
)
</code></pre>
""",
    tags=("qc", "bem"),
)


# Interpretation:
# - Surfaces should be smooth, closed, and non-intersecting.
# - Poorly formed surfaces can severely degrade source estimates.

# %%
# Summary
# -------
# A concise summary provides a checklist-style confirmation of completed QC.

html_summary = """

<ul>
  <li>Raw data and spectra inspected</li>
  <li>Events and epoch rejection verified</li>
  <li>Evoked responses checked for plausibility</li>
  <li>ICA components reviewed for artifacts</li>
  <li>Head position stability assessed (if available)</li>
  <li>Coregistration and BEM validated</li>
</ul>
<p>
For automated, large-scale QC across BIDS datasets, see the reports generated
by <code>mne-bids-pipeline</code>, which follow a similar philosophy but operate
at scale.
</p>
"""

report.add_html(
    title="QC summary",
    html=html_summary,
    tags=("qc", "summary"),
)

# %%
# Save report
# -----------

# %%
# Save report
# ----------------------------
# Run this script locally to generate the HTML report.
report.save(
    "qc_report.html",
    overwrite=True,
    open_browser=False,
)
