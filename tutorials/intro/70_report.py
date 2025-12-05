"""
.. _tut-report:

===============================
Getting started with mne.Report
===============================

:class:`mne.Report` is a way to create interactive HTML summaries of your data.
These reports can show many different visualizations for one or multiple
participants. A common use case is creating diagnostic summaries to check data
quality at different stages in the processing pipeline. The report can show
things like plots of data before and after each preprocessing step, epoch
rejection statistics, MRI slices with overlaid BEM shells, all the way up to
plots of estimated cortical activity.

Compared to a Jupyter notebook, :class:`mne.Report` is easier to deploy, as the
HTML pages it generates are self-contained and do not require a running Python
environment. However, it is less flexible as you can't change code and re-run
something directly within the browser. This tutorial covers the basics of
building a report.

This tutorial demonstrates how to generate a full MNE-Python report combining:
- Quality Control (QC) steps: raw data, PSD, events, epochs, ICA
- Full MNE diagnostics: covariance, projectors, BEM, coregistration,
  forward/inverse solutions, source estimates
- Custom figures and HTML content
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


# This block imports all the Python libraries we need. These include MNE for M/EEG
# processing, NumPy for numerical operations, Matplotlib for plotting, and
# Pathlib for handling file paths. They provide all the basic tools required to
# load data, run QC, create figures, and build the final HTML report.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.preprocessing import ICA

"""
Here we load the MNE sample dataset, which contains raw data, events, evoked
responses, and anatomical files. Using this dataset allows us to demonstrate how
to generate a full QC + MNE report without requiring your own data.
"""

data_path = Path(mne.datasets.sample.data_path(verbose=False))
sample_dir = data_path / "MEG" / "sample"
subjects_dir = data_path / "subjects"

raw_fname = sample_dir / "sample_audvis_raw.fif"
events_fname = sample_dir / "sample_audvis_raw-eve.fif"
evoked_fname = sample_dir / "sample_audvis-ave.fif"
cov_fname = sample_dir / "sample_audvis-cov.fif"
fwd_fname = sample_dir / "sample_audvis-meg-oct-6-fwd.fif"
inv_fname = sample_dir / "sample_audvis-meg-oct-6-meg-inv.fif"
trans_fname = sample_dir / "sample_audvis_raw-trans.fif"
proj_fname = sample_dir / "sample_audvis_ecg-proj.fif"

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.pick_types(meg=True, eeg=True, eog=True, stim=True)

events = mne.read_events(events_fname)

"""
This function makes a simple static plot of the first few seconds of raw data.
It avoids the interactive browser and instead gives a quick visual check of
signal quality, channel behavior, and overall data shape, which is important for
QC reports.
"""


def make_raw_overview_figure(raw, n_channels=20, duration=10):
    """Create a Matplotlib-only raw overview."""
    sfreq = raw.info["sfreq"]
    start = 0
    stop = int(duration * sfreq)

    data, times = raw.get_data(start=start, stop=stop, return_times=True)
    n = min(n_channels, data.shape[0])
    data = data[:n]
    data_norm = data / np.max(np.abs(data), axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    offset = np.arange(n) * 1.5
    for i in range(n):
        ax.plot(times, data_norm[i] + offset[i])
    ax.set_title(f"Raw Data Overview ({n} channels, {duration}s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels (offset)")
    return fig


"""
We create the MNE Report object and add basic QC components: a simple raw
overview, the power spectral density (PSD) plot, and the sensor layout. These
plots help verify noise levels, channel availability, and general data quality
before deeper analysis.
"""

report = mne.Report(title="QC + Full MNE Report (Clean, No Duplicates)")

fig = make_raw_overview_figure(raw)
report.add_figure(fig, title="Raw Data Overview")
plt.close(fig)

psd = raw.compute_psd()
fig = psd.plot(show=False)
report.add_figure(fig, title="PSD of Raw Data")
plt.close(fig)

fig = raw.plot_sensors(show_names=True)
report.add_figure(fig, title="Sensor Layout")
plt.close(fig)

"""
This block loads event markers and creates epochs (short time windows around
stimulus events). Epochs allow us to check data quality trial-by-trial and
understand how different conditions were recorded, which is essential in QC.
"""

event_id = dict(auditory=1, visual=3)
report.add_events(events, sfreq=raw.info["sfreq"], title="Events Overview")

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=-0.2,
    tmax=0.5,
    baseline=(None, 0),
    preload=True,
    reject=dict(mag=4e-12, eeg=150e-6),
)
report.add_epochs(epochs, title="Epochs Overview")

"""
Here we fit ICA to identify eye-movement components and mark them for removal.
We also add ICA diagnostics to the report, such as component maps and EOG
correlations. ICA is an important step in QC for detecting common artifacts.
"""

ica = ICA(n_components=20, random_state=97)
ica.fit(raw)

eog_epochs = mne.preprocessing.create_eog_epochs(raw)
eog_comps, eog_scores = ica.find_bads_eog(eog_epochs, ch_name="EEG 001")
ica.exclude = eog_comps

report.add_ica(
    ica=ica,
    inst=raw,
    title="ICA Components",
    eog_evoked=eog_epochs.average(),
    eog_scores=eog_scores,
)

"""
This section computes evoked responses (averaged brain activity) for auditory and
visual events and plots topographic maps. It also loads the noise covariance
matrix and any projectors. Together, these help evaluate data cleanliness and
prepare for source analysis.
"""

cov = mne.read_cov(cov_fname)
evokeds = [epochs[k].average() for k in event_id]
report.add_evokeds(
    evokeds, titles=list(event_id.keys()), noise_cov=cov, n_time_points=5
)

common_chs = list(set(cov.ch_names).intersection(raw.ch_names))
cov = cov.copy().pick_channels(common_chs)
report.add_covariance(cov=cov, info=raw.info, title="Noise Covariance")

projs = mne.read_proj(proj_fname)
report.add_projs(info=raw, projs=projs, title="ECG Projectors")

"""
Here we try adding anatomical elements like the MRI surfaces (BEM),
sensor-to-MRI coregistration, and the forward and inverse operators. These steps
link sensor data to the brain and are needed for source estimation. They are
wrapped in try/except so the script still runs even if anatomy files are
missing.
"""

try:
    report.add_bem(subject="sample", subjects_dir=subjects_dir, title="MRI & BEM")
except Exception:
    print("BEM unavailable.")

try:
    report.add_trans(
        trans=trans_fname,
        info=raw,
        subject="sample",
        subjects_dir=subjects_dir,
        title="Coregistration",
    )
except Exception:
    print("Coreg unavailable.")

try:
    report.add_forward(
        forward=fwd_fname,
        title="Forward Solution",
        plot=True,
        subjects_dir=subjects_dir,
    )
except Exception:
    print("Forward unavailable.")

try:
    report.add_inverse_operator(
        inverse_operator=inv_fname,
        title="Inverse Operator",
        plot=True,
        subjects_dir=subjects_dir,
    )
except Exception:
    print("Inverse unavailable.")

"""
This part adds source-level brain activity snapshots to the report. STCs show
how brain signals map onto cortex over time, giving a deeper understanding of
neural responses beyond sensor space.
"""

try:
    stc_path = sample_dir / "sample_audvis-meg"
    report.add_stc(
        stc=stc_path,
        subject="sample",
        subjects_dir=subjects_dir,
        title="Source Estimate",
        n_time_points=2,
    )
except Exception:
    print("STC unavailable.")

"""
A small HTML block is added to show that reports can include instructions,
interpretations, or descriptive text. This is useful for documentation,
explaining results, or guiding collaborators.
"""

html = """
<p>Example hypothesis:</p>
<ol>
<li>Auditory vs Visual differences.</li>
<li>Predicted N1 amplitude change.</li>
</ol>
"""
report.add_html(html, title="Hypothesis")

"""
Finally, we add an MNE logo and save the entire report as an HTML file. The
output is fully interactive, self-contained, and easy to share, combining both
QC and full MNE processing elements in one report.
"""

logo_path = Path(mne.__file__).parent / "icons" / "mne_icon-cropped.png"
report.add_image(logo_path, title="MNE Logo")

report.save("mne_report.html", overwrite=True)
print("Saved as mne_report.html")
