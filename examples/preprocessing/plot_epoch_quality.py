"""
.. _ex-epoch-quality:

=========================================
Exploring epoch quality before rejection
=========================================

This example shows how to identify potentially artifactual epochs before
calling :meth:`mne.Epochs.drop_bad`. We compute per-epoch outlier scores
from peak-to-peak amplitude, variance, and kurtosis - inspired by FASTER
:footcite:t:`NolanEtAl2010` and :footcite:t:`DelormeEtAl2007` - and use
them to rank epochs from cleanest to noisiest before making any rejection
decisions.
"""
# Authors: Aman Srivastava
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%
import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

# %%
# Load the sample dataset and create epochs
# ------------------------------------------
meg_path = data_path / "MEG" / "sample"
raw_fname = meg_path / "sample_audvis_filt-0-40_raw.fif"

raw = mne.io.read_raw_fif(raw_fname, preload=True)
events = mne.find_events(raw, "STI 014")

event_id = {"auditory/left": 1, "auditory/right": 2}
tmin, tmax = -0.2, 0.5
picks = mne.pick_types(raw.info, meg="grad", eeg=False)

epochs = mne.Epochs(
    raw, events, event_id, tmin, tmax, picks=picks, preload=True, baseline=(None, 0)
)

# %%
# Compute per-epoch outlier scores
# ---------------------------------
# Peak-to-peak amplitude, variance, and kurtosis are computed per epoch.
# Each feature is z-scored robustly using median absolute deviation (MAD)
# across epochs and averaged into a single outlier score, normalised
# between [0, 1]. Scores close to 1 indicate likely artifacts.

from scipy.stats import kurtosis  # noqa: E402

data = epochs.get_data()  # (n_epochs, n_channels, n_times)

ptp = np.ptp(data, axis=-1).mean(axis=-1)
var = data.var(axis=-1).mean(axis=-1)
kurt = np.array([kurtosis(data[i].ravel()) for i in range(len(data))])

features = np.column_stack([ptp, var, kurt])
median = np.median(features, axis=0)
mad = np.median(np.abs(features - median), axis=0) + 1e-10
z = np.abs((features - median) / mad)

raw_score = z.mean(axis=-1)
scores = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-10)

# %%
# Plot epoch quality scores
# --------------------------
# Epochs are ranked from cleanest to noisiest. The dashed red line shows
# an example threshold - epochs above it are candidates for rejection or
# manual inspection.
fig, ax = plt.subplots(layout="constrained")
sorted_idx = np.argsort(scores)
ax.bar(np.arange(len(scores)), scores[sorted_idx], color="steelblue")
ax.axhline(0.8, color="red", linestyle="--", label="Example threshold (0.8)")
ax.set(
    xlabel="Epoch (sorted by score)",
    ylabel="Outlier score",
    title="Epoch quality scores (0 = clean, 1 = likely artifact)",
)
ax.legend()

# %%
# Identify and handle suspicious epochs
# ---------------------------------------
# Epochs scoring above the threshold can be inspected visually using
# :meth:`mne.Epochs.plot`, or dropped directly using
# :meth:`mne.Epochs.drop`. The threshold of 0.8 is chosen here for
# illustration - users should adapt it based on their data and how
# many epochs they can afford to lose.
bad_epochs = np.where(scores > 0.8)[0]
print(f"Epochs worth inspecting: {bad_epochs}")
print(f"That's {len(bad_epochs)} out of {len(epochs)} total epochs")

# To drop these epochs directly:
# epochs.drop(bad_epochs, reason="quality-score")

# %%
# References
# ----------
# .. footbibliography::
