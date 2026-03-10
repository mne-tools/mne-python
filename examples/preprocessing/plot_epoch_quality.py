"""
.. _ex-epoch-quality:

=====================================
Exploring epoch quality before rejection
=====================================

Before rejecting epochs with :meth:`mne.Epochs.drop_bad`, it can be useful
to get a sense of which epochs are the most likely artifacts. This example
shows how to compute simple per-epoch statistics — peak-to-peak amplitude,
variance, and kurtosis — and use them to rank epochs by their outlier score.

The approach is inspired by established methods in the EEG artifact detection
literature, namely FASTER :footcite:t:`NolanEtAl2010` and
:footcite:t:`DelormeEtAl2007`, both of which use z-scored kurtosis and
variance across epochs to flag bad trials.
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
# Compute per-epoch statistics
# We compute three features for each epoch:
# - Peak-to-peak amplitude (sensitive to large jumps)
# - Variance (sensitive to sustained high-amplitude noise)
# - Kurtosis (sensitive to spike artifacts)
#
# Each feature is z-scored robustly using median absolute deviation (MAD)
# across epochs, then averaged into a single outlier score per epoch.

data = epochs.get_data()  # (n_epochs, n_channels, n_times)

# Feature 1: peak-to-peak
ptp = np.ptp(data, axis=-1).mean(axis=-1)

# Feature 2: variance
var = data.var(axis=-1).mean(axis=-1)

# Feature 3: kurtosis
from scipy.stats import kurtosis  # noqa: E402

kurt = np.array([kurtosis(data[i].ravel()) for i in range(len(data))])

# Robust z-score using MAD
features = np.column_stack([ptp, var, kurt])
median = np.median(features, axis=0)
mad = np.median(np.abs(features - median), axis=0) + 1e-10
z = np.abs((features - median) / mad)

# Normalize to [0, 1]
raw_score = z.mean(axis=-1)
scores = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-10)

# %%
# Plot the scores ranked from cleanest to noisiest
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
# Inspect the worst epochs
# Epochs scoring above 0.8 are worth inspecting manually
bad_epochs = np.where(scores > 0.8)[0]
print(f"Epochs worth inspecting: {bad_epochs}")
print(f"That's {len(bad_epochs)} out of {len(epochs)} total epochs")

# %%
# References
# ----------
# .. footbibliography::
