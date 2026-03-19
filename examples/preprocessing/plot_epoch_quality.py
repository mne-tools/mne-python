"""
.. _ex-epoch-quality:

=========================================
Exploring epoch quality before rejection
=========================================

This example shows how to identify potentially artifactual epochs before
calling :meth:`mne.Epochs.drop_bad`. We compute per-epoch outlier scores
from peak-to-peak amplitude, variance, and kurtosis — inspired by FASTER
:footcite:t:`NolanEtAl2010` and :footcite:t:`DelormeEtAl2007` — and use
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
from scipy.stats import kurtosis

import mne
from mne.datasets import eegbci

print(__doc__)

# %%
# Load the EEGBCI dataset and create epochs
# ------------------------------------------
raw_fname = eegbci.load_data(subjects=3, runs=(3,))[0]
raw = mne.io.read_raw(raw_fname, preload=True)
eegbci.standardize(raw)
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage)

events, event_id = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, preload=True, baseline=(None, 0))

# %%
# Compute per-epoch outlier scores
# ---------------------------------
# Peak-to-peak amplitude, variance, and kurtosis are computed per epoch.
# Each feature is z-scored robustly using median absolute deviation (MAD)
# across epochs and averaged into a single outlier score, normalised
# between [0, 1]. Scores close to 1 indicate likely artifacts.

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
# Epochs are ranked from cleanest to noisiest. The dashed lines show two
# example thresholds — demonstrating the quality-quantity trade-off when
# deciding how many epochs to reject.
fig, ax = plt.subplots(layout="constrained")
sorted_idx = np.argsort(scores)
ax.bar(np.arange(len(scores)), scores[sorted_idx], color="steelblue")
ax.axhline(0.8, color="red", linestyle="--", label="Strict threshold (0.8)")
ax.axhline(0.6, color="orange", linestyle="--", label="Lenient threshold (0.6)")
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
# :meth:`mne.Epochs.drop`. The threshold should be adapted based on
# your data and how many epochs you can afford to lose.
for threshold in [0.8, 0.6]:
    bad_epochs = np.where(scores > threshold)[0]
    print(
        f"Threshold {threshold}: {len(bad_epochs)} epochs flagged "
        f"out of {len(epochs)} total"
    )

# %%
# Plot epochs at different thresholds
# -------------------------------------
# The worst-scoring epoch (strict threshold) clearly contains an artifact.
# An epoch from the lenient threshold may look less obvious — illustrating
# why tuning the threshold matters for the quality-quantity trade-off.
worst_idx = np.argmax(scores)
epochs[worst_idx].plot(
    title=f"Strict threshold — worst epoch "
    f"(index {worst_idx}, score={scores[worst_idx]:.2f})",
    scalings=dict(eeg=100e-6),
)

lenient_idx = np.where(scores > 0.6)[0]
lenient_idx = lenient_idx[lenient_idx != worst_idx][0]
epochs[lenient_idx].plot(
    title=f"Lenient threshold — borderline epoch "
    f"(index {lenient_idx}, score={scores[lenient_idx]:.2f})",
    scalings=dict(eeg=100e-6),
)

# %%
# References
# ----------
# .. footbibliography::
