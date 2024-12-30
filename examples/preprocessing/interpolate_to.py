"""
.. _ex-interpolate-to-any-montage:

======================================================
Interpolate EEG data to any montage
======================================================

This example demonstrates how to interpolate EEG channels to match a given montage.
This can be useful for standardizing
EEG channel layouts across different datasets (see :footcite:`MellotEtAl2024`).

- Using the field interpolation for EEG data.
- Using the target montage "biosemi16".

In this example, the data from the original EEG channels will be
interpolated onto the positions defined by the "biosemi16" montage.
"""

# Authors: Antoine Collas <contact@antoinecollas.fr>
# License: BSD-3-Clause

import matplotlib.pyplot as plt

import mne
from mne.channels import make_standard_montage
from mne.datasets import sample

print(__doc__)

# %%
# Load EEG data
data_path = sample.data_path()
eeg_file_path = data_path / "MEG" / "sample" / "sample_audvis-ave.fif"
evoked = mne.read_evokeds(eeg_file_path, condition="Left Auditory", baseline=(None, 0))

# Select only EEG channels
evoked.pick("eeg")

# Plot the original EEG layout
evoked.plot(exclude=[], picks="eeg")

# %%
# Define the target montage
standard_montage = make_standard_montage("biosemi16")

# %%
# Use interpolate_to to project EEG data to the standard montage
evoked_interpolated = evoked.copy().interpolate_to(standard_montage)

# Plot the interpolated EEG layout
evoked_interpolated.plot(exclude=[], picks="eeg")

# %%
# Comparing before and after interpolation
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
evoked.plot(exclude=[], picks="eeg", axes=axs[0], show=False)
axs[0].set_title("Original EEG Layout")
evoked_interpolated.plot(exclude=[], picks="eeg", axes=axs[1], show=False)
axs[1].set_title("Interpolated to Standard 1020 Montage")

# %%
# References
# ----------
# .. footbibliography::
