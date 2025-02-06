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
# Copyright the MNE-Python contributors.

import matplotlib.pyplot as plt

import mne
from mne.channels import make_standard_montage
from mne.datasets import sample

print(__doc__)
ylim = (-10, 10)

# %%
# Load EEG data
data_path = sample.data_path()
eeg_file_path = data_path / "MEG" / "sample" / "sample_audvis-ave.fif"
evoked = mne.read_evokeds(eeg_file_path, condition="Left Auditory", baseline=(None, 0))

# Select only EEG channels
evoked.pick("eeg")

# Plot the original EEG layout
evoked.plot(exclude=[], picks="eeg", ylim=dict(eeg=ylim))

# %%
# Define the target montage
standard_montage = make_standard_montage("biosemi16")

# %%
# Use interpolate_to to project EEG data to the standard montage
evoked_interpolated_spline = evoked.copy().interpolate_to(
    standard_montage, method="spline"
)

# Plot the interpolated EEG layout
evoked_interpolated_spline.plot(exclude=[], picks="eeg", ylim=dict(eeg=ylim))

# %%
# Use interpolate_to to project EEG data to the standard montage
evoked_interpolated_mne = evoked.copy().interpolate_to(standard_montage, method="MNE")

# Plot the interpolated EEG layout
evoked_interpolated_mne.plot(exclude=[], picks="eeg", ylim=dict(eeg=ylim))

# %%
# Comparing before and after interpolation
fig, axs = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)
evoked.plot(exclude=[], picks="eeg", axes=axs[0], show=False, ylim=dict(eeg=ylim))
axs[0].set_title("Original EEG Layout")
evoked_interpolated_spline.plot(
    exclude=[], picks="eeg", axes=axs[1], show=False, ylim=dict(eeg=ylim)
)
axs[1].set_title("Interpolated to Standard 1020 Montage using spline interpolation")
evoked_interpolated_mne.plot(
    exclude=[], picks="eeg", axes=axs[2], show=False, ylim=dict(eeg=ylim)
)
axs[2].set_title("Interpolated to Standard 1020 Montage using MNE interpolation")

# %%
# References
# ----------
# .. footbibliography::
