"""
=====================================
Plot grouped triaxial OPM topomaps
=====================================

This example demonstrates grouped radial/tangential topomap rendering for
colocated triaxial OPM sensors. The grouped rendering places radial maps
alongside tangential maps so orientation information is explicit.
"""
# Authors: MNE contributors
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

import mne

# Create synthetic triaxial OPM data
# Simulate three colocated OPM sensors with radial and tangential orientations
n_channels = 6  # 3 locations × 2 orientations (radial + tangential)
n_samples = 500
sfreq = 100.0

# Create synthetic channel info with triaxial OPM layout
ch_names = [f"G{i // 2}-{['RAD', 'TAN'][i % 2]}" for i in range(n_channels)]
ch_types = ["mag"] * n_channels
info = mne.create_info(ch_names, sfreq, ch_types)

# Set channel locations in a line to simulate colocated triplets
locs = np.array([[0, i // 2 * 0.01, 0] for i in range(n_channels)])
info.set_montage(
    mne.channels.make_dig_montage(
        ch_pos={name: loc for name, loc in zip(ch_names, locs)}
    )
)

# Create synthetic data
data = np.random.randn(n_channels, n_samples) * 1e-12  # Tesla
data[::2] += (
    np.sin(2 * np.pi * 10 * np.arange(n_samples) / sfreq) * 1e-12
)  # 10 Hz signal

# Create evoked object
evoked = mne.EvokedArray(data, info, tmin=-0.5)

# Plot grouped topomap showing radial and tangential maps side-by-side
fig = evoked.plot_topomap(times=[0.0], ch_type="mag")
