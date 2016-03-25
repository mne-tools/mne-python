"""
===========================================================
Plot sensor denoising using oversampled temporal projection
===========================================================

This demonstrates denoising using the OTP algorithm on a dataset
with noise arising predominantly from sensors (not the environment).
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import mne

print(__doc__)

fname_raw = op.join(mne.datasets.multimodal.data_path(), 'multimodal_raw.fif')

###############################################################################
# Read raw file and crop for demo purposes
raw = mne.io.read_raw_fif(fname_raw).crop(4, 14).load_data()
events = mne.find_events(raw, 'STI 014')

###############################################################################
# Process using OTP

raw_clean = mne.preprocessing.oversampled_temporal_projection(
    raw, duration=10., verbose=True)

###############################################################################
# Plot the resulting data

raw.plot(events=events)
raw_clean.plot(events=events)
