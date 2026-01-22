"""
.. _ex-read-mef:

====================================
Reading MEF3 intracranial EEG data
====================================

This example shows how to read MEF3 (.mefd) files using
:func:`mne.io.read_raw_mef`.

MEF3 (Multiscale Electrophysiology Format version 3) is commonly used
for storing intracranial EEG data including sEEG and ECoG recordings.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import mne

# %%
# Reading MEF3 data
# -----------------
#
# MEF3 files are stored in .mefd directories. Use :func:`mne.io.read_raw_mef`
# to read them. Channel types default to sEEG (stereo-EEG).

# Example (requires actual MEF3 data):
# raw = mne.io.read_raw_mef('recording.mefd', preload=True)
# print(raw.info)

# %%
# Setting channel types
# ---------------------
#
# Channel types default to sEEG. For ECoG or other channel types, use
# :meth:`raw.set_channel_types() <mne.io.Raw.set_channel_types>`:

# Example:
# raw.set_channel_types({'CH01': 'ecog', 'CH02': 'ecog'})
# print(raw.get_channel_types())

# %%
# Notes
# -----
#
# - MEF3 data is assumed to be in microvolts (µV) and is automatically
#   converted to volts (V)
# - The pymef package is required: ``pip install pymef``
# - MEF3 sessions can be password-protected (empty password is default)
