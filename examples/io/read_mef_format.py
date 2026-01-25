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
# Channel types default to sEEG. Use :meth:`raw.set_channel_types()
# <mne.io.Raw.set_channel_types>` to set appropriate types:

# Example:
# raw.set_channel_types({'C3': 'eeg', 'C4': 'eeg', 'ECG': 'ecg'})
# print(raw.get_channel_types())

# %%
# Notes
# -----
#
# - Data are scaled to volts using MEF metadata (``units_conversion_factor`` and
#   ``units_description``)
# - The pymef package is required: ``pip install pymef``
# - MEF3 sessions can be password-protected; use ``password=...`` when needed
# - Record metadata are imported as annotations, and discontinuities become
#   ``BAD_ACQ_SKIP`` annotations when present
# - Session time metadata (``recording_time_offset``, ``DST_start_time``,
#   ``DST_end_time``) are stored in a ``MEF_METADATA`` annotation extras
