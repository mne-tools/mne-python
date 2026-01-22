"""
.. _ex-read-neo:

==============================================
Reading data via Neo (Neural Ensemble Objects)
==============================================

This example shows how to read electrophysiology data using the
`Neo <https://neo.readthedocs.io/>`_ library and convert it into an
MNE-Python `~mne.io.Raw` object.

Neo supports reading from many file formats including Intan, Blackrock,
Axon, Spike2, and more. See the
`Neo IO documentation <https://neo.readthedocs.io/en/stable/rawio.html>`_
for a complete list.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import neo

import mne

# %%
# Reading data with Neo
# ---------------------
#
# Use Neo's IO classes to read data, then convert to an MNE Raw object.
# This example uses Neo's ``ExampleIO`` which generates synthetic data.

reader = neo.io.ExampleIO("fakedata.nof")
block = reader.read(lazy=False)[0]  # get the first block
segment = block.segments[0]  # get data from first (and only) segment
signals = segment.analogsignals[0]  # get first (multichannel) signal

data = signals.rescale("V").magnitude.T
sfreq = signals.sampling_rate.rescale("Hz").magnitude
ch_names = [f"Neo {(idx + 1):02}" for idx in range(signals.shape[1])]
ch_types = ["eeg"] * len(ch_names)  # if not specified, type 'misc' is assumed

info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
raw = mne.io.RawArray(data, info)
print(raw.info)

# %%
# The Raw object works like any other MNE Raw object:

raw.plot(duration=2, show_scrollbars=False)

# %%
# Setting channel types
# ---------------------
#
# Channel types default to EEG. Set appropriate types using
# :meth:`raw.set_channel_types() <mne.io.Raw.set_channel_types>`:

# Example: set first channel to ECG
raw.set_channel_types({raw.ch_names[0]: "ecg"})
print(f"Channel types: {raw.get_channel_types()[:3]}...")

# %%
# Common Neo IO classes
# ---------------------
#
# Here are examples for common file formats:
#
# .. code-block:: python
#
#     # Intan RHD
#     reader = neo.io.IntanIO("data.rhd")
#
#     # Blackrock NSx
#     reader = neo.io.BlackrockIO("recording.ns5")
#
#     # Axon ABF
#     reader = neo.io.AxonIO("recording.abf")
#
# Use the same conversion steps as above after reading data with the
# desired IO class. See the `Neo documentation <https://neo.readthedocs.io/>`_
# for all available IO classes and their requirements.
