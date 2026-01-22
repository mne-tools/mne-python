"""
.. _ex-read-neo:

==============================================
Reading data via Neo (Neural Ensemble Objects)
==============================================

This example shows how to read electrophysiology data using the
`Neo <https://neo.readthedocs.io/>`_ library via :func:`mne.io.read_raw_neo`.

Neo supports reading from many file formats including Micromed TRC,
Intan, Blackrock, Axon, Spike2, and more. See the
`Neo IO documentation <https://neo.readthedocs.io/en/stable/rawio.html>`_
for a complete list.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import tempfile
from pathlib import Path

import mne

# %%
# Reading data with Neo
# ---------------------
#
# Use :func:`mne.io.read_raw_neo` with the appropriate ``neo_io_class``
# parameter. This example uses Neo's ``ExampleIO`` which generates
# synthetic data.

temp_dir = Path(tempfile.mkdtemp())
example_fname = temp_dir / "example_data.nof"
example_fname.touch()  # ExampleIO generates fake data but MNE requires file to exist

raw = mne.io.read_raw_neo(example_fname, neo_io_class="ExampleIO", preload=True)
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
#     # Micromed TRC (iEEG)
#     raw = mne.io.read_raw_neo("recording.trc", neo_io_class="MicromedIO")
#
#     # Intan RHD
#     raw = mne.io.read_raw_neo("data.rhd", neo_io_class="IntanIO")
#
#     # Blackrock NSx
#     raw = mne.io.read_raw_neo("recording.ns5", neo_io_class="BlackrockIO")
#
#     # Axon ABF
#     raw = mne.io.read_raw_neo("recording.abf", neo_io_class="AxonIO")
#
# See the `Neo documentation <https://neo.readthedocs.io/>`_ for all
# available IO classes and their requirements.
