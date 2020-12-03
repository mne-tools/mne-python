"""
.. _ex-read-neo:

How to use data in neural ensemble (NEO) format
================================================

This example shows how to create an MNE-Python `~mne.io.Raw` object from data
in the `neural ensemble <http://neuralensemble.org/neo/>`__ format. For general
information on creating MNE-Python's data objects from NumPy arrays, see
:ref:`tut_creating_data_structures`.
"""

import neo
import mne

###############################################################################
# This example uses NEO's ``ExampleIO`` object for creating fake data. The data
# will be all zeros, so the plot won't be very interesting; but it should
# demonstrate the steps to using NEO data. For actual data and different file
# formats, consult the NEO documentation.

reader = neo.io.ExampleIO('fakedata.nof')
block = reader.read(lazy=False)[0]  # get the first block
segment = block.segments[0]         # get data from first (and only) segment
signals = segment.analogsignals[0]  # get first (multichannel) signal

data = signals.rescale('V').magnitude.T
sfreq = signals.sampling_rate.magnitude
ch_names = [f'Neo {(idx + 1):02}' for idx in range(signals.shape[1])]
ch_types = ['eeg'] * len(ch_names)  # if not specified, type 'misc' is assumed

info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
raw = mne.io.RawArray(data, info)
raw.plot(show_scrollbars=False)
