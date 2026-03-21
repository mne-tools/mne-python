"""
======================
Reading BCI2000 files
======================

In this example, we use MNE-Python to read a BCI2000 ``.dat`` file.
BCI2000 is a general-purpose brain-computer interface (BCI) system widely
used in EEG research. The file is downloaded from the MNE testing data
repository using :mod:`pooch`.

"""  # noqa: D205 D400

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import pooch

import mne

# %%
# First, we download the sample BCI2000 ``.dat`` file using :mod:`pooch`.

fname = pooch.retrieve(
    url="https://raw.githubusercontent.com/mne-tools/mne-testing-data/master/BCI2k/bci2k_test.dat",
    known_hash="sha256:8efc7b5f700660a044086cb1449806ca408c2e6d32d9338c32e1bf31ce3ca9cb",
)

# %%
# Now we can read the file using :func:`mne.io.read_raw_bci2k`.
# Note that ``preload=True`` is required for BCI2000 files.

raw = mne.io.read_raw_bci2k(fname, preload=True)
print(raw.info)

# %%
# We can inspect the object representation, channel names, types, sampling
# frequency, and recording duration.

print(raw)
print(f"Channel names : {raw.ch_names}")
print(f"Channel types : {raw.get_channel_types()}")
print(f"Sampling freq : {raw.info['sfreq']} Hz")
print(f"Duration      : {raw.times[-1]:.2f} s")
print(f"n_channels    : {raw.info['nchan']}")
print(f"Data shape    : {raw.get_data().shape}  (n_channels, n_samples)")

# %%
# If the BCI2000 file contains a ``StimulusCode`` state, it is automatically
# mapped to a ``STI 014`` stim channel. We can extract events from it using
# :func:`mne.find_events`.

if "STI 014" in raw.ch_names:
    events = mne.find_events(raw, shortest_event=1)
    print(f"Found {len(events)} events")
    print(mne.count_events(events))
else:
    print("No stim channel found in this file.")

# %%
# Finally, we can visualize the raw data.

raw.plot(duration=5, n_channels=len(raw.ch_names), scalings="auto")
