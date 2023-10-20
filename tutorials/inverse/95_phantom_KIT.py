"""
.. _tut-phantom-kit:

============================
KIT phantom dataset tutorial
============================

Here we read KIT data TODO EXPLAIN.
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

# %%
import mne

data_path = mne.datasets.phantom_kit.data_path()

raw = mne.io.read_raw_kit(data_path / "002_phantom_11Hz_100uA.con")
raw.plot()
