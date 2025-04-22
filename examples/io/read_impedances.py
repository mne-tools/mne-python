"""
.. _ex-io-impedances:

=================================
Getting impedances from raw files
=================================

Many EEG systems provide impedance measurements for each channel within their file
format. MNE does not parse this information and does not store it in the
:class:`~mne.io.Raw` object. However, it is possible to extract this information from
the raw data and store it in a separate data structure.

ANT Neuro
---------

The ``.cnt`` file format from ANT Neuro stores impedance information in the form of
triggers. The function :func:`mne.io.read_raw_ant` reads this information and marks the
time-segment during which an impedance measurement was performed as
:class:`~mne.Annotations` with the description set in the argument
``impedance_annotation``. However, it doesn't extract the impedance values themselves.
To do so, use the function ``antio.parser.read_triggers``.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from antio import read_cnt
from antio.parser import read_triggers
from matplotlib import pyplot as plt

from mne.datasets import testing
from mne.io import read_raw_ant
from mne.viz import plot_topomap

fname = testing.data_path() / "antio" / "CA_208" / "test_CA_208.cnt"
cnt = read_cnt(fname)
_, _, _, impedances, _ = read_triggers(cnt)

raw = read_raw_ant(fname, eog=r"EOG")
impedances = [{ch: imp[k] for k, ch in enumerate(raw.ch_names)} for imp in impedances]
print(impedances[0])  # impedances measurement at the beginning of the recording

# %%
# Note that the impedance measurement contains all channels, including the bipolar ones.
# We can visualize the impedances on a topographic map; below we show a topography of
# impedances before and after the recording for the EEG channels only.

raw.pick("eeg").set_montage("standard_1020")
impedances = [{ch: imp[ch] for ch in raw.ch_names} for imp in impedances]

f, ax = plt.subplots(1, 2, layout="constrained", figsize=(10, 5))
f.suptitle("Impedances (kOhm)")
impedance = list(impedances[0].values())
plot_topomap(
    impedance,
    raw.info,
    vlim=(0, 50),
    axes=ax[0],
    show=False,
    names=[f"{elt:.1f}" for elt in impedance],
)
ax[0].set_title("Impedances at the beginning of the recording")
impedance = list(impedances[-1].values())
plot_topomap(
    impedance,
    raw.info,
    vlim=(0, 50),
    axes=ax[1],
    show=False,
    names=[f"{elt:.1f}" for elt in impedance],
)
ax[1].set_title("Impedances at the end of the recording")
plt.show()

# %%
# In this very short test file, the impedances are stable over time.
