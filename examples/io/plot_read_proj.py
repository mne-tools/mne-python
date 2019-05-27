
"""
==============================================
Read and visualize projections (SSP and other)
==============================================

This example shows how to read and visualize Signal Subspace Projectors (SSP)
vector. Such projections are sometimes referred to as PCA projections.
"""

# Author: Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

import mne
from mne import read_proj
from mne.io import read_raw_fif

from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

subjects_dir = data_path + '/subjects'
fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
ecg_fname = data_path + '/MEG/sample/sample_audvis_ecg-proj.fif'

###############################################################################
# Load the FIF file and display the projections present in the file. Here the
# projections are added to the file during the acquisition and are obtained
# from empty room recordings.
raw = read_raw_fif(fname)
empty_room_proj = raw.info['projs']

# Display the projections stored in `info['projs']` from the raw object
raw.plot_projs_topomap()

###############################################################################
# Display the projections one by one
fig, axes = plt.subplots(1, len(empty_room_proj))
for proj, ax in zip(empty_room_proj, axes):
    proj.plot_topomap(axes=ax)

###############################################################################
# Use the function in `mne.viz` to display a list of projections
assert isinstance(empty_room_proj, list)
mne.viz.plot_projs_topomap(empty_room_proj)

###############################################################################
# As shown in the tutorial on how to :ref:`tut-viz-raw`
# the ECG projections can be loaded from a file and added to the raw object

# read the projections
ecg_projs = read_proj(ecg_fname)

# add them to raw and plot everything
raw.add_proj(ecg_projs)
raw.plot_projs_topomap()

###############################################################################
# Displaying the projections from a raw object requires no extra information
# since all the layout information is present in `raw.info`.
# MNE is able to automatically determine the layout for some magnetometer and
# gradiometer configurations but not the layout of EEG electrodes.
#
# Here we display the `ecg_projs` individually and we provide extra parameters
# for EEG. (Notice that planar projection refers to the gradiometers and axial
# refers to magnetometers.)
#
# Notice that the conditional is just for illustration purposes. We could
# `raw.info` in all cases to avoid the guesswork in `plot_topomap` and ensure
# that the right layout is always found
fig, axes = plt.subplots(1, len(ecg_projs))
for proj, ax in zip(ecg_projs, axes):
    if proj['desc'].startswith('ECG-eeg'):
        proj.plot_topomap(axes=ax, info=raw.info)
    else:
        proj.plot_topomap(axes=ax)

###############################################################################
# The correct layout or a list of layouts from where to choose can also be
# provided. Just for illustration purposes, here we generate the
# `possible_layouts` from the raw object itself, but it can come from somewhere
# else.
possible_layouts = [mne.find_layout(raw.info, ch_type=ch_type)
                    for ch_type in ('grad', 'mag', 'eeg')]
mne.viz.plot_projs_topomap(ecg_projs, layout=possible_layouts)
