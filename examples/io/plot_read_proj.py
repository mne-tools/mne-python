
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
n_cols = len(empty_room_proj)
fig, axes = plt.subplots(1, n_cols, figsize=(2 * n_cols, 2))
for proj, ax in zip(empty_room_proj, axes):
    proj.plot_topomap(axes=ax, info=raw.info)

###############################################################################
# Use the function in `mne.viz` to display a list of projections
assert isinstance(empty_room_proj, list)
mne.viz.plot_projs_topomap(empty_room_proj, info=raw.info)

###############################################################################
# .. TODO: add this when the tutorial is up: "As shown in the tutorial
#    :doc:`../auto_tutorials/preprocessing/plot_projectors`, ..."
#
# The ECG projections can be loaded from a file and added to the raw object

# read the projections
ecg_projs = read_proj(ecg_fname)

# add them to raw and plot everything
raw.add_proj(ecg_projs)
raw.plot_projs_topomap()
