"""
.. _tut_working_with_ecog:

======================
Working with ECoG data
======================

MNE supports working with more than just MEG and EEG data. Here we show some
of the functions that can be used to facilitate working with
electrocorticography (ECoG) data.
"""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Chris Holdgraf <choldgraf@gmail.com>
#          Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.viz import plot_alignment, snapshot_brain_montage

print(__doc__)

# paths to mne datasets - sample ECoG and FreeSurfer subject
misc_path = mne.datasets.misc.data_path()
sample_path = mne.datasets.sample.data_path()

###############################################################################
# Let's load some ECoG electrode locations and names, and turn them into
# a :class:`mne.channels.DigMontage` class.
# First, use pandas to read in the .tsv file.

# In this tutorial, the electrode coordinates are assumed to be in meters
elec_df = pd.read_csv(misc_path + '/ecog/sample_ecog_electrodes.tsv',
                      sep='\t', header=0, index_col=None)
ch_names = elec_df['name'].tolist()
ch_coords = elec_df[['x', 'y', 'z']].to_numpy(dtype=float)
ch_pos = dict(zip(ch_names, ch_coords))

# Now we make a :class:`mne.channels.DigMontage` stating that the ECoG
# contacts are in head coordinate system (although they are in MRI). This is
# compensated below by the fact that we do not specify a trans file so the
# Head<->MRI transform is the identity.

montage = mne.channels.make_dig_montage(ch_pos, coord_frame='head')
print('Created %s channel positions' % len(ch_names))

###############################################################################
# Now that we have our montage, we can load in our corresponding
# time-series data and set the montage to the raw data.

# first we'll load in the sample dataset
raw = mne.io.read_raw_edf(misc_path + '/ecog/sample_ecog.edf')

# drop bad channels
raw.info['bads'].extend([ch for ch in raw.ch_names if ch not in ch_names])
raw.load_data()
raw.drop_channels(raw.info['bads'])

# attach montage
raw.set_montage(montage)

###############################################################################
# We then compute the signal power in certain frequency bands (e.g. 30-90 Hz).
# We compute the power in gamma and alpha bands.
gamma_power = np.sum(raw.copy().filter(30, 90).get_data() ** 2, axis=1)
alpha_power = np.sum(raw.copy().filter(8, 12).get_data() ** 2, axis=1)

###############################################################################
# We can then plot the locations of our electrodes on our subject's brain.
#
# .. note:: These are not real electrodes for this subject, so they
#           do not align to the cortical surface perfectly.

subjects_dir = sample_path + '/subjects'
fig = plot_alignment(raw.info, subject='sample', subjects_dir=subjects_dir,
                     surfaces=['pial'])
mne.viz.set_3d_view(fig, 200, 70)

###############################################################################
# Sometimes it is useful to make a scatterplot for the current figure view.
# This is best accomplished with matplotlib. We can capture an image of the
# current mayavi view, along with the xy position of each electrode, with the
# `snapshot_brain_montage` function. We can then visualize for example the
# gamma power on the brain using MNE and matplotlib functions.

# We'll once again plot the surface, then take a snapshot.
fig_scatter = plot_alignment(raw.info, subject='sample',
                             subjects_dir=subjects_dir, surfaces='pial')
mne.viz.set_3d_view(fig_scatter, 200, 70)
xy, im = snapshot_brain_montage(fig_scatter, montage)

# Convert from a dictionary to array to plot
xy_pts = np.vstack([xy[ch] for ch in raw.info['ch_names']])

# colormap to view spectral power
cmap = 'viridis'

# show power at higher frequencies
vmin, vmax = np.percentile(gamma_power, [10, 90])

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(im)
ax.set_axis_off()
sc = ax.scatter(*xy_pts.T, c=gamma_power, s=200,
                cmap=cmap, vmin=vmin, vmax=vmax)
ax.set_title("Gamma frequency (30-90 Hz)")
fig.colorbar(sc, ax=ax)

# show power between low frequency
vmin, vmax = np.percentile(alpha_power, [10, 90])

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(im)
ax.set_axis_off()
sc = ax.scatter(*xy_pts.T, c=alpha_power, s=200,
                cmap=cmap, vmin=vmin, vmax=vmax)
ax.set_title("Alpha frequency (8-12 Hz)")
fig.colorbar(sc, ax=ax)

plt.show()
