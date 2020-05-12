"""
======================
Working with ECoG data
======================

MNE supports working with more than just MEG and EEG data. Here we show some
of the functions that can be used to facilitate working with
electrocorticography (ECoG) data.
"""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Chris Holdgraf <choldgraf@gmail.com>
# Edited: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from sys import platform as sys_pf
# if sys_pf == 'darwin':
# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib
matplotlib.use("macOSX")
import matplotlib.pyplot as plt
from scipy.io import loadmat

import mne
from mne.viz import plot_alignment, snapshot_brain_montage

print(__doc__)

###############################################################################
# Let's load some ECoG electrode locations and names, and turn them into
# a :class:`mne.channels.DigMontage` class.

mat = loadmat(mne.datasets.misc.data_path() + '/ecog/sample_ecog.mat')
ch_names = mat['ch_names'].tolist()
elec = mat['elec']  # electrode positions given in meters

from mne_bids.tsv_handler import _from_tsv
elec_tsv = _from_tsv(mne.datasets.misc.data_path() + '/ecog/sample_ecog_electrodes.tsv')
ch_names = elec_tsv['name']
ch_coords = np.vstack((elec_tsv['x'], elec_tsv['y'], elec_tsv['z'])).T.astype(float)
ch_pos = dict(zip(ch_names, ch_coords))
montage = mne.channels.make_dig_montage(ch_pos,
                                        coord_frame='head')
print(ch_names)
print(ch_pos)
# Now we make a montage stating that the ECoG contacts are in head
# coordinate system (although they are in MRI). This is compensated
# by the fact that below we do not specicty a trans file so the Head<->MRI
# transform is the identity.
# montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, elec)),
#                                         coord_frame='head')
print('Created %s channel positions' % len(ch_names))

###############################################################################
# Now that we have our electrode positions in MRI coordinates, we can create
# our measurement info structure.

info = mne.create_info(ch_names, 1000., 'ecog').set_montage(montage)

###############################################################################
# We can then plot the locations of our electrodes on our subject's brain.
#
# .. note:: These are not real electrodes for this subject, so they
#           do not align to the cortical surface perfectly.

subjects_dir = mne.datasets.sample.data_path() + '/subjects'
fig = plot_alignment(info, subject='sample', subjects_dir=subjects_dir,
                     surfaces=['pial'])
mne.viz.set_3d_view(fig, 200, 70)

###############################################################################
# Sometimes it is useful to make a scatterplot for the current figure view.
# This is best accomplished with matplotlib. We can capture an image of the
# current mayavi view, along with the xy position of each electrode, with the
# `snapshot_brain_montage` function.

# We'll once again plot the surface, then take a snapshot.
fig_scatter = plot_alignment(info, subject='sample', subjects_dir=subjects_dir,
                             surfaces='pial')
mne.viz.set_3d_view(fig_scatter, 200, 70)
xy, im = snapshot_brain_montage(fig_scatter, montage)

# Convert from a dictionary to array to plot
xy_pts = np.vstack([xy[ch] for ch in info['ch_names']])

# Define an arbitrary "activity" pattern for viz
activity = np.linspace(100, 200, xy_pts.shape[0])

# # This allows us to use matplotlib to create arbitrary 2d scatterplots
_, ax = plt.subplots(figsize=(10, 10))
ax.imshow(im)
ax.scatter(*xy_pts.T, c=activity, s=200, cmap='coolwarm')
ax.set_axis_off()
# plt.show()

###############################################################################
# Sometimes it is useful to create an animation of the ECoG activity over time.
# We can visualize say the gamma frequency of the ECoG activity on the brain
# using MNE functions.

# first we'll load in the sample dataset
raw = mne.io.read_raw_edf(mne.datasets.misc.data_path() + '/ecog/sample_ecog.edf')

# drop bad channels
raw.info['bads'].extend([ch for ch in raw.ch_names if ch not in ch_names])

# attach montage
raw.set_montage(montage, on_missing='warn')

# perform gamma band frequency
epoch = mne.EpochsArray(raw.get_data()[np.newaxis, ...], info=raw.info)
print(epoch)
# print(epoch.shape)
tfr_pwr, tfr_itc = mne.time_frequency.tfr_morlet(epoch, freqs=np.linspace(30, 90, 60),
                                                 n_cycles=3)
print(tfr_pwr)
# Define an arbitrary "activity" pattern for viz
gamma_activity = tfr_pwr.data.mean(axis=(1, 2))

tfr_pwr, tfr_itc = mne.time_frequency.tfr_morlet(epoch, freqs=np.linspace(1, 30, 60),
                                                 n_cycles=3)
low_activity = tfr_pwr.data.mean(axis=(1, 2))


_, ax = plt.subplots(figsize=(10, 10))
# show activity between low frequency and higher frequencies
# We'll once again plot the surface, then take a snapshot.
fig_scatter = plot_alignment(raw.info, subject='sample', subjects_dir=subjects_dir,
                             surfaces='pial')
mne.viz.set_3d_view(fig_scatter, 200, 70)
xy, im = snapshot_brain_montage(fig_scatter, montage)
# Convert from a dictionary to array to plot
xy_pts = np.vstack([xy[ch] for ch in info['ch_names']])

ax.imshow(im)
ax.set_axis_off()
ax.scatter(*xy_pts.T, c=gamma_activity, s=200, cmap='coolwarm')
# ax.set_title("Gamma frequency (30-90 Hz)")

_, ax = plt.subplots(figsize=(10, 10))
# show activity between low frequency and higher frequencies
# We'll once again plot the surface, then take a snapshot.
fig_scatter = plot_alignment(raw.info, subject='sample', subjects_dir=subjects_dir,
                             surfaces='pial')
mne.viz.set_3d_view(fig_scatter, 200, 70)
xy, im = snapshot_brain_montage(fig_scatter, montage)
# Convert from a dictionary to array to plot
xy_pts = np.vstack([xy[ch] for ch in info['ch_names']])
ax.imshow(im)
ax.set_axis_off()
ax.scatter(*xy_pts.T, c=low_activity, s=200, cmap='coolwarm')
# ax.set_title("Low frequency (0-30 Hz)")

plt.show()
