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

import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.time_frequency import tfr_morlet
from mne.viz import plot_alignment, snapshot_brain_montage

print(__doc__)

# paths to mne datasets - sample ECoG and FreeSurfer subject
misc_path = mne.datasets.misc.data_path()
sample_path = mne.datasets.sample.data_path()

###############################################################################
# Let's load some ECoG electrode locations and names, and turn them into
# a :class:`mne.channels.DigMontage` class.
# First, define a helper function to read in .tsv file.

# read in the electrode coordinates file
# in this tutorial, these are assumed to be in meters
elec_df = pd.read_csv(misc_path + '/ecog/sample_ecog_electrodes.tsv',
                      sep='\t', header=0, index_col=None)
ch_names = elec_df['name'].tolist()
ch_coords = elec_df[['x', 'y', 'z']].to_numpy(dtype=float)
ch_pos = dict(zip(ch_names, ch_coords))

# create montage from channel coordinates in the 'head' coordinate frame
montage = mne.channels.make_dig_montage(ch_pos,
                                        coord_frame='head')

# Now we make a montage stating that the ECoG contacts are in head
# coordinate system (although they are in MRI). This is compensated
# by the fact that below we do not specify a trans file so the Head<->MRI
# transform is the identity.
# montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, elec)),
#                                         coord_frame='head')
print('Created %s channel positions' % len(ch_names))

###############################################################################
# Now that we have our electrode positions in MRI coordinates, we can create
# our measurement info structure.

info = mne.create_info(ch_names, 1000., 'ecog').set_montage(montage)

###############################################################################
# Now that we have our electrode positions in MRI coordinates, we can load in
# our corresponding time-series data. We then compute a time-frequency
# representation of the data (i.e. 1-30, or 30-90 Hz).

# first we'll load in the sample dataset
raw = mne.io.read_raw_edf(misc_path + '/ecog/sample_ecog.edf')

# drop bad channels
raw.info['bads'].extend([ch for ch in raw.ch_names if ch not in ch_names])

# attach montage
raw.set_montage(montage, on_missing='ignore')

# create a 1 Epoch data structure
epoch = mne.EpochsArray(raw.get_data()[np.newaxis, ...],
                        info=raw.info)

# perform gamma band frequency averaged over entire time period
tfr_pwr, _ = tfr_morlet(epoch, freqs=np.linspace(30, 90, 60),
                        n_cycles=2)
gamma_activity = tfr_pwr.data.mean(axis=(1, 2))

# perform low frequency activity averaged over entire time period
tfr_pwr, _ = tfr_morlet(epoch, freqs=np.linspace(8, 12, 4),
                        n_cycles=2)
low_activity = tfr_pwr.data.mean(axis=(1, 2))

###############################################################################
# We can then plot the locations of our electrodes on our subject's brain.
#
# .. note:: These are not real electrodes for this subject, so they
#           do not align to the cortical surface perfectly.

subjects_dir = sample_path + '/subjects'
fig = plot_alignment(info, subject='sample', subjects_dir=subjects_dir,
                     surfaces=['pial'])
mne.viz.set_3d_view(fig, 200, 70)

###############################################################################
# Sometimes it is useful to make a scatterplot for the current figure view.
# This is best accomplished with matplotlib. We can capture an image of the
# current mayavi view, along with the xy position of each electrode, with the
# `snapshot_brain_montage` function. We can visualize say the gamma frequency
# of the ECoG activity on the brain using MNE functions.

# We'll once again plot the surface, then take a snapshot.
fig_scatter = plot_alignment(raw.info, subject='sample',
                             subjects_dir=subjects_dir, surfaces='pial')
mne.viz.set_3d_view(fig_scatter, 200, 70)
xy, im = snapshot_brain_montage(fig_scatter, montage)

# Convert from a dictionary to array to plot
xy_pts = np.vstack([xy[ch] for ch in info['ch_names']])

vmin, vmax = np.percentile(gamma_activity, [10, 90])

# show activity at higher frequencies
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(im)
ax.set_axis_off()
sc = ax.scatter(*xy_pts.T, c=gamma_activity, s=200,
                cmap='viridis', vmin=vmin, vmax=vmax)
ax.set_title("Gamma frequency (30-90 Hz)")
fig.colorbar(sc, ax=ax)

vmin, vmax = np.percentile(low_activity, [10, 90])

# show activity between low frequency
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(im)
ax.set_axis_off()
sc = ax.scatter(*xy_pts.T, c=low_activity, s=200,
                cmap='viridis', vmin=vmin, vmax=vmax)
ax.set_title("Low frequency (0-30 Hz)")
fig.colorbar(sc, ax=ax)

plt.show()
