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

from collections import OrderedDict
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


def _from_tsv(fname):
    """Read a tsv file into an OrderedDict.


    Parameters
    ----------
    fname : str
        Path to the file being loaded.

    Returns
    -------
    data_dict : collections.OrderedDict
        Keys are the column names, and values are the column data.

    """
    data = np.loadtxt(fname, dtype=str, delimiter='\t',
                      comments=None, encoding='utf-8')
    column_names = data[0, :]
    info = data[1:, :]
    data_dict = OrderedDict()
    dtypes = [str] * info.shape[1]
    for i, name in enumerate(column_names):
        data_dict[name] = info[:, i].astype(dtypes[i]).tolist()
    return data_dict


# read in the electrode coordinates file
# in this tutorial, these are assumed to be in meters
elec_tsv = _from_tsv(misc_path + '/ecog/sample_ecog_electrodes.tsv')
ch_names = elec_tsv['name']
ch_coords = np.vstack((elec_tsv['x'],
                       elec_tsv['y'],
                       elec_tsv['z'])).T.astype(float)
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
