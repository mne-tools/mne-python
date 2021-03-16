"""
.. _tut_working_with_ecog:

======================
Working with ECoG data
======================

MNE supports working with more than just MEG and EEG data. Here we show some
of the functions that can be used to facilitate working with
electrocorticography (ECoG) data.

This example shows how to use:

- ECoG data
- channel locations in subject's MRI space
- projection onto a surface

For a complementary example that involves sEEG data, channel locations in
MNI space, or projection into a volume, see :ref:`tut_working_with_seeg`.
"""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Chris Holdgraf <choldgraf@gmail.com>
#          Adam Li <adam2392@gmail.com>
#          Alex Rockhill <aprockhill@mailbox.org>
#          Liberty Hamilton <libertyhamilton@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mne_bids import BIDSPath, read_raw_bids

import mne
from mne.viz import plot_alignment, snapshot_brain_montage

print(__doc__)

# paths to mne datasets - sample ECoG and FreeSurfer subject
bids_root = mne.datasets.epilepsy_ecog.data_path()
sample_path = mne.datasets.sample.data_path()
subjects_dir = op.join(sample_path, 'subjects')


###############################################################################
# Let's load some ECoG electrode data with ``mne-bids``.
# first define the bids path
bids_path = BIDSPath(root=bids_root, subject='pt1', session='presurgery',
                     task='ictal', datatype='ieeg', extension='vhdr')

# then we'll use it to load in the sample dataset
# XXX: RuntimeWarning: iEEG Coordinate frame is not accepted BIDS keyword.
# The allowed keywords are: ['acpc', 'pixels', 'other']
with warnings.catch_warnings(record=True):
    raw = read_raw_bids(bids_path=bids_path, verbose=False)

# load data and drop bad channels
raw.load_data()
raw.drop_channels(raw.info['bads'])

# get the montage
montage = raw.get_montage()

###############################################################################
# We can then plot the locations of our electrodes on the fsaverage brain.
# We'll use :func:`~mne.viz.snapshot_brain_montage` to save the plot as image
# data (along with xy positions of each electrode in the image), so that later
# we can plot frequency band power on top of it.

fig = plot_alignment(raw.info, subject='fsaverage', subjects_dir=subjects_dir,
                     surfaces=['pial'])
mne.viz.set_3d_view(fig, 160, -70, focalpoint=[0.067, -0.040, 0.018])

xy, im = snapshot_brain_montage(fig, raw.info)

###############################################################################
# Next, we'll compute the signal power in the gamma (30-90 Hz) and alpha
# (8-12 Hz) bands.
gamma_power_t = raw.copy().filter(30, 90).apply_hilbert(
    envelope=True).get_data()
alpha_power_t = raw.copy().filter(8, 12).apply_hilbert(
    envelope=True).get_data()
gamma_power = gamma_power_t.mean(axis=-1)
alpha_power = alpha_power_t.mean(axis=-1)

###############################################################################
# Now let's use matplotlib to overplot frequency band power onto the electrodes
# which can be plotted on top of the brain from
# :func:`~mne.viz.snapshot_brain_montage`.

# Convert from a dictionary to array to plot
xy_pts = np.vstack([xy[ch] for ch in raw.info['ch_names']])

# colormap to view spectral power
cmap = 'viridis'

# Create a 1x2 figure showing the average power in gamma and alpha bands.
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
# choose a colormap range wide enough for both frequency bands
_gamma_alpha_power = np.concatenate((gamma_power, alpha_power)).flatten()
vmin, vmax = np.percentile(_gamma_alpha_power, [10, 90])
for ax, band_power, band in zip(axs,
                                [gamma_power, alpha_power],
                                ['Gamma', 'Alpha']):
    ax.imshow(im)
    ax.set_axis_off()
    sc = ax.scatter(*xy_pts.T, c=band_power, s=200,
                    cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'{band} band power', size='x-large')
    ax.set_xlim([0, im.shape[0]])
    ax.set_ylim([im.shape[1], 0])
fig.colorbar(sc, ax=axs)

###############################################################################
# Say we want to visualize the evolution of the power in the gamma band,
# instead of just plotting the average. We can use
# `matplotlib.animation.FuncAnimation` to create an animation and apply this
# to the brain figure.


# create an initialization and animation function
# to pass to FuncAnimation
def init():
    """Create an empty frame."""
    return paths,


def animate(i, activity):
    """Animate the plot."""
    paths.set_array(activity[:, i])
    return paths,


# create the figure and apply the animation of the
# gamma frequency band activity
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(im)
ax.set_axis_off()
vmin, vmax = np.percentile(gamma_power, [10, 90])
paths = ax.scatter(*xy_pts.T, c=np.zeros(len(xy_pts)), s=100,
                   cmap=cmap, vmin=vmin, vmax=vmax)
ax.set_xlim([0, im.shape[0]])
ax.set_ylim([im.shape[1], 0])
fig.colorbar(paths, ax=ax)
ax.set_title('Gamma frequency over time\n(Hilbert transform)',
             size='large')

# avoid edge artifacts and decimate, showing just a short chunk
sl = slice(100, 150)
show_power = gamma_power_t[:, sl]
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               fargs=(show_power,),
                               frames=show_power.shape[1],
                               interval=100, blit=True)

###############################################################################
# Now let's animate the raw time series data and add events related
# to seizure onset that are marked in the annotations. This will be done
# very similarly to the high gamma visualization

# colormap to view spectral power - RdBu_r works well for directional
# data such as voltage time series, where positive and negative values
# matter and there is a true 0 point.
cmap = 'RdBu_r'

raw_notched = raw.copy().notch_filter([60, 120])
ts_data = raw_notched.get_data()

# Find the annotated events
events, event_id = mne.events_from_annotations(raw)

# Let's use the third event, which is the start of the first siezure
# (the first two events are pre-seizure behavioral events)
start_sample = int(events[2, 0] - 1 * raw.info['sfreq'])


# Create an initialization and animation function
# to pass to FuncAnimation. This time we will
# also pass the plot title so that can be updated
# with information from the annotations.
def init():
    """Create an empty frame."""
    return paths, title


def animate(i, activity, events):
    """Animate the plot."""
    paths.set_array(activity[:, i])
    # If this sample contains an annotation (for a seizure-related behavior)
    # then change the title of the plot
    if i + start_sample in events.keys():
        # Currently this doesn't replace the text, but writes over it.
        # This needs fixing
        title.set_text(events[i + start_sample])

    return paths, title


# create the figure and apply the animation of the
# raw time series data
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(im)
ax.set_axis_off()
# We want the mid-point (0 uV) to be white, so we will scale from -vmax to vmax
# so that negative voltages are blue and positive voltages are red
vmax = np.percentile(ts_data.flatten(), 90)
paths = ax.scatter(*xy_pts.T, c=np.zeros(len(xy_pts)), s=40,
                   cmap=cmap, vmin=-vmax, vmax=vmax)
ax.set_xlim([0, im.shape[0]])
ax.set_ylim([im.shape[1], 0])
fig.colorbar(paths, ax=ax)
title = ax.set_title('iEEG voltage over time', size='large')


# this will be a much longer animation, but the seizure is also
# quite long..
sl = slice(start_sample, ts_data.shape[1])
show_power = ts_data[:, sl]
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               fargs=(show_power, events),
                               frames=show_power.shape[1],
                               interval=1, blit=True)


###############################################################################
# Alternatively, we can project the sensor data to the nearest locations on
# the pial surface and visualize that:

# sphinx_gallery_thumbnail_number = 4

evoked = mne.EvokedArray(
    gamma_power_t[:, sl], raw.info, tmin=raw.times[sl][0])
src = mne.read_source_spaces(
    op.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
trans = None  # identity transform
stc = mne.stc_near_sensors(evoked, trans, 'fsaverage', src=src,
                           subjects_dir=subjects_dir)
clim = dict(kind='value', lims=[vmin * 0.9, vmin, vmax])
brain = stc.plot(surface='pial', hemi='both', initial_time=0.68,
                 colormap='viridis', clim=clim, views='parietal',
                 subjects_dir=subjects_dir, size=(500, 500))

# You can save a movie like the one on our documentation website with:
# brain.save_movie(time_dilation=50, interpolation='linear', framerate=10,
#                  time_viewer=True)
