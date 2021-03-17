"""
.. _tut_working_with_ecog:

======================
Working with ECoG data
======================

MNE supports working with more than just MEG and EEG data. Here we show some
of the functions that can be used to facilitate working with
electrocorticography (ECoG) data.

This example shows how to use:

- ECoG data (`available here <https://openneuro.org/datasets/ds003029>`_)
from an epilepsy patient during a seizure
- channel locations in FreeSurfer's ``fsaverage`` MRI space
- projection onto a pial surface

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
# Here we use a format (iEEG) that is only available in MNE-BIDS 0.7+, so it
# will emit a warning on versions <= 0.6
raw = read_raw_bids(bids_path=bids_path, verbose=False)

# Find the annotated events
events, event_id = mne.events_from_annotations(raw)

# To make the example run much faster, we will start 5 seconds before the
# seizure onset event and use 15 seconds of seizure
onset_events = events[events[:, 2] == event_id['onset']]
start = (onset_events[0, 0] - raw.first_samp) / raw.info['sfreq']
raw.crop(start - 1, start + 3)

# And then downsample. This is just to save time in this example, you should
# not need to do this in general!
raw.resample(200)  # Hz, will also load the data for us

# Then we remove line frequency interference
raw.notch_filter([60], trans_bandwidth=3)

# drop bad channels
raw.drop_channels(raw.info['bads'])

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
    sc = ax.scatter(*xy_pts.T, c=band_power, s=100,
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
paths = ax.scatter(*xy_pts.T, c=np.zeros(len(xy_pts)), s=200,
                   cmap=cmap, vmin=vmin, vmax=vmax)
ax.set_xlim([0, im.shape[0]])
ax.set_ylim([im.shape[1], 0])
fig.colorbar(paths, ax=ax)
ax.set_title('Gamma frequency over time', size='large')

# avoid edge artifacts and decimate, showing just a short chunk
show_power = gamma_power_t
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               fargs=(show_power,),
                               frames=show_power.shape[1],
                               interval=100, blit=True)

###############################################################################
# Now let's animate the raw time series data and add events related
# to seizure onset that are marked in the annotations. This will be done
# very similarly to the high gamma visualization.
#
# As can be seen in the plot below, the seizure originates in the temporal
# lobe and spreads to new electrodes when electrode label names come up in
# the title. The seizure eventually becomes generalized and then subsides.

# colormap to view spectral power - RdBu_r works well for directional
# data such as voltage time series, where positive and negative values
# matter and there is a true 0 point
cmap = 'RdBu_r'

# Get a copy of the filtered data earlier and only get the ecog channels
raw_ecog = raw.copy().pick_types(meg=False, ecog=True)

# Apply a high pass filter to get rid of drift, decrease time range
raw_ecog.filter(l_freq=5, h_freq=None)

# Downsample again, to compute the animation faster
sfreq = 50  # Hz
raw_ecog.resample(sfreq)
ts_data = raw_ecog.get_data()

# recompute events for new sampling frequency
events, event_id = mne.events_from_annotations(raw_ecog)
onset_events = events[events[:, 2] == event_id['onset']]

# invert the event_ids so that we can look up by id and get the name
# to display on the plot
inv_event_id = {v: k for k, v in event_id.items()}

# Use one second before the seizure onset as the animation start
start_sample = int(onset_events[0, 0])


# Create an initialization and animation function to pass to FuncAnimation.
# This time we will also pass the plot title so that can be updated
# with information from the annotations.
def init():
    """Create an empty frame."""
    return paths, title, *tpaths


def animate(i, activity, events):
    """Animate the plot."""
    paths.set_array(activity[:, i])
    tsl = slice(i + start_sample - sfreq, i + start_sample + sfreq)
    for j, tpath in enumerate(tpaths):
        tpath.set_data(range(2 * sfreq), ts_data[j, tsl])
    # If this sample contains an annotation (for a seizure-related behavior)
    # then change the title of the plot
    if i + start_sample in events[:, 0]:
        # Currently this doesn't replace the text, but writes over it.
        # This needs fixing
        event_idx = np.argwhere(events[:, 0] == (i + start_sample))[0][0]
        title.set_text(inv_event_id[events[event_idx, 2]])
        fig.canvas.draw()  # force redrawing
    return paths, title, *tpaths


# create the figure and apply the animation of the
# raw time series data
fig, ax = plt.subplots(figsize=(5, 5))
tax = fig.add_axes([0.125, 0.02, 0.6, 0.15])
ax.imshow(im)
ax.set_axis_off()
tax.set_axis_off()

# We want the mid-point (0 uV) to be white, so we will scale from -vmax to vmax
# so that negative voltages are blue and positive voltages are red
ts_data = raw_ecog.get_data()
vmax = np.percentile(ts_data, 90)
paths = ax.scatter(*xy_pts.T, c=np.zeros(len(xy_pts)), s=100,
                   cmap=cmap, vmin=-vmax, vmax=vmax)
ax.set_xlim([0, im.shape[0]])
ax.set_ylim([im.shape[1], 0])
tsl = slice(start_sample - sfreq, start_sample + sfreq)
tpaths = tax.plot(np.zeros((ts_data.shape[0], 2 * sfreq)).T)
tax.plot([sfreq, sfreq], [ts_data.min(), ts_data.max()], color='k')
fig.colorbar(paths, ax=ax)
title = ax.set_title('iEEG voltage over time', size='large')

# this will be a much longer animation because the seizure is quite long
sl = slice(start_sample, ts_data.shape[1])
show_power = ts_data[:, sl]
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               fargs=(show_power, events),
                               frames=show_power.shape[1] - sfreq,
                               interval=20, blit=True)

###############################################################################
# Alternatively, we can project the sensor data to the nearest locations on
# the pial surface and visualize that:

# sphinx_gallery_thumbnail_number = 5

evoked = mne.EvokedArray(gamma_power_t, raw.info, tmin=raw.times[0])
evoked.resample(10)  # downsample to 10 Hz

src = mne.read_source_spaces(
    op.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
trans = None  # identity transform
stc = mne.stc_near_sensors(evoked, trans, 'fsaverage', src=src,
                           subjects_dir=subjects_dir)
clim = dict(kind='value', lims=[vmin * 0.9, vmin, vmax])
brain = stc.plot(surface='pial', hemi='both',
                 colormap='viridis', clim=clim, views='lat',
                 subjects_dir=subjects_dir, size=(500, 500))
brain.show_view(view=dict(azimuth=-20, elevation=60))

# You can save a movie like the one on our documentation website with:
# brain.save_movie(time_dilation=1, interpolation='linear', framerate=5,
#                  time_viewer=True)
