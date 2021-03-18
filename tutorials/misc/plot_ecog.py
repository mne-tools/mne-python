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
# Load in data and perform basic preprocessing
# --------------------------------------------
#
# Let's load some ECoG electrode data with `mne-bids
# <https://mne.tools/mne-bids/>`_.

# first define the bids path
bids_path = BIDSPath(root=bids_root, subject='pt1', session='presurgery',
                     task='ictal', datatype='ieeg', extension='vhdr')

# then we'll use it to load in the sample dataset
# Here we use a format (iEEG) that is only available in MNE-BIDS 0.7+, so it
# will emit a warning on versions <= 0.6
raw = read_raw_bids(bids_path=bids_path, verbose=False)

# Find the annotated events
events, event_id = mne.events_from_annotations(raw)

# To make the example run much faster, we will start 1 seconds before the
# seizure onset event and use 4 seconds of seizure
onset_events = events[events[:, 2] == event_id['onset']]
start = (onset_events[0, 0] - raw.first_samp) / raw.info['sfreq']
raw.crop(start - 0.5, start + 3.5)

# .. note:
# And then downsample. This is just to save time in this example, you should
# not need to do this in general!
raw.resample(200)  # Hz, will also load the data for us

# Then we remove line frequency interference
raw.notch_filter([60], trans_bandwidth=3)

# drop bad channels
raw.drop_channels(raw.info['bads'])

# the coordinate frame of the montage
print(raw.get_montage().get_positions()['coord_frame'])

###############################################################################
# Explore the electrodes on a template brain
# ------------------------------------------
#
# Our electrodes are shown in ``mni_tal`` space, which corresponds to
# the fsaverage brain. We can then plot the locations of our electrodes
# on the fsaverage brain. We'll use :func:`~mne.viz.snapshot_brain_montage`
# to save the plot as image data (along with xy positions of each electrode
# in the image), so that later we can plot frequency band power on top of it.

fig = plot_alignment(raw.info, subject='fsaverage', subjects_dir=subjects_dir,
                     surfaces=['pial'], coord_frame='mri')
az, el, focalpoint = 160, -70, [0.067, -0.040, 0.018]
mne.viz.set_3d_view(fig, azimuth=az, elevation=el, focalpoint=focalpoint)

xy, im = snapshot_brain_montage(fig, raw.info)

###############################################################################
# Compute frequency features of the data
# --------------------------------------
#
# Next, we'll compute the signal power in the gamma (30-90 Hz) and alpha
# (8-12 Hz) bands, downsampling the result to 10 Hz (to save time).

sfreq = 10
gamma_power_t = raw.copy().filter(30, 90).apply_hilbert(
    envelope=True).resample(sfreq)
gamma_info = gamma_power_t.info
gamma_power_t = gamma_power_t.get_data()
alpha_power_t = raw.copy().filter(8, 12).apply_hilbert(
    envelope=True).resample(sfreq).get_data()

# we compute the mean power over time
gamma_power = gamma_power_t.mean(axis=-1)
alpha_power = alpha_power_t.mean(axis=-1)

###############################################################################
# Overlay the mean gamma and alpha band power on the brain
# --------------------------------------------------------
#
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

    # show colorbar for each frequency band
    plt.colorbar(im, ax=ax)

###############################################################################
# Visualize the time-evolution of the gamma power on the brain
# ------------------------------------------------------------
#
# Say we want to visualize the evolution of the power in the gamma band,
# instead of just plotting the average. We can use
# `matplotlib.animation.FuncAnimation` to create an animation and apply this
# to the brain figure.

# create the figure and apply the animation of the gamma band activity
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(im)
ax.set_axis_off()
vmin, vmax = np.percentile(gamma_power, [50, 99])
paths = ax.scatter(*xy_pts.T, c=np.zeros(len(xy_pts)), s=50,
                   cmap=cmap, vmin=vmin, vmax=vmax)
ax.set_xlim([0, im.shape[0]])
ax.set_ylim([im.shape[1], 0])
fig.colorbar(paths, ax=ax)
ax.set_title('Gamma frequency over time', size='large')
t_text = ax.text(0.5, 0, 't=0', animated=True, ha='center', va='bottom',
                 transform=ax.transAxes)
artists = [paths, t_text]


def animate_gamma(i):
    """Animate the plot."""
    paths.set_array(gamma_power_t[:, i])
    t_text.set_text(f't={raw.first_time + i / sfreq:0.3f} sec')
    return artists


anim = animation.FuncAnimation(fig, animate_gamma, init_func=lambda: artists,
                               frames=gamma_power_t.shape[1], blit=True,
                               interval=1000 / sfreq)

###############################################################################
# Visualize the raw activity over time relative to seizure onset
# --------------------------------------------------------------
#
# Now let's animate the raw time series data and add events related
# to seizure onset that are marked in the annotations. This will be done
# very similarly to the high gamma visualization.
#
# As can be seen in the plot below, the seizure originates in the temporal
# lobe and spreads to new electrodes when electrode label names come up in
# the title. The seizure eventually becomes generalized and then subsides.
# As mentioned earlier, we sub-sample and only show a small portion of
# the seizure for demonstration purposes. Feel free to run this example
# to visualize the spread of the seizure yourself, specifically in
# channels annotated in the ``events`` array.

# colormap to view spectral power - RdBu_r works well for directional
# data such as voltage time series, where positive and negative values
# matter and there is a true 0 point
cmap = 'RdBu_r'

# Get a copy of the filtered data earlier and only get the ecog channels
raw.pick_types(ecog=True)

# Apply a high pass filter to get rid of drift, decrease time range
raw.filter(l_freq=2, h_freq=None)

# Downsample again, to compute the animation faster
sfreq = 40  # Hz
raw.resample(sfreq)
ts_data = raw.get_data() * 1e6  # ->μV

# recompute events for new sampling frequency
events, event_id = mne.events_from_annotations(raw)
inv_event_id = {v: k for k, v in event_id.items()}

# create the figure and apply the animation of the
# raw time series data
fig = plt.figure(figsize=(5, 5))
ax = fig.add_axes([0.025, 0.2, 0.78, 0.78])
tax = fig.add_axes([0.025, 0.02, 0.78, 0.15])
cax = fig.add_axes([0.82, 0.2, 0.05, 0.78])
ax.imshow(im)
ax.set_axis_off()
tax.set_axis_off()

# We want the mid-point (0 uV) to be white, so we will scale from -vmax to vmax
# so that negative voltages are blue and positive voltages are red
raw_vmax = np.percentile(np.abs(ts_data), 90)
paths = ax.scatter(*xy_pts.T, c=np.zeros(len(xy_pts)), s=100,
                   cmap=cmap, vmin=-raw_vmax, vmax=raw_vmax)
ax.set_xlim([0, im.shape[0]])
ax.set_ylim([im.shape[1], 0])
half_width = sfreq // 2  # show one second of data, with time point centered
tpaths = tax.plot(
    np.zeros((ts_data.shape[0], 2 * half_width + 1)).T,
    animated=True, lw=0.5, color='k', alpha=0.5)
artists = [paths] + list(tpaths)
artists.append(tax.axvline(half_width, color='k', animated=True))
title = ax.text(0.5, 1, 'iEEG voltage over time', animated=True,
                transform=ax.transAxes, ha='center', va='top')
t_text = tax.text(0.5, 0, 't=0', animated=True, ha='left', va='bottom',
                  transform=tax.transAxes)
artists += [title, t_text]
tax.set_ylim(np.percentile(np.abs(ts_data), [99.9]) * [-1, 1])
fig.colorbar(paths, cax=cax, ax=ax, label='Activation (μV)')


def animate_raw(i):
    """Animate the plot."""
    # i is the start
    center = i + half_width
    paths.set_array(ts_data[:, center])
    tsl = slice(center - half_width, center + half_width + 1)
    x = np.arange(2 * half_width + 1)
    for j, tpath in enumerate(tpaths):
        tpath.set_data(x, ts_data[j, tsl])
    t_text.set_text(f't={raw.first_time + center / sfreq:0.2f} sec')
    # If the middle sample contains an annotation (for a seizure-related
    # behavior) then change the title of the plot for all following frames
    event_idx = np.where(events[:, 0] - raw.first_samp == center)[0]
    if len(event_idx):
        title.set_text(inv_event_id[events[event_idx[0], 2]])
    return artists


anim = animation.FuncAnimation(
    fig, animate_raw, init_func=lambda: artists,
    frames=ts_data.shape[1] - 2 * half_width - 1, blit=True,
    interval=2000 / sfreq)

###############################################################################
# Alternatively, we can project the sensor data to the nearest locations on
# the pial surface and visualize that:

# sphinx_gallery_thumbnail_number = 5

evoked = mne.EvokedArray(gamma_power_t, gamma_info)
src = mne.read_source_spaces(
    op.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
trans = None  # identity transform
stc = mne.stc_near_sensors(evoked, trans, 'fsaverage', src=src,
                           mode='nearest', subjects_dir=subjects_dir,
                           distance=0.02)
clim = dict(kind='value', lims=[vmin * 0.9, vmin, vmax])
brain = stc.plot(surface='pial', hemi='rh',
                 colormap='viridis', clim=clim, views=['lat', 'med'],
                 subjects_dir=subjects_dir, size=(600, 800),
                 smoothing_steps=5)

# You can save a movie like the one on our documentation website with:
# brain.save_movie(time_dilation=1, interpolation='linear', framerate=5,
#                  time_viewer=True)
