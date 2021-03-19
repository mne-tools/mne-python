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
import matplotlib
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

# Pick only the ECoG channels, removing the EKG channels
raw.pick_types(ecog=True)

# Load the data
raw.load_data()

# Then we remove line frequency interference
raw.notch_filter([60], trans_bandwidth=3)

# drop bad channels
raw.drop_channels(raw.info['bads'])

# the coordinate frame of the montage
print(raw.get_montage().get_positions()['coord_frame'])

# Find the annotated events
events, event_id = mne.events_from_annotations(raw)

# Make a 25 second epoch that spans before and after the seizure onset
epoch_length = 25  # seconds
epochs = mne.Epochs(raw, events, event_id=event_id['onset'],
                    tmin=13, tmax=13 + epoch_length, baseline=None)

# And then load data and downsample.
# .. note: This is just to save execution time in this example, you should
#          not need to do this in general!
epochs.load_data()
epochs.resample(200)  # Hz, will also load the data for us


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

# look at second view
fig = plot_alignment(raw.info, subject='fsaverage', subjects_dir=subjects_dir,
                     surfaces=['pial'], coord_frame='mri')
az, el, focalpoint = -120, -140, [0.027, 0.017, -0.033]
mne.viz.set_3d_view(fig, azimuth=az, elevation=el, focalpoint=focalpoint)

xy2, im2 = snapshot_brain_montage(fig, raw.info)

###############################################################################
# Compute frequency features of the data
# --------------------------------------
#
# Next, we'll compute the signal power in the gamma (30-90 Hz) and alpha
# (8-12 Hz) bands, downsampling the result to 10 Hz (to save time).

freqs = np.linspace(5, 95, 10)
power = mne.time_frequency.tfr_multitaper(epochs, freqs, n_cycles=freqs / 2,
                                          return_itc=False)

###############################################################################
# We can project gamma power from the sensor data to the nearest locations on
# the pial surface and visualize that:

# sphinx_gallery_thumbnail_number = 5

sfreq = 10
gamma_power_t = np.median(power.data[:, power.freqs >= 30], axis=1)
evoked = mne.EvokedArray(gamma_power_t, epochs.info)

# downsample for animation
evoked.resample(sfreq)
src = mne.read_source_spaces(
    op.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
trans = None  # identity transform
stc = mne.stc_near_sensors(evoked, trans, 'fsaverage', src=src,
                           mode='nearest', subjects_dir=subjects_dir,
                           distance=0.02)
vmin, vmax = np.percentile(gamma_power_t, [10, 90])
clim = dict(kind='value', lims=[vmin * 0.9, vmin, vmax])
brain = stc.plot(surface='pial', hemi='rh', colormap='viridis',
                 clim=clim, views=['lat', 'med'], subjects_dir=subjects_dir,
                 size=(600, 800), smoothing_steps=5)

# You can save a movie like the one on our documentation website with:
# brain.save_movie(time_dilation=1, interpolation='linear', framerate=5,
#                  time_viewer=True)

###############################################################################
# Visualize the time-evolution of seizure activity on the brain
# -------------------------------------------------------------
#
# Say we want to visualize the evolution of seizure in time. We can use
# `matplotlib.animation.FuncAnimation` to create an animation and apply this
# to the brain figure.
#
# As can be seen in the plot below, the seizure originates in the temporal
# lobe and spreads to new electrodes when electrode label names come up in
# the title. The seizure eventually becomes generalized and then subsides.
# As mentioned earlier, we sub-sample and only show a small portion of
# the seizure for demonstration purposes. Feel free to run this example
# to visualize the spread of the seizure yourself, specifically in
# channels annotated in the ``events`` array.


def update_time_series(t_path, x, y, t_data):
    """Update the line plots for the animation."""
    t_path.set_data(x + np.linspace(-50, 50, t_data.size),
                    y + t_data - t_data.mean())


# colormap to associate electrode positions
cmap = 'viridis'
cmap_call = matplotlib.cm.get_cmap(cmap)

# get epochs data, normalize for plots
epoch_data = epochs.copy().resample(sfreq).get_data()[0]
epoch_data *= 200 / epoch_data.max()

# Convert from a dictionary to array to plot
xy_pts = np.vstack([xy[ch] for ch in raw.info['ch_names']])
xy_pts2 = np.vstack([xy2[ch] for ch in raw.info['ch_names']])

# Make a group to plot the temporal lobe contacts
group = np.where(np.logical_and(xy_pts[:, 1] > 800, xy_pts[:, 0] > 1000))[0]
color_line = np.linspace(0, 1, xy_pts.shape[0])

# create the figure to apply the animation
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax.imshow(im)
ax.set_axis_off()
ax.set_xlim([100, im.shape[0]])
ax.set_ylim([im.shape[1] - 200, 0])
ax2.imshow(im2)
ax2.set_axis_off()
ax2.set_xlim([375, im.shape[0] - 750])
ax2.set_ylim([im.shape[1] - 400, 650])
ax3 = ax2.inset_axes((0.7, 0.0, 0.3, 0.3))
ax3.imshow(im2)
ax3.plot([375, 375, im.shape[0] - 750, im.shape[0] - 750, 375],
         [im.shape[1] - 400, 650, 650, im.shape[1] - 400, im.shape[1] - 400],
         color='k')
ax3.set_axis_off()

ax.set_title('Seizure time course', size='large')
t_text = ax2.text(0.5, 0, 't=0', animated=True, ha='center', va='bottom',
                  transform=ax2.transAxes)
fig.tight_layout()

t_paths = list()
for i in range(xy_pts.shape[0]):
    x, y = xy_pts[i]
    color = cmap_call(color_line[i])
    t_paths.append(ax.plot(x + np.linspace(-50, 50, sfreq),
                           np.nan * np.zeros((sfreq)), color=color)[0])


t_paths2 = list()
for i in group:
    x, y = xy_pts2[i]
    color = cmap_call(color_line[i])
    t_paths2.append(ax2.plot(x + np.linspace(-50, 50, sfreq),
                             np.nan * np.zeros((sfreq)), color=color)[0])


artists = [*t_paths, *t_paths2, t_text]


def animate_lines(i):
    """Animate the plot."""
    for idx, t_path in zip(range(xy_pts.shape[0]), t_paths):
        x, y = xy_pts[idx]
        update_time_series(t_path, x, y, epoch_data[idx, i: i + sfreq])
    for idx, t_path in zip(group, t_paths2):
        x, y = xy_pts2[idx]
        update_time_series(t_path, x, y, epoch_data[idx, i: i + sfreq])
    t_text.set_text(f't={raw.first_time + i / sfreq:0.3f} sec')
    return artists


anim = animation.FuncAnimation(fig, animate_lines, init_func=lambda: artists,
                               frames=epoch_data.shape[1] - sfreq, blit=True,
                               interval=1000 / sfreq)

###############################################################################
# Visualize the spectral power on the brain
# -----------------------------------------
#
# Say we want to visualize the evolution of the power, instead of just
# plotting the average. We can use also animation this and apply it
# to the brain figure.


def update_power_series(f_path, x, y, f_data):
    """Update the bar plots for the animation."""
    for line, p, offset in zip(f_path, f_data, offsets):
        line.set_data([x + offset, x + offset], [y - p, y + p])


# get power data, resample
power_data = np.zeros((len(power.ch_names), power.freqs.size,
                       epoch_length * sfreq))
n_points = int(power.data.shape[2] / (epoch_length * sfreq))
for i in range(epoch_length * sfreq):
    power_data[:, :, i] = \
        power.data[:, :, i * n_points:(i + 1) * n_points].mean(axis=2)


# normalize for plots
power_data /= power_data.mean(axis=2)[:, :, np.newaxis]
power_data *= 1000 / power_data.max()

# create the figure to apply the power animation
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax.imshow(im)
ax.set_axis_off()
ax.set_xlim([100, im.shape[0]])
ax.set_ylim([im.shape[1] - 200, 0])
ax2.imshow(im2)
ax2.set_axis_off()
ax2.set_xlim([375, im.shape[0] - 750])
ax2.set_ylim([im.shape[1] - 400, 650])
ax3 = ax2.inset_axes((0.7, 0.0, 0.3, 0.3))
ax3.imshow(im2)
ax3.plot([375, 375, im.shape[0] - 750, im.shape[0] - 750, 375],
         [im.shape[1] - 400, 650, 650, im.shape[1] - 400, im.shape[1] - 400],
         color='k')
ax3.set_axis_off()

ax.set_title('Seizure time-frequency course (5-95 Hz)', size='large')
t_text = ax2.text(0.5, 0, 't=0', animated=True, ha='center', va='bottom',
                  transform=ax2.transAxes)
fig.tight_layout()

offsets = np.linspace(-50, 50, power.freqs.size)
colors = color = cmap_call(np.linspace(0, 1, power.freqs.size))
f_paths = list()
for i in range(xy_pts.shape[0]):
    x, y = xy_pts[i]
    f_paths.append([ax.plot([x + offset, x + offset], [np.nan, np.nan],
                            color=color, linewidth=3)[0]
                    for color, offset in zip(colors, offsets)])


f_paths2 = list()
for i in group:
    x, y = xy_pts2[i]
    f_paths2.append([ax2.plot([x + offset, x + offset], [np.nan, np.nan],
                              color=color, linewidth=3)[0]
                     for color, offset in zip(colors, offsets)])


artists = [*[line for f_path in f_paths for line in f_path],
           *[line for f_path in f_paths2 for line in f_path], t_text]


def animate_bars(i):
    """Animate the plot."""
    for idx, f_path in zip(range(xy_pts.shape[0]), f_paths):
        x, y = xy_pts[idx]
        f_data = power_data[idx, :, i]
        update_power_series(f_path, x, y, f_data)
    for idx, f_path in zip(group, f_paths2):
        x, y = xy_pts2[idx]
        f_data = power_data[idx, :, i]
        update_power_series(f_path, x, y, f_data)
    t_text.set_text(f't={raw.first_time + i / sfreq:0.3f} sec')
    return artists


anim = animation.FuncAnimation(fig, animate_bars, init_func=lambda: artists,
                               frames=power_data.shape[2], blit=True,
                               interval=1000 / sfreq)
