# -*- coding: utf-8 -*-
"""
.. _tut-eyetrack-heatmap:

=============================================
Plotting eye-tracking heatmaps in MNE-Python
=============================================

This tutorial covers plotting eye-tracking position data as a heatmap.

.. seealso:: :ref:`tut-importing-eyetracking-data`

"""

# %%
# Data loading
# ------------
#
# As usual we start by importing the modules we need and loading some
# :ref:`example data <eyelink-dataset>`: eye-tracking data recorded from SR research's
# ``'.asc'`` file format. We'll also define a helper function to plot image files.


import matplotlib.pyplot as plt

import mne
from mne.viz.eyetracking import plot_gaze


# Define a function to plot stimuli photos
def plot_images(image_paths, ax, titles=None):
    for i, image_path in enumerate(image_paths):
        ax[i].imshow(plt.imread(image_path))
        if titles:
            ax[i].set_title(titles[i])
    return fig


# define variables to pass to the plot_gaze function
px_width, px_height = 1920, 1080

task_fpath = mne.datasets.eyelink.data_path() / "freeviewing"
et_fpath = task_fpath / "sub-01_task-freeview_eyetrack.asc"
natural_stim_fpath = task_fpath / "stim" / "naturalistic.png"
scrambled_stim_fpath = task_fpath / "stim" / "scrambled.png"
image_paths = list([natural_stim_fpath, scrambled_stim_fpath])


raw = mne.io.read_raw_eyelink(et_fpath)

# %%
# Task background
# ---------------
#
# Participants watched videos while eye-tracking data was collected. The videos showed
# people dancing, or scrambled versions of those videos. Each video lasted about 20
# seconds. An image of each video is shown below.

fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
plot_images(image_paths, ax, ["Natural", "Scrambled"])


# %%
# Process and epoch the data
# --------------------------
#
# First we will interpolate missing data during blinks and epoch the data.

mne.preprocessing.eyetracking.interpolate_blinks(raw, interpolate_gaze=True)
raw.annotations.rename({"dvns": "natural", "dvss": "scrambled"})  # more intuitive
event_ids = {"natural": 1, "scrambled": 2}
events, event_dict = mne.events_from_annotations(raw, event_id=event_ids)

epochs = mne.Epochs(
    raw, events=events, event_id=event_dict, tmin=0, tmax=20, baseline=None
)

# %%
# .. seealso:: :ref:`tut-eyetrack`
#

# %%
# Plot a heatmap of the eye-tracking data
# ---------------------------------------
#
# To make a heatmap of the eye-tracking data, we can use the function
# :func:`~mne.viz.eyetracking.plot_gaze`. We will need to define the dimensions of our
# canvas; for this file, the eye position data are reported in pixels, so we'll use the
# screen resolution of the participant screen (1920x1080) as the width and height. We
# can also use the sigma parameter to smooth the plot.

fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
plot_gaze(
    epochs["natural"],
    width=px_width,
    height=px_height,
    sigma=50,
    cmap="viridis",
    axes=ax[0],
    show=False,
)
ax[0].set_title("Gaze Heatmap (Natural)")
plot_gaze(
    epochs["scrambled"],
    width=px_width,
    height=px_height,
    sigma=50,
    cmap="viridis",
    axes=ax[1],
    show=False,
)
ax[1].set_title("Gaze Heatmap (Scrambled)")
plt.show()

# %%
# Overlaying plots with images
# ----------------------------
#
# We can use matplotlib to plot the gaze heatmaps on top of the stimuli images. We'll
# customize a :class:`~matplotlib.colors.Colormap` to make some values of the heatmap
# (in this case, the color black) completely transparent. We'll then use the ``vlim``
# parameter to force the heatmap to start at a value greater than the darkest value in
# our previous heatmap, which will make the darkest colors of the heatmap transparent.

cmap = plt.get_cmap("viridis")
cmap.set_under("k", alpha=0)  # make the lowest values transparent
fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

plot_images(image_paths, ax)
plot_gaze(
    epochs["natural"],
    width=px_width,
    height=px_height,
    vlim=(0.0003, None),
    sigma=50,
    cmap=cmap,
    axes=ax[0],
    show=False,
)
ax[0].set_title("Natural")

plot_gaze(
    epochs["scrambled"],
    width=px_width,
    height=px_height,
    sigma=50,
    vlim=(0.0001, None),
    cmap=cmap,
    axes=ax[1],
    show=False,
)
ax[1].set_title("Scrambled")
plt.show()
