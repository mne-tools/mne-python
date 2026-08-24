"""
.. _plot_montage:

Plotting sensor layouts of EEG systems
======================================

This example illustrates how to load all the EEG system montages
shipped in MNE-python, and display it on the fsaverage template subject.
"""  # noqa: D205, D400
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import gc
import os.path as op

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.channels.montage import get_builtin_montages
from mne.datasets import fetch_fsaverage
from mne.viz import (
    clear_3d_figure,
    close_3d_figure,
    concatenate_images,
    create_3d_figure,
    set_3d_view,
)

# sphinx_gallery_thumbnail_number = 2

# %%
# There are a lot of montages to look at, so rather than opening one 3D figure per
# montage (which uses a lot of memory), we plot them one at a time into a single
# reusable figure, take a screenshot of each, and combine the screenshots into one
# Matplotlib figure at the end.

montages = get_builtin_montages()
size = (400, 400)  # size of each montage screenshot, in pixels
bgcolor = (0.5, 0.5, 0.5)
n_cols = 6
n_rows = int(np.ceil(len(montages) / n_cols))
# Size of each montage in the combined figure. Setting this rather than deriving it
# from ``size`` keeps the title font size independent of the screenshot resolution.
inches_per_montage = 2.5


def plot_montage_grid(images, titles):
    """Combine 3D screenshots into a single Matplotlib figure."""
    rows = [
        concatenate_images(images[start : start + n_cols], axis=1, bgcolor=bgcolor)
        for start in range(0, len(images), n_cols)
    ]
    grid = concatenate_images(rows, axis=0, bgcolor=bgcolor, centered=False)
    height, width = images[0].shape[:2]
    fig = plt.figure()
    # sizes the window to the figure, so the whole grid is visible interactively
    fig.set_size_inches(n_cols * inches_per_montage, n_rows * inches_per_montage)
    ax = fig.add_axes([0, 0, 1, 1])  # fill the entire figure
    ax.set_axis_off()
    ax.imshow(grid)
    for idx, title in enumerate(titles):
        row, col = divmod(idx, n_cols)
        ax.text(
            (col + 0.5) * width,
            (row + 0.05) * height,
            title,
            color="w",
            ha="center",
            va="top",
            fontsize=11,
        )
    return fig


# %%
# Check all montages against a sphere

# The figure must be created and closed within a single code block, because
# Sphinx-Gallery screenshots (and closes) every open 3D figure at the end of a block.
fig_3d = create_3d_figure(size=size, bgcolor=bgcolor)
set_3d_view(
    figure=fig_3d,
    azimuth=135,
    elevation=80,
    distance=0.6,
    focalpoint=(0.0, 0.0, 0.0),
)
images = list()
for current_montage in montages:
    montage = mne.channels.make_standard_montage(current_montage)
    info = mne.create_info(ch_names=montage.ch_names, sfreq=100.0, ch_types="eeg")
    info.set_montage(montage)
    sphere = mne.make_sphere_model(
        r0="auto", head_radius="auto", info=info, verbose="error"
    )
    mne.viz.plot_alignment(
        # Plot options
        show_axes=True,
        dig="fiducials",
        surfaces="head",
        trans=mne.Transform("head", "mri", trans=np.eye(4)),  # identity
        bem=sphere,
        info=info,
        fig=fig_3d,
        set_view=False,  # keep the view we set above
    )
    images.append(fig_3d.plotter.screenshot())
    clear_3d_figure(fig_3d)  # reuse the same figure for the next montage
    gc.collect()

close_3d_figure(fig_3d)
plot_montage_grid(images, montages)

# %%
# Check all montages against fsaverage

subjects_dir = op.dirname(fetch_fsaverage())

fig_3d = create_3d_figure(size=size, bgcolor=bgcolor)
set_3d_view(
    figure=fig_3d,
    azimuth=135,
    elevation=80,
    distance=0.6,
    focalpoint=(0.0, 0.0, 0.0),
)
images = list()
for current_montage in montages:
    montage = mne.channels.make_standard_montage(current_montage)
    # Create dummy info
    info = mne.create_info(ch_names=montage.ch_names, sfreq=100.0, ch_types="eeg")
    info.set_montage(montage)
    mne.viz.plot_alignment(
        # Plot options
        show_axes=True,
        dig="fiducials",
        surfaces="head",
        mri_fiducials=True,
        subject="fsaverage",
        subjects_dir=subjects_dir,
        info=info,
        coord_frame="mri",
        trans="fsaverage",  # transform from head coords to fsaverage's MRI
        fig=fig_3d,
        set_view=False,  # keep the view we set above
    )
    images.append(fig_3d.plotter.screenshot())
    clear_3d_figure(fig_3d)
    gc.collect()

close_3d_figure(fig_3d)
plot_montage_grid(images, montages)
