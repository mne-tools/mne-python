# -*- coding: utf-8 -*-
"""
.. _ex-evoked-topomap:

========================================
Plotting topographic maps of evoked data
========================================

Load evoked data and plot topomaps for selected time points using multiple
additional options.
"""
# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#          Tal Linzen <linzen@nyu.edu>
#          Denis A. Engeman <denis.engemann@gmail.com>
#          Mikołaj Magnuski <mmagnuski@swps.edu.pl>
#
# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 5

import numpy as np
import matplotlib.pyplot as plt

from mne.datasets import sample
from mne import read_evokeds

print(__doc__)

path = sample.data_path()
fname = path + '/MEG/sample/sample_audvis-ave.fif'

# load evoked corresponding to a specific condition
# from the fif file and subtract baseline
condition = 'Left Auditory'
evoked = read_evokeds(fname, condition=condition, baseline=(None, 0))

###############################################################################
# Basic `plot_topomap` options
# ----------------------------
#
# We plot evoked topographies using :func:`mne.Evoked.plot_topomap`. The first
# argument, ``times`` allows to specify time instants (in seconds!) for which
# topographies will be shown. We select timepoints from 50 to 150 ms with a
# step of 20ms and plot magnetometer data:
times = np.arange(0.05, 0.151, 0.02)
evoked.plot_topomap(times, ch_type='mag', time_unit='s')

###############################################################################
# If times is set to None at most 10 regularly spaced topographies will be
# shown:
evoked.plot_topomap(ch_type='mag', time_unit='s')

###############################################################################
# Instead of showing topographies at specific time points we can compute
# averages of 50 ms bins centered on these time points to reduce the noise in
# the topographies:
evoked.plot_topomap(times, ch_type='mag', average=0.05, time_unit='s')

###############################################################################
# We can plot gradiometer data (plots the RMS for each pair of gradiometers)
evoked.plot_topomap(times, ch_type='grad', time_unit='s')

###############################################################################
# Additional `plot_topomap` options
# ---------------------------------
#
# We can also use a range of various :func:`mne.viz.plot_topomap` arguments
# that control how the topography is drawn. For example:
#
# * ``cmap`` - to specify the color map
# * ``res`` - to control the resolution of the topographies (lower resolution
#   means faster plotting)
# * ``outlines='skirt'`` to see the topography stretched beyond the head circle
# * ``contours`` to define how many contour lines should be plotted
evoked.plot_topomap(times, ch_type='mag', cmap='Spectral_r', res=32,
                    outlines='skirt', contours=4, time_unit='s')

###############################################################################
# If you look at the edges of the head circle of a single topomap you'll see
# the effect of extrapolation. By default ``extrapolate='box'`` is used which
# extrapolates to a large box stretching beyond the head circle.
# Compare this with ``extrapolate='head'`` (second topography below) where
# extrapolation goes to 0 at the head circle and ``extrapolate='local'`` where
# extrapolation is performed only within some distance from channels:

extrapolations = ['box', 'head', 'local']
fig, axes = plt.subplots(figsize=(7.5, 2.5), ncols=3)

# Here we look at EEG channels, and use a custom head sphere to get all the
# sensors to be well within the drawn head surface
for ax, extr in zip(axes, extrapolations):
    evoked.plot_topomap(0.1, ch_type='eeg', size=2, extrapolate=extr, axes=ax,
                        show=False, colorbar=False, sphere=(0., 0., 0., 0.09))
    ax.set_title(extr, fontsize=14)

###############################################################################
# More advanced usage
# -------------------
#
# Now we plot magnetometer data as topomap at a single time point: 100 ms
# post-stimulus, add channel labels, title and adjust plot margins:
evoked.plot_topomap(0.1, ch_type='mag', show_names=True, colorbar=False,
                    size=6, res=128, title='Auditory response',
                    time_unit='s')
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88)

###############################################################################
# Animating the topomap
# ---------------------
#
# Instead of using a still image we can plot magnetometer data as an animation
# (animates only in matplotlib interactive mode)
evoked.animate_topomap(ch_type='mag', times=times, frame_rate=10,
                       time_unit='s')
