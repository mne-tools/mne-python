# -*- coding: utf-8 -*-
"""
.. _evoked-plotting-tutorial:

Plotting Evoked data
====================

.. include:: ../../tutorial_links.inc

This tutorial covers plotting methods of Evoked objects.
"""

###############################################################################
# We'll start by importing the modules we need, loading some example data,
# epoching it, and averaging within condition to get :class:`~mne.Evoked`
# objects:

import os
import numpy as np
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)
# get events
events = mne.find_events(raw, stim_channel='STI 014')
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'button': 32}
# make epochs
raw.info['bads'].append('EEG 007')
reject_criteria = dict(mag=4e-12, grad=4e-10, eeg=150e-6, eog=250e-6)
epochs = mne.Epochs(raw, events, event_id=event_dict, reject=reject_criteria,
                    preload=True)
# make evokeds
aud_left = epochs['auditory/left'].average()
aud_right = epochs['auditory/right'].average()
vis_left = epochs['visual/left'].average()
vis_right = epochs['visual/right'].average()

# TODO Why is this so different from the manual averaging above?
# sample_data_ave_file = os.path.join(sample_data_folder, 'MEG', 'sample',
#                                     'sample_audvis-ave.fif')
# evokeds = mne.read_evokeds(sample_data_ave_file)
# aud_left, aud_right, vis_left, vis_right = evokeds

###############################################################################
# We saw in the :ref:`introduction to evoked data <evoked-intro-tutorial>`
# tutorial that the :meth:`~mne.Evoked.plot` method provides a quick look at
# the evoked data, with separate butterfly plots for each sensor type. Those
# basic plots can be customized in a number of ways. One useful parameter is
# ``spatial_colors``, which will color-code the channel traces based on their
# physical location, such that channels closer in space will be more similar in
# color. A color-coded sensor map will be inset in the plot area when
# ``spatial_colors=True``:

aud_left.plot(spatial_colors=True)

###############################################################################
# Another useful parameter is ``gfp``, which adds a plot of the global field
# power alongside the sensor traces:

aud_left.plot(gfp=True)

###############################################################################
# The :meth:`~mne.Evoked.plot` method also has both a ``picks`` parameter for
# specifying channels or channel types to include, and an ``exclude`` parameter
# for specifying channel(s) to omit. By default, any channels marked as "bad"
# are omitted; this behavior can be suppressed by passing an empty list to
# ``exclude``, resulting in a plot with good channels plotted in black and bad
# channels in red:

aud_left.plot(picks='grad', exclude=[])

###############################################################################
# :class:`~mne.Evoked` data can also be visualized as a time-by-channel image,
# with signal magnitude encoded by pixel color. :meth:`~mne.Evoked.plot_image`
# also has ``picks`` and ``exclude``  parameters, as well as various other ways
# to customize the look of the plot (see the documentation of
# :meth:`~mne.Evoked.plot_image` for details). Here, we can see how failing to
# exclude bad channels yields a plot where one noisy channel's high dynamic
# range compresses the range of all other channels into the middle of the
# colormap, making the pattern of activity difficult to distinguish.

aud_left.plot_image(picks='grad')
aud_left.plot_image(picks='grad', exclude=[])

###############################################################################
# Plotting the spatial distribution of :class:`~mne.Evoked` signals
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The spatial distribution of signals in an :class:`~mne.Evoked` object can be
# visualized in a couple different ways. Sensor-by-sensor time series plots are
# possible with :meth:`~mne.Evoked.plot_topo`:

aud_left.plot_topo(color='r')

###############################################################################
# Another option is to plot scalp topographies at specific times with
# :meth:`~mne.Evoked.plot_topomap`. This method includes automatic time
# selection by passing ``times='auto'``; this is the default, and will select
# the :class:`~mne.Evoked` object's endpoints plus up to 8 equally spaced
# points in between (the number depends on the :class:`~mne.Evoked` object's
# duration):

vis_left.plot_topomap(times='auto')

###############################################################################
# You can also specify ``times='peaks'``, which will choose time points based
# on local maxima of the global field power:

vis_left.plot_topomap(times='peaks')

###############################################################################
# ...or you can specify specific time points yourself, as shown below. Each of
# these methods also works with the ``average`` parameter, which specifies the
# length of a time window centered around each value of ``times``, over which
# the signal will be averaged before creating the topomap for that time point.
# Here we look at a series of overlapping 50 ms windows around the peak
# response:

vis_left.plot_topomap(times=np.linspace(0.1, 0.25, 7), average=0.05)

###############################################################################
# .. warning::
#
#     By default, :meth:`~mne.Evoked.plot_topomap` sets the color scale to best
#     fit the data maxima/minima *across the displayed time points*, not across
#     the entire :class:`~mne.Evoked` object. Therefore, the plots above do not
#     all have the same color scale limits, so while it is sensible to compare
#     subplots within a given call to :meth:`~mne.Evoked.plot_topomap`, use
#     caution when comparing across plots from different calls. You can enforce
#     specific color scale limits using the ``vmin`` and ``vmax`` parameters if
#     needed.
#
#
# If you want to combine butterfly plots of the :class:`~mne.Evoked` time
# series with scalp topographies at specific times, there is a dedicated method
# called :meth:`~mne.Evoked.plot_joint`. By default, it will generate separate
# figures for each sensor type, select times based on peaks in global field
# power, and color-code channel traces by spatial location. As with other
# plotting methods for :class:`~mne.Evoked` objects, you can customize the plot
# with parameters like ``picks``, ``exclude``, ``times``, etc. See the
# documentation of :meth:`~mne.Evoked.plot_joint` for details.

vis_left.plot_joint(picks='mag', times=[0.09, 0.12, 0.18, 0.36])

###############################################################################
# Comparing :class:`~mne.Evoked` responses across conditions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Finally, you can easily compare evoked responses from two conditions using
# the :func:`mne.viz.plot_compare_evokeds` function. The function accepts a
# list or dictionary of :class:`~mne.Evoked` objects; if a dictionary is
# provided its keys will be used as the legend labels:

conditions = {'Right auditory': aud_right, 'Right visual': vis_right}
mne.viz.plot_compare_evokeds(conditions, picks='grad',
                             title='Average of gradiometers')

###############################################################################
# There is also a function :func:`mne.viz.plot_evoked_topo` (corresponding to
# the :meth:`~mne.Evoked.plot_topo` method) that allows multiple
# :class:`~mne.Evoked` objects to be plotted on the same sensor grid, by
# providing a list of :class:`~mne.Evoked` objects. In this tutorial the figure
# is too small to be very informative, but when it is possible to zoom in on
# the subplots (e.g., during interactive plotting, or when the figure is saved
# in a vector graphic format) it can be quite useful.

mne.viz.plot_evoked_topo([aud_left, aud_right, vis_left, vis_right])
