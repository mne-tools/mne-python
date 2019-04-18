# -*- coding: utf-8 -*-
"""
.. _evoked-plotting-tutorial:

Plotting Evoked data
====================

.. include:: ../../tutorial_links.inc

This tutorial covers plotting methods of Evoked objects.
"""

###############################################################################
# We'll start by importing the modules we need and loading some example data,
# but this time we'll load sample data that's already been epoched and averaged
# to create :class:`~mne.Evoked` objects. Note that :class:`~mne.Evoked`
# objects saved in ``.fif`` format do not retain information about the baseline
# correction period, so :func:`~mne.read_evokeds` takes a ``baseline``
# parameter. As when creating :class:`~mne.Epochs` objects, the default is to
# not apply baseline correction, and passing a two-element iterable as
# ``baseline`` is interpreted as (beginning, end) of the baseline period in
# seconds; see the documentation of the ``baseline`` parameter in
# :func:`~mne.read_evokeds` for more information.

import os
import numpy as np
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_ave_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis-ave.fif')
evokeds = mne.read_evokeds(sample_data_ave_file, baseline=[None, 0])

###############################################################################
# The file contains four :class:`~mne.Evoked` objects, created from the four
# conditions we've seen already in the :ref:`epoching tutorials
# <epochs-intro-tutorial>` (the condition name is printed just after the "|"
# character in each object's summary representation):

_ = [print(ev) for ev in evokeds]

###############################################################################
# Before continuing, let's assign each to its own variable name, so we don't
# have to keep track of position in the ``evokeds`` list. We'll also drop the
# EOG channel since we won't need it in this tutorial:

for ev in evokeds:
    ev.drop_channels('EOG 061')
aud_left, aud_right, vis_left, vis_right = evokeds


###############################################################################
# You can get a sense of what preprocessing was done before creating these
# :class:`~mne.Evoked` objects by looking at the ``info`` dictionary. Here
# we'll just view a few relevant keys; you can see that the data were highpass
# filtered at 0.1 Hz (to remove slow signal drifts) and lowpass filtered at 40
# Hz, and have 3 empty room projectors, one EEG average reference projector,
# and two channels marked as "bad".

for key in ('highpass', 'lowpass', 'bads', 'projs'):
    print(f'{key:9}: {aud_left.info[key]}')

###############################################################################
# We saw in the :ref:`introduction to evoked data <evoked-intro-tutorial>`
# tutorial that the :meth:`~mne.Evoked.plot` method provides a quick look at
# the evoked data, with separate butterfly plots for each sensor type. Those
# basic plots can be customized in a number of ways. One useful parameter is
# ``spatial_colors``, which will color-code the channel traces based on their
# physical location, such that channels closer in space will be more similar in
# color. A color-coded sensor map will be inset in the plot area when
# ``spatial_colors=True``:

vis_right.plot(spatial_colors=True)

###############################################################################
# Another useful parameter is ``gfp``, which adds a plot of the global field
# power alongside the sensor traces:

vis_right.plot(gfp=True)

###############################################################################
# The :meth:`~mne.Evoked.plot` method also has both a ``picks`` parameter for
# specifying channels or channel types to include, and an ``exclude`` parameter
# for specifying channel(s) to omit. By default, any channels marked as "bad"
# are omitted; this behavior can be suppressed by passing an empty list to
# ``exclude``, resulting in a plot with good channels plotted in black and bad
# channels in red:

vis_right.plot(picks='grad', exclude=[])

###############################################################################
# :class:`~mne.Evoked` data can also be visualized as a time-by-channel image,
# with signal magnitude encoded by pixel color. :meth:`~mne.Evoked.plot_image`
# also has ``picks`` and ``exclude``  parameters, as well as various other ways
# to customize the look of the plot (see the documentation of
# :meth:`~mne.Evoked.plot_image` for details). Here, we can see how failing to
# exclude bad channels yields a plot where one noisy channel's high dynamic
# range compresses the range of all other channels into the middle of the
# colormap, making the pattern of activity difficult to distinguish.

vis_left.plot_image(picks='grad')
vis_left.plot_image(picks='grad', exclude=[])

###############################################################################
# Plotting the spatial distribution of :class:`~mne.Evoked` signals
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The spatial distribution of signals in an :class:`~mne.Evoked` object can be
# visualized in a couple different ways. Sensor-by-sensor time series plots are
# possible with :meth:`~mne.Evoked.plot_topo`:

vis_left.plot_topo(color='r')

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

conditions = {'Left auditory': aud_left, 'Left visual': vis_left}
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
