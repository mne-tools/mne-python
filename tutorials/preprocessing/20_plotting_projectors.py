# -*- coding: utf-8 -*-
"""
.. _projectors-topomap-tutorial:

Plotting the spatial pattern of projectors
==========================================

.. include:: ../../tutorial_links.inc

This tutorial describes how to plot the topography (spatial pattern) of a
projector.
"""

###############################################################################
# As usual we'll start by importing the modules we need, loading some
# example data, and cropping it to save on memory:

import os
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
raw.crop(tmax=60).load_data()

###############################################################################
# You can plot the sensor distribution of individual projectors using the
# :meth:`~mne.Projection.plot_topomap` method on the :class:`~mne.Projection`
# object, to see the spatial pattern of what's being projected out:

first_projector = raw.info['projs'][0]
first_projector.plot_topomap()

###############################################################################
# The :class:`~mne.io.Raw` class also has a convenient
# :meth:`~mne.io.Raw.plot_projs_topomap` method for plotting all its projectors
# at once. In this example data, retaining three PCA components was sufficient
# to reduce environmental noise to acceptable levels, so there are three
# projectors in ``raw.info['projs']`` and hence three topographic maps:

raw.plot_projs_topomap()

###############################################################################
# You can plot scalp topographies for projectors loaded from disk just like
# with projectors incorporated in the :class:`~mne.io.Raw` object. When we
# plotted the sensor distribution of ``first_projector`` above, MNE-Python
# inferred sensor position from the projector data for MEG channels, but for
# EEG channels you must supply a channel layout so that MNE-Python can
# correctly map EEG channel names to locations on the head. Fortunately, sensor
# layouts can be extracted from an :class:`~mne.Info` object, and
# :func:`~mne.viz.plot_projs_topomap` has an ``info`` parameter that can be
# used for this purpose:

ecg_proj_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                             'sample_audvis_ecg-proj.fif')
ecg_projs = mne.read_proj(ecg_proj_file)
mne.viz.plot_projs_topomap(ecg_projs, info=raw.info)

###############################################################################
# Alternatively, you can extract the sensor layout from the :class:`~mne.Info`
# object using the function :func:`~mne.channels.find_layout`, and provide that
# to the :func:`~mne.viz.plot_projs_topomap` function instead. However, if
# you're plotting topomaps for both EEG and MEG channels at the same time, you
# must supply a list of layouts for *all* channel types (even though the MEG
# layouts could be inferred). In other words, either you must pass layouts for
# all channel types, or it must be possible to infer all layouts; providing
# layouts for some channel types and inferring layouts for others is not
# possible in a single call to :func:`~mne.viz.plot_projs_topomap`. Luckily
# :func:`~mne.viz.plot_projs_topomap` accepts a list of layouts, and we can use
# a `list comprehension`_ to extract layouts for all three channel types. You
# can visualize the layouts with the :meth:`~mne.channels.Layout.plot()`
# method:

all_channel_layouts = [mne.find_layout(raw.info, ch_type=channel_type)
                       for channel_type in ('grad', 'mag', 'eeg')]
for layout in all_channel_layouts:
    layout.plot()

###############################################################################
# With the layouts in hand, you can achieve the same topographic plot of
# projectors shown above by passing the list of layouts to
# :func:`~mne.viz.plot_projs_topomap` instead of the :class:`~mne.Info` object:
#
# .. code-block:: python3
#
#     mne.viz.plot_projs_topomap(ecg_projs, layout=all_channel_layouts)
#
