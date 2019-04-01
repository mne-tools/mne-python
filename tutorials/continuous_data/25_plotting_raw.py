# -*- coding: utf-8 -*-
"""
.. _plotting-raw-tutorial:

Built-in plotting methods for :class:`~mne.io.Raw` objects
==========================================================

.. include:: ../../tutorial_links.inc

This tutorial covers two plotting methods for Raw objects: Raw.plot() and
Raw.plot_psd().
"""

###############################################################################
# As always we'll start by importing the modules we need, and loading some
# example data:

import os
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

###############################################################################
# We've seen in :ref:`a previous tutorial <subselecting-raw-tutorial>` how to
# plot data from a :class:`~mne.io.Raw` object using matplotlib, but
# :class:`~mne.io.Raw` objects also have several built-in plotting methods:
#
# - :meth:`~mne.io.Raw.plot`
# - :meth:`~mne.io.Raw.plot_psd`
# - :meth:`~mne.io.Raw.plot_psd_topo`
# - :meth:`~mne.io.Raw.plot_projs_topomap`
# - :meth:`~mne.io.Raw.plot_sensors`
#
# We'll discuss the first three here; :meth:`~mne.io.Raw.plot_projs_topomap` is
# discussed in :ref:`projectors-topomap-tutorial`;
# :meth:`~mne.io.Raw.plot_sensors` is discussed in
# :ref:`sensor-locations-tutorial`.
#
# The :meth:`~mne.io.Raw.plot` method of :class:`~mne.io.Raw` objects provides
# a versatile interface for exploring continuous data. For interactive viewing
# and data quality checking, it can be called with no additional parameters:

raw.plot()

###############################################################################
# It may not be obvious when viewing this tutorial online, but by default, the
# :meth:`~mne.io.Raw.plot` method generates an *interactive* plot window with
# several useful features:
#
# - It spaces the channels equally along the y-axis.
#
#   - 20 channels are shown by default; you can scroll through the channels
#     using the :kbd:`up` and :kbd:`down` arrow keys, or by clicking on the
#     colored scroll bar on the right edge of the plot.
#
#   - The number of visible channels can be adjusted by the ``n_channels``
#     parameter, or changed interactively using :kbd:`page up` and :kbd:`page
#     down` keys.
#
#   - You can toggle the display to "butterfly" mode (superimposing all
#     channels of the same type on top of one another) by pressing :kbd:`b`,
#     or start in butterfly mode by passing the ``butterfly=True`` parameter.
#
# - It shows the first 10 seconds of the :class:`~mne.io.Raw` object.
#
#   - You can shorten or lengthen the window length using :kbd:`home` and
#     :kbd:`end` keys, or start with a specific window duration by passing the
#     ``duration`` parameter.
#
#   - You can scroll in the time domain using the :kbd:`left` and
#     :kbd:`right` arrow keys, or start at a specific point by passing the
#     ``start`` parameter. Scrolling using :kbd:`shift`:kbd:`right` or
#     :kbd:`shift`:kbd:`left` scrolls a full window width at a time.
#
# - It allows clicking on channels to mark/unmark as "bad".
#
#   - When the plot window is closed, the :class:`~mne.io.Raw` object's
#     ``info`` attribute will be updated, adding or removing the newly
#     (un)marked channels to the :class:`~mne.Info` object's ``bads`` field.
#
# - It allows interactive :term:`annotation <annotations>` of the raw data.
#
#   - This allows you to mark time spans that should be excluded from future
#     computations due to large movement artifacts, line noise, or other
#     distortions of the signal. Annotation mode is entered by pressing
#     :kbd:`a`. See :ref:`annotations-tutorial` for details.
#
# - It automatically applies any :term:`projectors <projector>` before plotting
#   the data.
#
#   - These can be enabled/disabled interactively by clicking the ``Proj``
#     button at the lower right corner of the plot window, or disabled by
#     default by passing the ``proj=False`` parameter.  See
#     :ref:`projectors-basics-tutorial` for more info on projectors.
#
# These and other keyboard shortcuts are listed in the Help window, accessed
# through the ``Help`` button at the lower left corner of the plot window.
# Other plot properties (such as color of the channel traces, channel order and
# grouping, simultaneous plotting of :term:`events`, scaling, clipping,
# filtering, etc.) can also be adjusted through parameters passed to the
# :meth:`~mne.io.Raw.plot` method; see the documentation of
# :meth:`~mne.io.Raw.plot` for details.
#
# To visualize the frequency content of continuous data, the
# :class:`~mne.io.Raw` object provides a :meth:`~mne.io.Raw.plot_psd` to plot
# the `spectral density`_ of the data.

raw.plot_psd(average=True)

###############################################################################
# If the data have been filtered, vertical dashed lines will automatically
# indicate filter boundaries. The spectrum for each channel type is drawn in
# its own subplot; here we've passed the ``average=True`` parameter to get a
# summary for each channel type, but it is also possible to plot each channel
# individually, with options for how the spectrum should be computed,
# color-coding the channels by location, and more. See the documentation of
# :meth:`~mne.io.Raw.plot_psd` for full details.
#
# You can also plot the spectral density of continuous data on a
# sensor-by-sensor basis. :meth:`~mne.io.Raw.plot_psd` has a ``picks``
# parameter that accepts strings or lists of strings indicating channel name or
# type.

midline = ['EEG 002', 'EEG 012', 'EEG 030', 'EEG 048', 'EEG 058', 'EEG 060']
raw.plot_psd(picks=midline)

###############################################################################
# Alternatively, you can plot the PSD for every sensor on its own axes, with
# the axes arranged spatially to correspond to sensor locations in space, using
# :meth:`~mne.io.Raw.plot_psd_topo`:

raw.plot_psd_topo()

###############################################################################
# This plot is also interactive; hovering over each "thumbnail" plot will
# display the channel name in the bottom left of the plot window, and clicking
# on a thumbnail plot will create a second figure showing a larger version of
# the selected channel's spectral density (as if you had called
# :meth:`~mne.io.Raw.plot_psd` on that channel).
#
# By default, :meth:`~mne.io.Raw.plot_psd_topo` will show only the MEG
# channels if MEG channels are present; if only EEG channels are found, they
# will be plotted instead:

raw.copy().pick_types(meg=False, eeg=True).plot_psd_topo()

###############################################################################
# Alternatively, to show only EEG channels without resorting to
# :meth:`~mne.io.Raw.copy` and :meth:`~mne.io.Raw.pick_types`, you can pass an
# EEG *layout* to :meth:`~mne.io.Raw.plot_psd_topo`:

eeg_layout = mne.channels.find_layout(raw.info, ch_type='eeg')
raw.plot_psd_topo(layout=eeg_layout)

###############################################################################
# See :ref:`sensor-locations-tutorial` for more info on layouts.
