# -*- coding: utf-8 -*-
"""
.. _tut-visualize-raw:

=========================================
Built-in plotting methods for Raw objects
=========================================

This tutorial shows how to plot continuous data as a time series, how to plot
the spectral density of continuous data, and how to plot the sensor locations
and projectors stored in `~mne.io.Raw` objects.

As usual we'll start by importing the modules we need, loading some
:ref:`example data <sample-dataset>`, and cropping the `~mne.io.Raw`
object to just 60 seconds before loading it into RAM to save memory:
"""

# %%

import os

import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)
raw.crop(tmax=60).load_data()

# %%
# We've seen in :ref:`a previous tutorial <tut-raw-class>` how to plot data
# from a `~mne.io.Raw` object using :doc:`matplotlib <matplotlib:index>`,
# but `~mne.io.Raw` objects also have several built-in plotting methods:
#
# - `~mne.io.Raw.plot`
# - `~mne.io.Raw.plot_sensors`
# - `~mne.io.Raw.plot_projs_topomap`
#
# The first one is discussed here in detail; the last two are shown briefly
# and covered in-depth in other tutorials. This tutorial also covers a few
# ways of plotting the spectral content of :class:`~mne.io.Raw` data.
#
#
# Interactive data browsing with ``Raw.plot()``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The `~mne.io.Raw.plot` method of `~mne.io.Raw` objects provides
# a versatile interface for exploring continuous data. For interactive viewing
# and data quality checking, it can be called with no additional parameters:

raw.plot()

# %%
# It may not be obvious when viewing this tutorial online, but by default, the
# `~mne.io.Raw.plot` method generates an *interactive* plot window with
# several useful features:
#
# - It spaces the channels equally along the y-axis.
#
#   - 20 channels are shown by default; you can scroll through the channels
#     using the :kbd:`↑` and :kbd:`↓` arrow keys, or by clicking on the
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
# - It shows the first 10 seconds of the `~mne.io.Raw` object.
#
#   - You can shorten or lengthen the window length using :kbd:`home` and
#     :kbd:`end` keys, or start with a specific window duration by passing the
#     ``duration`` parameter.
#
#   - You can scroll in the time domain using the :kbd:`←` and
#     :kbd:`→` arrow keys, or start at a specific point by passing the
#     ``start`` parameter. Scrolling using :kbd:`shift`:kbd:`→` or
#     :kbd:`shift`:kbd:`←` scrolls a full window width at a time.
#
# - It allows clicking on channels to mark/unmark as "bad".
#
#   - When the plot window is closed, the `~mne.io.Raw` object's
#     ``info`` attribute will be updated, adding or removing the newly
#     (un)marked channels to/from the `~mne.Info` object's ``bads``
#     field (A.K.A. ``raw.info['bads']``).
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
#     default by passing the ``proj=False`` parameter. See
#     :ref:`tut-projectors-background` for more info on projectors.
#
# These and other keyboard shortcuts are listed in the Help window, accessed
# through the ``Help`` button at the lower left corner of the plot window.
# Other plot properties (such as color of the channel traces, channel order and
# grouping, simultaneous plotting of :term:`events`, scaling, clipping,
# filtering, etc.) can also be adjusted through parameters passed to the
# `~mne.io.Raw.plot` method; see the docstring for details.
#
#
# Plotting spectral density of continuous data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To visualize the frequency content of continuous data, the `~mne.io.Raw`
# object provides a :meth:`~mne.io.Raw.compute_psd` method to compute
# `spectral density`_ and the resulting :class:`~mne.time_frequency.Spectrum`
# object has a :meth:`~mne.time_frequency.Spectrum.plot` method:

spectrum = raw.compute_psd()
spectrum.plot(average=True)

# %%
# If the data have been filtered, vertical dashed lines will automatically
# indicate filter boundaries. The spectrum for each channel type is drawn in
# its own subplot; here we've passed the ``average=True`` parameter to get a
# summary for each channel type, but it is also possible to plot each channel
# individually, with options for how the spectrum should be computed,
# color-coding the channels by location, and more. For example, here is a plot
# of just a few sensors (specified with the ``picks`` parameter), color-coded
# by spatial location (via the ``spatial_colors`` parameter, see the
# documentation of `~mne.time_frequency.Spectrum.plot` for full details):

midline = ['EEG 002', 'EEG 012', 'EEG 030', 'EEG 048', 'EEG 058', 'EEG 060']
spectrum.plot(picks=midline)

# %%
# It is also possible to plot spectral power estimates across sensors as a
# scalp topography, using the :class:`~mne.time_frequency.Spectrum`'s
# :meth:`~mne.time_frequency.Spectrum.plot_topomap` method. The default
# parameters will plot five frequency bands (δ, θ, α, β, γ), will compute power
# based on magnetometer channels (if present), and will plot the power
# estimates on a dB-like log-scale:

spectrum.plot_topomap()

# %%
# Alternatively, you can plot the PSD for every sensor on its own axes, with
# the axes arranged spatially to correspond to sensor locations in space, using
# `~mne.time_frequency.Spectrum.plot_topo`:

spectrum.plot_topo()

# %%
# This plot is also interactive; hovering over each "thumbnail" plot will
# display the channel name in the bottom left of the plot window, and clicking
# on a thumbnail plot will create a second figure showing a larger version of
# the selected channel's spectral density (as if you had called
# `~mne.time_frequency.Spectrum.plot` with that channel passed as ``picks``).
#
# By default, `~mne.time_frequency.Spectrum.plot_topo` will show only the MEG
# channels if MEG channels are present; if only EEG channels are found, they
# will be plotted instead:

spectrum.pick('eeg').plot_topo()

# %%
# .. note::
#
#    Prior to the addition of the :class:`~mne.time_frequency.Spectrum` class,
#    the above plots were possible via::
#
#        raw.plot_psd(average=True)
#        raw.plot_psd_topo()
#        raw.pick('eeg').plot_psd_topo()
#
#    (there was no ``plot_topomap`` method for :class:`~mne.io.Raw`). The
#    :meth:`~mne.io.Raw.plot_psd` and :meth:`~mne.io.Raw.plot_psd_topo` methods
#    of :class:`~mne.io.Raw` objects are still provided to support legacy
#    analysis scripts, but new code should instead use the
#    :class:`~mne.time_frequency.Spectrum` object API.
#
#
# Plotting sensor locations from ``Raw`` objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The channel locations in a `~mne.io.Raw` object can be easily plotted
# with the `~mne.io.Raw.plot_sensors` method. A brief example is shown
# here; notice that channels in ``raw.info['bads']`` are plotted in red. More
# details and additional examples are given in the tutorial
# :ref:`tut-sensor-locations`.

raw.plot_sensors(ch_type='eeg')

# %%
# .. _`tut-section-raw-plot-proj`:
#
# Plotting projectors from ``Raw`` objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As seen in the output of `mne.io.read_raw_fif` above, there are
# :term:`projectors <projector>` included in the example `~mne.io.Raw`
# file (representing environmental noise in the signal, so it can later be
# "projected out" during preprocessing). You can visualize these projectors
# using the `~mne.io.Raw.plot_projs_topomap` method. By default it will
# show one figure per channel type for which projectors are present, and each
# figure will have one subplot per projector. The three projectors in this file
# were only computed for magnetometers, so one figure with three subplots is
# generated. More details on working with and plotting projectors are given in
# :ref:`tut-projectors-background` and :ref:`tut-artifact-ssp`.

raw.plot_projs_topomap(colorbar=True)

# %%
# .. LINKS
#
# .. _spectral density: https://en.wikipedia.org/wiki/Spectral_density
