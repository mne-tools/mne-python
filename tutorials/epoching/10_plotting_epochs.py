# -*- coding: utf-8 -*-
"""
.. _plotting-epochs-tutorial:

Built-in plotting methods for :class:`~mne.Epochs` objects
==========================================================

.. include:: ../../tutorial_links.inc

This tutorial covers various ways of visualizing epoched data.
"""

###############################################################################
# As usual we'll start by importing the modules we need, loading some example
# data, and creating the epochs object:

import os
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)
# get events
events = mne.find_events(raw, stim_channel='STI 014')
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'button': 32}
# create epochs
epochs = mne.Epochs(raw, events, event_id=event_dict, preload=True)

###############################################################################
# You'll notice that this time we omitted the ``tmin`` and ``tmax`` parameters,
# which default to ``-0.2`` and ``0.5`` (respectively), so we'll end up with
# epochs that are 0.7 seconds in duration.
#
# Although :class:`~mne.Epochs` objects represent discontinuous data, they can
# be plotted end-to-end so they can be browsed just like :class:`~mne.io.Raw`
# objects:

epochs.plot()

###############################################################################
# The plot window looks quite similar to the plot window for continuous data,
# with some key differences:
#
# - epochs are separated by heavy dashed vertical lines, and a lighter solid
#   line indicates the ``t=0`` time of each epoch
#
# - the top edge of the plot area shows the integer Event ID, while the bottom
#   edge of the plot area shows the epoch index
#
# Many of the actions available in a continuous data plot window are also
# possible here: you can scroll through time with :kbd:`left` and :kbd:`right`
# arrows, you can scroll through channels with :kbd:`up` and :kbd:`down`
# arrows, change the temporal span of the window with :kbd:`home` and
# :kbd:`end`, you can change the number of visible channels with :kbd:`page up`
# and :kbd:`page down`, you can click on a channel name along the left axis to
# mark that channel as "bad", and you can mark individual epochs as "bad" by
# clicking on the data within that epoch.
#
# There are additional interactive features useful for data exploration: you
# can automatically generate a peak-to-peak histogram for each channel type by
# pressing :kbd:`h`. You can also right-click on a channel name to open an
# ERP/ERF image for that channel in another figure window (the equivalent of a
# call to :meth:`~mne.Epochs.plot_image` with the ``picks`` parameter
# determined by which channel name was clicked):

epochs.plot_image(picks='MEG 0213')

###############################################################################
# These image plots, where time is on the horizontal axis, each epoch is a
# single row of the image, and signal amplitude is indicated by color of each
# pixel, are also useful for identifying bad channels; for example, you can see
# here why channel ``EEG 053`` was marked as bad:

epochs.plot_image(picks='EEG 053')

###############################################################################
# Epoch image plots can be reproduced for all sensors at once using the
# :meth:`~mne.Epochs.plot_topo_image` method. By default, this will show only
# the MEG channels in the sample data:

epochs.plot_topo_image()

###############################################################################
# To display epoch image plots for EEG channels, you can pass an EEG layout to
# :meth:`~mne.Epochs.plot_topo_image` (see :ref:`sensor-locations-tutorial` for
# more info on layouts). If you don't want the black background, you can change
# it with the ``fig_facecolor`` parameter, though if you do that you may also
# want to change the ``font_color`` parameter at the same time, or else you may
# have trouble seeing the axis labels on the colorbar. You can also choose a
# different colormap with the ``cmap`` parameter:

eeg_layout = mne.channels.make_eeg_layout(raw.info)
epochs.plot_topo_image(layout=eeg_layout, fig_facecolor='w', font_color='k',
                       cmap='viridis')

###############################################################################
# These topo image plots are also interactive; hovering over each "thumbnail"
# imagemap will display the channel name in the bottom left of the plot window,
# and clicking on a thumbnail imagemap will create a second figure showing a
# larger version of the selected channel's imagemap (as if you had called
# :meth:`~mne.Epochs.plot_image` on that channel).
#
# Finally, you can visualize the frequency content of epoched data using
# :meth:`~mne.Epochs.plot_psd` to plot the `spectral density`_ of the epoched
# data for each channel type:

epochs.plot_psd()

###############################################################################
# ...or plot signal topography across the sensors within specific frequency
# bands, using :meth:`~mne.Epochs.plot_psd_topomap`:

epochs.plot_psd_topomap()
