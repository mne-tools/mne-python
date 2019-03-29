# -*- coding: utf-8 -*-
"""
.. _plotting-epochs-tutorial:

Visualizing epoched data
========================

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
# and many of the same actions are possible: you can scroll through time with
# :kbd:`left` and :kbd:`right` arrows, you can scroll through channels with
# :kbd:`up` and :kbd:`down` arrows, change the temporal span of the window with
# :kbd:`home` and :kbd:`end`, change the number of visible channels with
# :kbd:`page up` and :kbd:`page down`, you can click on a channel name along
# the left axis to mark that channel as "bad", and you can mark individual
# epochs as "bad" by clicking on the data within that epoch.
#
# TODO plot_psd, plot_psd_topomap, plot_image, plot_topo_image
