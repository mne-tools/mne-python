"""
.. _`sensor-locations-tutorial`:

Working with sensor locations
=============================

This tutorial describes how information about the physical location of sensors
is handled in MNE-Python, and the ways of visualizing sensor locations. As
always, we start by importing the Python modules we need, and loading some
example data:
"""

import os
from mayavi import mlab
import matplotlib.pyplot as plt
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

###############################################################################
# Montages and layout files
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# MNE-Python comes pre-loaded with information about the sensor layout of many
# MEG and EEG systems. This information is stored in *layout files*, which are
# stored in the ``mne/channels/data/layouts`` folder within your ``mne-python``
# installation directory:

install_dir = os.path.dirname(mne.__file__)
layout_dir = os.path.join(install_dir, 'channels', 'data', 'layouts')
os.listdir(layout_dir)

###############################################################################
# To load one of these layout files, use the :func:`~mne.channels.read_layout`
# function, and provide the filename *without* its file extension. You can then
# visualize the layout using its :meth:`~mne.channels.Layout.plot` method, or
# (equivalently) by passing it to :func:`mne.viz.plot_layout`:

vv_layout = mne.channels.read_layout('Vectorview-all')
vv_layout.plot()  # same result as: mne.viz.plot_layout(vv_layout)

###############################################################################
# Similar to the ``picks`` argument for selecting channels from
# :class:`~mne.io.Raw` objects, the :meth:`~mne.channels.Layout.plot` method of
# :class:`~mne.channels.Layout` objects also has a ``picks`` argument that
# allows you to select either specific channels (by name) or types of channels:

# vv_layout.plot(picks=['mag'])  # TODO: broken, issue #6030

###############################################################################
# Similar functionality is also available with the
# :meth:`~mne.io.Raw.plot_sensors` method of :class:`~mne.io.Raw` objects, with
# the option to plot in either 2D or 3D:

fig = plt.figure()
ax2d = fig.add_subplot(121)
ax3d = fig.add_subplot(122, projection='3d', aspect='equal')
raw.plot_sensors(ch_type='eeg', axes=ax2d)
raw.plot_sensors(ch_type='eeg', axes=ax3d, kind='3d')
ax3d.view_init(azim=30, elev=15)

###############################################################################
# It is also possible to render an image of the sensor helmet in 3D, using
# mayavi instead of matplotlib, by calling the :func:`~mne.viz.plot_alignment`
# function:

fig = mne.viz.plot_alignment(raw.info, trans=None, dig=False, eeg=False,
                             surfaces=[], meg=['helmet', 'sensors', 'ref'],
                             coord_frame='meg')
mlab.view(azimuth=50, elevation=90, distance=0.5)

###############################################################################
# TODO:
#
# - raw.set_montage()
# - mne.channels.read_dig_montage()
#
# .. sources to draw from:
#     https://mne-tools.github.io/dev/auto_examples/visualization/plot_meg_sensors.html
#     https://mne-tools.github.io/dev/manual/io.html
#     https://mne-tools.github.io/dev/auto_tutorials/plot_visualize_raw.html
