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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 lgtm[py/unused-import]
from mayavi import mlab
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

###############################################################################
# About montages and layouts
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# MNE-Python comes pre-loaded with information about the sensor positions of
# many MEG and EEG systems. This information is stored in *layout files* and
# *montages*, which are stored in your ``mne-python`` installation directory,
# in the ``mne/channels/data/layouts`` and ``mne/channels/data/montages``
# folders (respectively):

data_dir = os.path.join(os.path.dirname(mne.__file__), 'channels', 'data')
for subfolder in ['layouts', 'montages']:
    print(f'\nBUILT-IN {subfolder[:-1].upper()} FILES\n=====================')
    print(sorted(os.listdir(os.path.join(data_dir, subfolder))))

###############################################################################
# *Layouts* give sensor positions in 2 dimensions (defined by ``x``, ``y``,
# ``width``, and ``height`` values for each sensor), and are primarily used for
# illustrative purposes (i.e., making diagrams of approximate sensor
# positions). In contrast, *montages* give sensor positions in 3D (``x``,
# ``y``, ``z``, in meters). In addition, montages come in two "flavors" —
# :class:`~mne.channels.Montage` objects (for idealized sensor montages) and
# :class:`~mne.channels.DigMontage` (for subject-specific digitizations of
# sensor positions).
#
#
# Working with layout files
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To load a layout file, use the :func:`~mne.channels.read_layout`
# function, and provide the filename *without* its file extension. You can then
# visualize the layout using its :meth:`~mne.channels.Layout.plot` method, or
# (equivalently) by passing it to :func:`mne.viz.plot_layout`:

biosemi_layout = mne.channels.read_layout('biosemi')
biosemi_layout.plot()  # same result as: mne.viz.plot_layout(biosemi_layout)

###############################################################################
# Similar to the ``picks`` argument for selecting channels from
# :class:`~mne.io.Raw` objects, the :meth:`~mne.channels.Layout.plot` method of
# :class:`~mne.channels.Layout` objects also has a ``picks`` argument. However,
# because layouts only contain information about sensor name and location (not
# sensor type), the :meth:`~mne.channels.Layout.plot` method only allows
# picking channels by index (not by name or by type).

midline = np.where([name.endswith('z') for name in biosemi_layout.names])[0]
biosemi_layout.plot(picks=midline)

###############################################################################
# Working with montage files
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Built-in montages are loaded and plotted in a very similar way to layouts.
# However, the :meth:`~mne.channels.Montage.plot` method of
# :class:`~mne.channels.Montage` objects has some additional parameters, such
# as whether to display channel names or just points (the ``show_names``
# parameter) and whether to display sensor positions in 3D or as a 2D topomap
# (the ``kind`` parameter):

ten_twenty_montage = mne.channels.read_montage('standard_1020')
ten_twenty_montage.plot(show_names=False)
fig = ten_twenty_montage.plot(kind='3d')
fig.gca().view_init(azim=70, elev=15)

###############################################################################
# Similar functionality is also available with the
# :meth:`~mne.io.Raw.plot_sensors` method of :class:`~mne.io.Raw` objects,
# again with the option to plot in either 2D or 3D.
# :meth:`~mne.io.Raw.plot_sensors` also allows channel selection by type, can
# color-code channels in various ways (by default, channels listed in
# ``raw.info['bads']`` will be plotted in red), and allows drawing into an
# existing matplotlib ``axes`` object (so the channel positions can easily be
# made as a subplot in a multi-panel figure):

fig = plt.figure()
ax2d = fig.add_subplot(121)
ax3d = fig.add_subplot(122, projection='3d', aspect='equal')
raw.plot_sensors(ch_type='eeg', axes=ax2d)
raw.plot_sensors(ch_type='eeg', axes=ax3d, kind='3d')
ax3d.view_init(azim=70, elev=15)

###############################################################################
# Reading sensor digitization files
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# It's probably evident from the 2D topomap above that there is some
# irregularity in the sensor positions — this is because the sensor positions
# stored in the :class:`~mne.io.Raw` object are based on digitizations of the
# sensor positions on an actual subject's head. Sensor digitizations are read
# with :func:`mne.channels.read_dig_montage` and added to :class:`~mne.io.Raw`
# objects with the :meth:`~mne.io.Raw.set_montage` method. See the function
# documentation for further details.
#
# .. TODO sample data doesn't have separate .hsp or .elp files, so can't demo
#    this functionality
#
#
# Rendering sensor position with mayavi
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# It is also possible to render an image of the sensor helmet in 3D, using
# mayavi instead of matplotlib, by calling the :func:`mne.viz.plot_alignment`
# function:

fig = mne.viz.plot_alignment(raw.info, trans=None, dig=False, eeg=False,
                             surfaces=[], meg=['helmet', 'sensors'],
                             coord_frame='meg')
mlab.view(azimuth=50, elevation=90, distance=0.5)

###############################################################################
# :func:`~mne.viz.plot_alignment` requires an :class:`~mne.Info`
# object, and can also render MRI surfaces of the scalp, skull, and brain,
# making it useful for :ref:`assessing coordinate frame transformations
# <plot_source_alignment>`.
#
#
# TODO
# ^^^^
#
# - mne.channels.make_eeg_layout() ?
# - mne.channels.make_1020_channel_selections() ?
