"""
.. _tut-sensor-locations:

Working with sensor locations
=============================

This tutorial describes how to read and plot sensor locations, and how
the physical location of sensors is handled in MNE-Python.

.. contents:: Page contents
   :local:
   :depth: 2

As usual we'll start by importing the modules we need and loading some
:ref:`example data <sample-dataset>`:
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
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
# *montages*. :class:`Layouts <mne.channels.Layout>` give sensor positions in 2
# dimensions (defined by ``x``, ``y``, ``width``, and ``height`` values for
# each sensor), and are primarily used for illustrative purposes (i.e., making
# diagrams of approximate sensor positions in top-down diagrams of the head).
# In contrast, :class:`montages <mne.channels.DigMontage>` contain sensor
# positions in 3D (``x``, ``y``, ``z``, in meters). Many layout and montage
# files are included during MNE-Python installation, and are stored in your
# ``mne-python`` directory, in the :file:`mne/channels/data/layouts` and
# :file:`mne/channels/data/montages` folders, respectively:

data_dir = os.path.join(os.path.dirname(mne.__file__), 'channels', 'data')
for subfolder in ['layouts', 'montages']:
    print('\nBUILT-IN {} FILES'.format(subfolder[:-1].upper()))
    print('======================')
    print(sorted(os.listdir(os.path.join(data_dir, subfolder))))

###############################################################################
# .. sidebar:: Computing sensor locations
#
#     If you are interested in how standard ("idealized") EEG sensor positions
#     are computed on a spherical head model, the `eeg_positions`_ repository
#     provides code and documentation to this end.
#
# As you may be able to tell from the filenames shown above, the included
# montage files are all for EEG systems. These are *idealized* sensor positions
# based on a spherical head model. Montage files for MEG systems are not
# provided because the 3D coordinates of MEG sensors are included in the raw
# recordings from MEG systems, and are automatically stored in the ``info``
# attribute of the :class:`~mne.io.Raw` file upon loading. In contrast, layout
# files *are* included for MEG systems (to facilitate easy plotting of MEG
# sensor location diagrams).
#
# You may also have noticed that the file formats and filename extensions of
# layout and montage files vary considerably. This reflects different
# manufacturers' conventions; to simplify this, the montage and layout loading
# functions in MNE-Python take the filename *without its extension* so you
# don't have to keep track of which file format is used by which manufacturer.
# Examples of this can be seen in the following sections.
#
# If you have digitized the locations of EEG sensors on the scalp during your
# recording session (e.g., with a Polhemus Fastrak digitizer), these can be
# loaded in MNE-Python as :class:`~mne.channels.DigMontage` objects; see
# :ref:`reading-dig-montages` (below).
#
#
# Working with layout files
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To load a layout file, use the :func:`mne.channels.read_layout`
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
# picking channels by index (not by name or by type). Here we find the indices
# we want using :func:`numpy.where`; selection by name or type is possible via
# :func:`mne.pick_channels` or :func:`mne.pick_types`.

midline = np.where([name.endswith('z') for name in biosemi_layout.names])[0]
biosemi_layout.plot(picks=midline)

###############################################################################
# If you're working with a :class:`~mne.io.Raw` object that already has sensor
# positions incorporated, you can create a :class:`~mne.channels.Layout` object
# with either the :func:`mne.channels.make_eeg_layout` function or
# (equivalently) the :func:`mne.channels.find_layout` function.

layout_from_raw = mne.channels.make_eeg_layout(raw.info)
# same result as: mne.channels.find_layout(raw.info, ch_type='eeg')
layout_from_raw.plot()

###############################################################################
# .. note::
#
#     There is no corresponding ``make_meg_layout`` function because sensor
#     locations are fixed in a MEG system (unlike in EEG, where the sensor caps
#     deform to fit each subject's head). Thus MEG layouts are consistent for a
#     given system and you can simply load them with
#     :func:`mne.channels.read_layout`, or use :func:`mne.channels.find_layout`
#     with the ``ch_type`` parameter, as shown above for EEG.
#
# All :class:`~mne.channels.Layout` objects have a
# :meth:`~mne.channels.Layout.save` method that allows writing layouts to disk,
# in either :file:`.lout` or :file:`.lay` format (which format gets written is
# inferred from the file extension you pass to the method's ``fname``
# parameter). The choice between :file:`.lout` and :file:`.lay` format only
# matters if you need to load the layout file in some other software
# (MNE-Python can read either format equally well).
#
#
# Working with montage files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Built-in montages are loaded and plotted in a very similar way to layouts.
# However, the :meth:`~mne.channels.DigMontage.plot` method of
# :class:`~mne.channels.DigMontage` objects has some additional parameters,
# such as whether to display channel names or just points (the ``show_names``
# parameter) and whether to display sensor positions in 3D or as a 2D topomap
# (the ``kind`` parameter):

ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
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
ax3d = fig.add_subplot(122, projection='3d')
raw.plot_sensors(ch_type='eeg', axes=ax2d)
raw.plot_sensors(ch_type='eeg', axes=ax3d, kind='3d')
ax3d.view_init(azim=70, elev=15)

###############################################################################
# .. _reading-dig-montages:
#
# Reading sensor digitization files
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# It's probably evident from the 2D topomap above that there is some
# irregularity in the EEG sensor positions in the :ref:`sample dataset
# <sample-dataset>` — this is because the sensor positions in that dataset are
# digitizations of the sensor positions on an actual subject's head. Depending
# on what system was used to scan the positions one can use different
# reading functions (see :ref:`dig-formats`).
# The read :class:`montage <mne.channels.DigMontage>` can then be added
# to :class:`~mne.io.Raw` objects with the :meth:`~mne.io.Raw.set_montage`
# method; in the sample data this was done prior to saving the
# :class:`~mne.io.Raw` object to disk, so the sensor positions are already
# incorporated into the ``info`` attribute of the :class:`~mne.io.Raw` object.
# See the documentation of the reading functions and
# :meth:`~mne.io.Raw.set_montage` for further details. Once loaded,
# locations can be plotted with :meth:`~mne.channels.DigMontage.plot` and
# saved with :meth:`~mne.channels.DigMontage.save`, like when working
# with a standard montage.
#
# The possibilities to read in digitized montage files are summarized
# in :ref:`dig-formats`.
#
# .. note::
#
#     When setting a montage with :meth:`~mne.io.Raw.set_montage`
#     the measurement info is updated at two places (the `chs`
#     and `dig` entries are updated). See :ref:`tut-info-class`.
#     `dig` will potentially contain more than channel locations,
#     such HPI, head shape points or fiducials 3D coordinates.
#
# Rendering sensor position with mayavi
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# It is also possible to render an image of a MEG sensor helmet in 3D, using
# mayavi instead of matplotlib, by calling the :func:`mne.viz.plot_alignment`
# function:

fig = mne.viz.plot_alignment(raw.info, trans=None, dig=False, eeg=False,
                             surfaces=[], meg=['helmet', 'sensors'],
                             coord_frame='meg')
mne.viz.set_3d_view(fig, azimuth=50, elevation=90, distance=0.5)

###############################################################################
# :func:`~mne.viz.plot_alignment` requires an :class:`~mne.Info` object, and
# can also render MRI surfaces of the scalp, skull, and brain (by passing
# keywords like ``'head'``, ``'outer_skull'``, or ``'brain'`` to the
# ``surfaces`` parameter) making it useful for :ref:`assessing coordinate frame
# transformations <plot_source_alignment>`. For examples of various uses of
# :func:`~mne.viz.plot_alignment`, see
# :doc:`../../auto_examples/visualization/plot_montage`,
# :doc:`../../auto_examples/visualization/plot_eeg_on_scalp`, and
# :doc:`../../auto_examples/visualization/plot_meg_sensors`.
#
# .. LINKS
#
# .. _`eeg_positions`: https://github.com/sappelhoff/eeg_positions
