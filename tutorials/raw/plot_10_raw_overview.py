# -*- coding: utf-8 -*-
"""
.. _tut-raw-class:

The Raw data structure: continuous data
=======================================

This tutorial covers the basics of working with raw EEG/MEG data in Python. It
introduces the :class:`~mne.io.Raw` data structure in detail, including how to
load, query, subselect, export, and plot data from a :class:`~mne.io.Raw`
object. For more info on visualization of :class:`~mne.io.Raw` objects, see
:ref:`tut-visualize-raw`. For info on creating a :class:`~mne.io.Raw` object
from simulated data in a :class:`NumPy array <numpy.ndarray>`, see
:ref:`tut_creating_data_structures`.

.. contents:: Page contents
   :local:
   :depth: 2

As usual we'll start by importing the modules we need:
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne

###############################################################################
# Loading continuous data
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# .. sidebar:: Datasets in MNE-Python
#
#     There are ``data_path`` functions for several example datasets in
#     MNE-Python (e.g., :func:`mne.datasets.kiloword.data_path`,
#     :func:`mne.datasets.spm_face.data_path`, etc). All of them will check the
#     default download location first to see if the dataset is already on your
#     computer, and only download it if necessary.  The default download
#     location is also configurable; see the documentation of any of the
#     ``data_path`` functions for more information.
#
# As mentioned in :ref:`the introductory tutorial <tut-overview>`,
# MNE-Python data structures are based around
# the :file:`.fif` file format from Neuromag. This tutorial uses an
# :ref:`example dataset <sample-dataset>` in :file:`.fif` format, so here we'll
# use the function :func:`mne.io.read_raw_fif` to load the raw data; there are
# reader functions for :ref:`a wide variety of other data formats
# <data-formats>` as well.
#
# There are also :ref:`several other example datasets
# <datasets>` that can be downloaded with just a few lines
# of code. Functions for downloading example datasets are in the
# :mod:`mne.datasets` submodule; here we'll use
# :func:`mne.datasets.sample.data_path` to download the ":ref:`sample-dataset`"
# dataset, which contains EEG, MEG, and structural MRI data from one subject
# performing an audiovisual experiment. When it's done downloading,
# :func:`~mne.datasets.sample.data_path` will return the folder location where
# it put the files; you can navigate there with your file browser if you want
# to examine the files yourself. Once we have the file path, we can load the
# data with :func:`~mne.io.read_raw_fif`. This will return a
# :class:`~mne.io.Raw` object, which we'll store in a variable called ``raw``.

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)

###############################################################################
# As you can see above, :func:`~mne.io.read_raw_fif` automatically displays
# some information about the file it's loading. For example, here it tells us
# that there are three "projection items" in the file along with the recorded
# data; those are :term:`SSP projectors <projector>` calculated to remove
# environmental noise from the MEG signals, and are discussed in a the tutorial
# :ref:`tut-projectors-background`.
# In addition to the information displayed during loading, you can
# get a glimpse of the basic details of a :class:`~mne.io.Raw` object by
# printing it:

print(raw)

###############################################################################
# By default, the :samp:`mne.io.read_raw_{*}` family of functions will *not*
# load the data into memory (instead the data on disk are `memory-mapped`_,
# meaning the data are only read from disk as-needed). Some operations (such as
# filtering) require that the data be copied into RAM; to do that we could have
# passed the ``preload=True`` parameter to :func:`~mne.io.read_raw_fif`, but we
# can also copy the data into RAM at any time using the
# :meth:`~mne.io.Raw.load_data` method. However, since this particular tutorial
# doesn't do any serious analysis of the data, we'll first
# :meth:`~mne.io.Raw.crop` the :class:`~mne.io.Raw` object to 60 seconds so it
# uses less memory and runs more smoothly on our documentation server.

raw.crop(tmax=60)

###############################################################################
# Querying the Raw object
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# .. sidebar:: Attributes vs. Methods
#
#     **Attributes** are usually static properties of Python objects — things
#     that are pre-computed and stored as part of the object's representation
#     in memory. Attributes are accessed with the ``.`` operator and do not
#     require parentheses after the attribute name (example: ``raw.ch_names``).
#
#     **Methods** are like specialized functions attached to an object.
#     Usually they require additional user input and/or need some computation
#     to yield a result. Methods always have parentheses at the end; additional
#     arguments (if any) go inside those parentheses (examples:
#     ``raw.estimate_rank()``, ``raw.drop_channels(['EEG 030', 'MEG 2242'])``).
#
# We saw above that printing the :class:`~mne.io.Raw` object displays some
# basic information like the total number of channels, the number of time
# points at which the data were sampled, total duration, and the approximate
# size in memory. Much more information is available through the various
# *attributes* and *methods* of the :class:`~mne.io.Raw` class. Some useful
# attributes of :class:`~mne.io.Raw` objects include a list of the channel
# names (:attr:`~mne.io.Raw.ch_names`), an array of the sample times in seconds
# (:attr:`~mne.io.Raw.times`), and the total number of samples
# (:attr:`~mne.io.Raw.n_times`); a list of all attributes and methods is given
# in the documentation of the :class:`~mne.io.Raw` class.
#
#
# The ``Raw.info`` attribute
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# There is also quite a lot of information stored in the ``raw.info``
# attribute, which stores an :class:`~mne.Info` object that is similar to a
# :class:`Python dictionary <dict>` (in that it has fields accessed via named
# keys). Like Python dictionaries, ``raw.info`` has a ``.keys()`` method that
# shows all the available field names; unlike Python dictionaries, printing
# ``raw.info`` will print a nicely-formatted glimpse of each field's data. See
# :ref:`tut-info-class` for more on what is stored in :class:`~mne.Info`
# objects, and how to interact with them.

n_time_samps = raw.n_times
time_secs = raw.times
ch_names = raw.ch_names
n_chan = len(ch_names)  # note: there is no raw.n_channels attribute
print('the (cropped) sample data object has {} time samples and {} channels.'
      ''.format(n_time_samps, n_chan))
print('The last time sample is at {} seconds.'.format(time_secs[-1]))
print('The first few channel names are {}.'.format(', '.join(ch_names[:3])))
print()  # insert a blank line in the output

# some examples of raw.info:
print('bad channels:', raw.info['bads'])  # chs marked "bad" during acquisition
print(raw.info['sfreq'], 'Hz')            # sampling frequency
print(raw.info['description'], '\n')      # miscellaneous acquisition info

print(raw.info)

###############################################################################
# .. note::
#
#     Most of the fields of ``raw.info`` reflect metadata recorded at
#     acquisition time, and should not be changed by the user. There are a few
#     exceptions (such as ``raw.info['bads']`` and ``raw.info['projs']``), but
#     in most cases there are dedicated MNE-Python functions or methods to
#     update the :class:`~mne.Info` object safely (such as
#     :meth:`~mne.io.Raw.add_proj` to update ``raw.info['projs']``).
#
# .. _`time-as-index`:
#
# Time, sample number, and sample index
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. sidebar:: Sample numbering in VectorView data
#
#     For data from VectorView systems, it is important to distinguish *sample
#     number* from *sample index*. See :term:`first_samp` for more information.
#
# One method of :class:`~mne.io.Raw` objects that is frequently useful is
# :meth:`~mne.io.Raw.time_as_index`, which converts a time (in seconds) into
# the integer index of the sample occurring closest to that time. The method
# can also take a list or array of times, and will return an array of indices.
#
# It is important to remember that there may not be a data sample at *exactly*
# the time requested, so the number of samples between ``time = 1`` second and
# ``time = 2`` seconds may be different than the number of samples between
# ``time = 2`` and ``time = 3``:

print(raw.time_as_index(20))
print(raw.time_as_index([20, 30, 40]), '\n')

print(np.diff(raw.time_as_index([1, 2, 3])))

###############################################################################
# Modifying ``Raw`` objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. sidebar:: ``len(raw)``
#
#     Although the :class:`~mne.io.Raw` object underlyingly stores data samples
#     in a :class:`NumPy array <numpy.ndarray>` of shape (n_channels,
#     n_timepoints), the :class:`~mne.io.Raw` object behaves differently from
#     :class:`NumPy arrays <numpy.ndarray>` with respect to the :func:`len`
#     function. ``len(raw)`` will return the number of timepoints (length along
#     data axis 1), not the number of channels (length along data axis 0).
#     Hence in this section you'll see ``len(raw.ch_names)`` to get the number
#     of channels.
#
# :class:`~mne.io.Raw` objects have a number of methods that modify the
# :class:`~mne.io.Raw` instance in-place and return a reference to the modified
# instance. This can be useful for `method chaining`_
# (e.g., ``raw.crop(...).pick_channels(...).filter(...).plot()``)
# but it also poses a problem during interactive analysis: if you modify your
# :class:`~mne.io.Raw` object for an exploratory plot or analysis (say, by
# dropping some channels), you will then need to re-load the data (and repeat
# any earlier processing steps) to undo the channel-dropping and try something
# else. For that reason, the examples in this section frequently use the
# :meth:`~mne.io.Raw.copy` method before the other methods being demonstrated,
# so that the original :class:`~mne.io.Raw` object is still available in the
# variable ``raw`` for use in later examples.
#
#
# Selecting, dropping, and reordering channels
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Altering the channels of a :class:`~mne.io.Raw` object can be done in several
# ways. As a first example, we'll use the :meth:`~mne.io.Raw.pick_types` method
# to restrict the :class:`~mne.io.Raw` object to just the EEG and EOG channels:

eeg_and_eog = raw.copy().pick_types(meg=False, eeg=True, eog=True)
print(len(raw.ch_names), '→', len(eeg_and_eog.ch_names))

###############################################################################
# Similar to the :meth:`~mne.io.Raw.pick_types` method, there is also the
# :meth:`~mne.io.Raw.pick_channels` method to pick channels by name, and a
# corresponding :meth:`~mne.io.Raw.drop_channels` method to remove channels by
# name:

raw_temp = raw.copy()
print('Number of channels in raw_temp:')
print(len(raw_temp.ch_names), end=' → drop two → ')
raw_temp.drop_channels(['EEG 037', 'EEG 059'])
print(len(raw_temp.ch_names), end=' → pick three → ')
raw_temp.pick_channels(['MEG 1811', 'EEG 017', 'EOG 061'])
print(len(raw_temp.ch_names))

###############################################################################
# If you want the channels in a specific order (e.g., for plotting),
# :meth:`~mne.io.Raw.reorder_channels` works just like
# :meth:`~mne.io.Raw.pick_channels` but also reorders the channels; for
# example, here we pick the EOG and frontal EEG channels, putting the EOG
# first and the EEG in reverse order:

channel_names = ['EOG 061', 'EEG 003', 'EEG 002', 'EEG 001']
eog_and_frontal_eeg = raw.copy().reorder_channels(channel_names)
print(eog_and_frontal_eeg.ch_names)

###############################################################################
# Changing channel name and type
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. sidebar:: Long channel names
#
#     Due to limitations in the :file:`.fif` file format (which MNE-Python uses
#     to save :class:`~mne.io.Raw` objects), channel names are limited to a
#     maximum of 15 characters.
#
# You may have noticed that the EEG channel names in the sample data are
# numbered rather than labelled according to a standard nomenclature such as
# the `10-20 <ten_twenty_>`_ or `10-05 <ten_oh_five_>`_ systems, or perhaps it
# bothers you that the channel names contain spaces. It is possible to rename
# channels using the :meth:`~mne.io.Raw.rename_channels` method, which takes a
# Python dictionary to map old names to new names. You need not rename all
# channels at once; provide only the dictionary entries for the channels you
# want to rename. Here's a frivolous example:

raw.rename_channels({'EOG 061': 'blink detector'})

###############################################################################
# This next example replaces spaces in the channel names with underscores,
# using a Python `dict comprehension`_:

print(raw.ch_names[-3:])
channel_renaming_dict = {name: name.replace(' ', '_') for name in raw.ch_names}
raw.rename_channels(channel_renaming_dict)
print(raw.ch_names[-3:])

###############################################################################
# If for some reason the channel types in your :class:`~mne.io.Raw` object are
# inaccurate, you can change the type of any channel with the
# :meth:`~mne.io.Raw.set_channel_types` method. The method takes a
# :class:`dictionary <dict>` mapping channel names to types; allowed types are
# ``ecg, eeg, emg, eog, exci, ias, misc, resp, seeg, stim, syst, ecog, hbo,
# hbr``. A common use case for changing channel type is when using frontal EEG
# electrodes as makeshift EOG channels:

raw.set_channel_types({'EEG_001': 'eog'})
print(raw.copy().pick_types(meg=False, eog=True).ch_names)

###############################################################################
# Selection in the time domain
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# If you want to limit the time domain of a :class:`~mne.io.Raw` object, you
# can use the :meth:`~mne.io.Raw.crop` method, which modifies the
# :class:`~mne.io.Raw` object in place (we've seen this already at the start of
# this tutorial, when we cropped the :class:`~mne.io.Raw` object to 60 seconds
# to reduce memory demands). :meth:`~mne.io.Raw.crop` takes parameters ``tmin``
# and ``tmax``, both in seconds (here we'll again use :meth:`~mne.io.Raw.copy`
# first to avoid changing the original :class:`~mne.io.Raw` object):

raw_selection = raw.copy().crop(tmin=10, tmax=12.5)
print(raw_selection)

###############################################################################
# :meth:`~mne.io.Raw.crop` also modifies the :attr:`~mne.io.Raw.first_samp` and
# :attr:`~mne.io.Raw.times` attributes, so that the first sample of the cropped
# object now corresponds to ``time = 0``. Accordingly, if you wanted to re-crop
# ``raw_selection`` from 11 to 12.5 seconds (instead of 10 to 12.5 as above)
# then the subsequent call to :meth:`~mne.io.Raw.crop` should get ``tmin=1``
# (not ``tmin=11``), and leave ``tmax`` unspecified to keep everything from
# ``tmin`` up to the end of the object:

print(raw_selection.times.min(), raw_selection.times.max())
raw_selection.crop(tmin=1)
print(raw_selection.times.min(), raw_selection.times.max())

###############################################################################
# Remember that sample times don't always align exactly with requested ``tmin``
# or ``tmax`` values (due to sampling), which is why the ``max`` values of the
# cropped files don't exactly match the requested ``tmax`` (see
# :ref:`time-as-index` for further details).
#
# If you need to select discontinuous spans of a :class:`~mne.io.Raw` object —
# or combine two or more separate :class:`~mne.io.Raw` objects — you can use
# the :meth:`~mne.io.Raw.append` method:

raw_selection1 = raw.copy().crop(tmin=30, tmax=30.1)     # 0.1 seconds
raw_selection2 = raw.copy().crop(tmin=40, tmax=41.1)     # 1.1 seconds
raw_selection3 = raw.copy().crop(tmin=50, tmax=51.3)     # 1.3 seconds
raw_selection1.append([raw_selection2, raw_selection3])  # 2.5 seconds total
print(raw_selection1.times.min(), raw_selection1.times.max())

###############################################################################
# .. warning::
#
#     Be careful when concatenating :class:`~mne.io.Raw` objects from different
#     recordings, especially when saving: :meth:`~mne.io.Raw.append` only
#     preserves the ``info`` attribute of the initial :class:`~mne.io.Raw`
#     object (the one outside the :meth:`~mne.io.Raw.append` method call).
#
#
# Extracting data from ``Raw`` objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# So far we've been looking at ways to modify a :class:`~mne.io.Raw` object.
# This section shows how to extract the data from a :class:`~mne.io.Raw` object
# into a :class:`NumPy array <numpy.ndarray>`, for analysis or plotting using
# functions outside of MNE-Python. To select portions of the data,
# :class:`~mne.io.Raw` objects can be indexed using square brackets. However,
# indexing :class:`~mne.io.Raw` works differently than indexing a :class:`NumPy
# array <numpy.ndarray>` in two ways:
#
# 1. Along with the requested sample value(s) MNE-Python also returns an array
#    of times (in seconds) corresponding to the requested samples. The data
#    array and the times array are returned together as elements of a tuple.
#
# 2. The data array will always be 2-dimensional even if you request only a
#    single time sample or a single channel.
#
#
# Extracting data by index
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# To illustrate the above two points, let's select a couple seconds of data
# from the first channel:

sampling_freq = raw.info['sfreq']
start_stop_seconds = np.array([11, 13])
start_sample, stop_sample = (start_stop_seconds * sampling_freq).astype(int)
channel_index = 0
raw_selection = raw[channel_index, start_sample:stop_sample]
print(raw_selection)

###############################################################################
# You can see that it contains 2 arrays. This combination of data and times
# makes it easy to plot selections of raw data (although note that we're
# transposing the data array so that each channel is a column instead of a row,
# to match what matplotlib expects when plotting 2-dimensional ``y`` against
# 1-dimensional ``x``):

x = raw_selection[1]
y = raw_selection[0].T
plt.plot(x, y)

###############################################################################
# Extracting channels by name
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The :class:`~mne.io.Raw` object can also be indexed with the names of
# channels instead of their index numbers. You can pass a single string to get
# just one channel, or a list of strings to select multiple channels. As with
# integer indexing, this will return a tuple of ``(data_array, times_array)``
# that can be easily plotted. Since we're plotting 2 channels this time, we'll
# add a vertical offset to one channel so it's not plotted right on top
# of the other one:

# sphinx_gallery_thumbnail_number = 2
channel_names = ['MEG_0712', 'MEG_1022']
two_meg_chans = raw[channel_names, start_sample:stop_sample]
y_offset = np.array([5e-11, 0])  # just enough to separate the channel traces
x = two_meg_chans[1]
y = two_meg_chans[0].T + y_offset
lines = plt.plot(x, y)
plt.legend(lines, channel_names)

###############################################################################
# Extracting channels by type
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# There are several ways to select all channels of a given type from a
# :class:`~mne.io.Raw` object. The safest method is to use
# :func:`mne.pick_types` to obtain the integer indices of the channels you
# want, then use those indices with the square-bracket indexing method shown
# above. The :func:`~mne.pick_types` function uses the :class:`~mne.Info`
# attribute of the :class:`~mne.io.Raw` object to determine channel types, and
# takes boolean or string parameters to indicate which type(s) to retain. The
# ``meg`` parameter defaults to ``True``, and all others default to ``False``,
# so to get just the EEG channels, we pass ``eeg=True`` and ``meg=False``:

eeg_channel_indices = mne.pick_types(raw.info, meg=False, eeg=True)
eeg_data, times = raw[eeg_channel_indices]
print(eeg_data.shape)

###############################################################################
# Some of the parameters of :func:`mne.pick_types` accept string arguments as
# well as booleans. For example, the ``meg`` parameter can take values
# ``'mag'``, ``'grad'``, ``'planar1'``, or ``'planar2'`` to select only
# magnetometers, all gradiometers, or a specific type of gradiometer. See the
# docstring of :meth:`mne.pick_types` for full details.
#
#
# The ``Raw.get_data()`` method
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# If you only want the data (not the corresponding array of times),
# :class:`~mne.io.Raw` objects have a :meth:`~mne.io.Raw.get_data` method. Used
# with no parameters specified, it will extract all data from all channels, in
# a (n_channels, n_timepoints) :class:`NumPy array <numpy.ndarray>`:

data = raw.get_data()
print(data.shape)

###############################################################################
# If you want the array of times, :meth:`~mne.io.Raw.get_data` has an optional
# ``return_times`` parameter:

data, times = raw.get_data(return_times=True)
print(data.shape)
print(times.shape)

###############################################################################
# The :meth:`~mne.io.Raw.get_data` method can also be used to extract specific
# channel(s) and sample ranges, via its ``picks``, ``start``, and ``stop``
# parameters. The ``picks`` parameter accepts integer channel indices, channel
# names, or channel types, and preserves the requested channel order given as
# its ``picks`` parameter.

first_channel_data = raw.get_data(picks=0)
eeg_and_eog_data = raw.get_data(picks=['eeg', 'eog'])
two_meg_chans_data = raw.get_data(picks=['MEG_0712', 'MEG_1022'],
                                  start=1000, stop=2000)

print(first_channel_data.shape)
print(eeg_and_eog_data.shape)
print(two_meg_chans_data.shape)

###############################################################################
# Summary of ways to extract data from ``Raw`` objects
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The following table summarizes the various ways of extracting data from a
# :class:`~mne.io.Raw` object.
#
# .. cssclass:: table-bordered
# .. rst-class:: midvalign
#
# +-------------------------------------+-------------------------+
# | Python code                         | Result                  |
# |                                     |                         |
# |                                     |                         |
# +=====================================+=========================+
# | ``raw.get_data()``                  | :class:`NumPy array     |
# |                                     | <numpy.ndarray>`        |
# |                                     | (n_chans × n_samps)     |
# +-------------------------------------+-------------------------+
# | ``raw[:]``                          | :class:`tuple` of (data |
# +-------------------------------------+ (n_chans × n_samps),    |
# | ``raw.get_data(return_times=True)`` | times (1 × n_samps))    |
# +-------------------------------------+-------------------------+
# | ``raw[0, 1000:2000]``               |                         |
# +-------------------------------------+                         |
# | ``raw['MEG 0113', 1000:2000]``      |                         |
# +-------------------------------------+                         |
# | ``raw.get_data(picks=0,             | :class:`tuple` of       |
# | start=1000, stop=2000,              | (data (1 × 1000),       |
# | return_times=True)``                | times (1 × 1000))       |
# +-------------------------------------+                         |
# | ``raw.get_data(picks='MEG 0113',    |                         |
# | start=1000, stop=2000,              |                         |
# | return_times=True)``                |                         |
# +-------------------------------------+-------------------------+
# | ``raw[7:9, 1000:2000]``             |                         |
# +-------------------------------------+                         |
# | ``raw[[2, 5], 1000:2000]``          | :class:`tuple` of       |
# +-------------------------------------+ (data (2 × 1000),       |
# | ``raw[['EEG 030', 'EOG 061'],       | times (1 × 1000))       |
# | 1000:2000]``                        |                         |
# +-------------------------------------+-------------------------+
#
#
# Exporting and saving Raw objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :class:`~mne.io.Raw` objects have a built-in :meth:`~mne.io.Raw.save` method,
# which can be used to write a partially processed :class:`~mne.io.Raw` object
# to disk as a :file:`.fif` file, such that it can be re-loaded later with its
# various attributes intact (but see :ref:`precision` for an important
# note about numerical precision when saving).
#
# There are a few other ways to export just the sensor data from a
# :class:`~mne.io.Raw` object. One is to use indexing or the
# :meth:`~mne.io.Raw.get_data` method to extract the data, and use
# :func:`numpy.save` to save the data array:

data = raw.get_data()
np.save(file='my_data.npy', arr=data)

###############################################################################
# It is also possible to export the data to a :class:`Pandas DataFrame
# <pandas.DataFrame>` object, and use the saving methods that :mod:`Pandas
# <pandas>` affords. The :class:`~mne.io.Raw` object's
# :meth:`~mne.io.Raw.to_data_frame` method is similar to
# :meth:`~mne.io.Raw.get_data` in that it has a ``picks`` parameter for
# restricting which channels are exported, and ``start`` and ``stop``
# parameters for restricting the time domain. Note that, by default, times will
# be converted to milliseconds, rounded to the nearest millisecond, and used as
# the DataFrame index; see the ``scaling_time`` parameter in the documentation
# of :meth:`~mne.io.Raw.to_data_frame` for more details.

sampling_freq = raw.info['sfreq']
start_end_secs = np.array([10, 13])
start_sample, stop_sample = (start_end_secs * sampling_freq).astype(int)
df = raw.to_data_frame(picks=['eeg'], start=start_sample, stop=stop_sample)
# then save using df.to_csv(...), df.to_hdf(...), etc
print(df.head())

###############################################################################
# .. note::
#     When exporting data as a :class:`NumPy array <numpy.ndarray>` or
#     :class:`Pandas DataFrame <pandas.DataFrame>`, be sure to properly account
#     for the :ref:`unit of representation <units>` in your subsequent
#     analyses.
#
#
# .. LINKS
#
# .. _`method chaining`: https://en.wikipedia.org/wiki/Method_chaining
# .. _`memory-mapped`: https://en.wikipedia.org/wiki/Memory-mapped_file
# .. _ten_twenty: https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)
# .. _ten_oh_five: https://doi.org/10.1016%2FS1388-2457%2800%2900527-7
# .. _`dict comprehension`:
#    https://docs.python.org/3/tutorial/datastructures.html#dictionaries
