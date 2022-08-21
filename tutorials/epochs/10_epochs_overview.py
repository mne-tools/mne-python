# -*- coding: utf-8 -*-
"""
.. _tut-epochs-class:

=============================================
The Epochs data structure: discontinuous data
=============================================

This tutorial covers the basics of creating and working with :term:`epoched
<epochs>` data. It introduces the :class:`~mne.Epochs` data structure in
detail, including how to load, query, subselect, export, and plot data from an
:class:`~mne.Epochs` object. For more information about visualizing
:class:`~mne.Epochs` objects, see :ref:`tut-visualize-epochs`. For info on
creating an :class:`~mne.Epochs` object from (possibly simulated) data in a
:class:`NumPy array <numpy.ndarray>`, see :ref:`tut-creating-data-structures`.

As usual we'll start by importing the modules we need:
"""

# %%

import mne

# %%
# :class:`~mne.Epochs` objects are a data structure for representing and
# analyzing equal-duration chunks of the EEG/MEG signal. :class:`~mne.Epochs`
# are most often used to represent data that is time-locked to repeated
# experimental events (such as stimulus onsets or subject button presses), but
# can also be used for storing sequential or overlapping frames of a continuous
# signal (e.g., for analysis of resting-state activity; see
# :ref:`fixed-length-events`). Inside an :class:`~mne.Epochs` object, the data
# are stored in an :class:`array <numpy.ndarray>` of shape ``(n_epochs,
# n_channels, n_times)``.
#
# :class:`~mne.Epochs` objects have many similarities with :class:`~mne.io.Raw`
# objects, including:
#
# - They can be loaded from and saved to disk in ``.fif`` format, and their
#   data can be exported to a :class:`NumPy array <numpy.ndarray>` through the
#   :meth:`~mne.Epochs.get_data` method or to a :class:`Pandas DataFrame
#   <pandas.DataFrame>` through the :meth:`~mne.Epochs.to_data_frame` method.
#
# - Both :class:`~mne.Epochs` and :class:`~mne.io.Raw` objects support channel
#   selection by index or name, including :meth:`~mne.Epochs.pick`,
#   :meth:`~mne.Epochs.pick_channels` and :meth:`~mne.Epochs.pick_types`
#   methods.
#
# - :term:`SSP projector <projector>` manipulation is possible through
#   :meth:`~mne.Epochs.add_proj`, :meth:`~mne.Epochs.del_proj`, and
#   :meth:`~mne.Epochs.plot_projs_topomap` methods.
#
# - Both :class:`~mne.Epochs` and :class:`~mne.io.Raw` objects have
#   :meth:`~mne.Epochs.copy`, :meth:`~mne.Epochs.crop`,
#   :meth:`~mne.Epochs.time_as_index`, :meth:`~mne.Epochs.filter`, and
#   :meth:`~mne.Epochs.resample` methods.
#
# - Both :class:`~mne.Epochs` and :class:`~mne.io.Raw` objects have
#   :attr:`~mne.Epochs.times`, :attr:`~mne.Epochs.ch_names`,
#   :attr:`~mne.Epochs.proj`, and :class:`info <mne.Info>` attributes.
#
# - Both :class:`~mne.Epochs` and :class:`~mne.io.Raw` objects have built-in
#   plotting methods :meth:`~mne.Epochs.plot`, :meth:`~mne.Epochs.plot_psd`,
#   and :meth:`~mne.Epochs.plot_psd_topomap`.
#
#
# Creating Epoched data from a ``Raw`` object
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The example dataset we've been using thus far doesn't include pre-epoched
# data, so in this section we'll load the continuous data and create epochs
# based on the events recorded in the :class:`~mne.io.Raw` object's STIM
# channels. As we often do in these tutorials, we'll :meth:`~mne.io.Raw.crop`
# the :class:`~mne.io.Raw` data to save memory:

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = (sample_data_folder / 'MEG' / 'sample' /
                        'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False).crop(tmax=60)

# %%
# As we saw in the :ref:`tut-events-vs-annotations` tutorial, we can extract an
# events array from :class:`~mne.io.Raw` objects using :func:`mne.find_events`:

events = mne.find_events(raw, stim_channel='STI 014')

# %%
# .. note::
#
#     We could also have loaded the events from file, using
#     :func:`mne.read_events`::
#
#         sample_data_events_file = os.path.join(sample_data_folder,
#                                                'MEG', 'sample',
#                                                'sample_audvis_raw-eve.fif')
#         events_from_file = mne.read_events(sample_data_events_file)
#
#     See :ref:`tut-section-events-io` for more details.
#
#
# The :class:`~mne.io.Raw` object and the events array are the bare minimum
# needed to create an :class:`~mne.Epochs` object, which we create with the
# :class:`mne.Epochs` class constructor. However, you will almost surely want
# to change some of the other default parameters. Here we'll change ``tmin``
# and ``tmax`` (the time relative to each event at which to start and end each
# epoch). Note also that the :class:`~mne.Epochs` constructor accepts
# parameters ``reject`` and ``flat`` for rejecting individual epochs based on
# signal amplitude. See the :ref:`tut-reject-epochs-section` section for
# examples.

epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7)

# %%
# You'll see from the output that:
#
# - all 320 events were used to create epochs
#
# - baseline correction was automatically applied (by default, baseline is
#   defined as the time span from ``tmin`` to ``0``, but can be customized with
#   the ``baseline`` parameter)
#
# - no additional metadata was provided (see :ref:`tut-epochs-metadata` for
#   details)
#
# - the projection operators present in the :class:`~mne.io.Raw` file were
#   copied over to the :class:`~mne.Epochs` object
#
# If we print the :class:`~mne.Epochs` object, we'll also see a note that the
# epochs are not copied into memory by default, and a count of the number of
# epochs created for each integer Event ID.

print(epochs)

# %%
# Notice that the Event IDs are in quotes; since we didn't provide an event
# dictionary, the :class:`mne.Epochs` constructor created one automatically and
# used the string representation of the integer Event IDs as the dictionary
# keys. This is more clear when viewing the ``event_id`` attribute:

print(epochs.event_id)

# %%
# This time let's pass ``preload=True`` and provide an event dictionary; our
# provided dictionary will get stored as the ``event_id`` attribute and will
# make referencing events and pooling across event types easier:

event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'buttonpress': 32}
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict,
                    preload=True)
print(epochs.event_id)
del raw  # we're done with raw, free up some memory

# %%
# Notice that the output now mentions "1 bad epoch dropped". In the tutorial
# section :ref:`tut-reject-epochs-section` we saw how you can specify channel
# amplitude criteria for rejecting epochs, but here we haven't specified any
# such criteria. In this case, it turns out that the last event was too close
# the end of the (cropped) raw file to accommodate our requested ``tmax`` of
# 0.7 seconds, so the final epoch was dropped because it was too short. Here
# are the ``drop_log`` entries for the last 4 epochs (empty lists indicate
# epochs that were *not* dropped):

print(epochs.drop_log[-4:])

# %%
# .. note::
#
#     If you forget to provide the event dictionary to the :class:`~mne.Epochs`
#     constructor, you can add it later by assigning to the ``event_id``
#     attribute::
#
#         epochs.event_id = event_dict
#
#
# Basic visualization of ``Epochs`` objects
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The :class:`~mne.Epochs` object can be visualized (and browsed interactively)
# using its :meth:`~mne.Epochs.plot` method:

epochs.plot(n_epochs=10)

# %%
# Notice that the individual epochs are sequentially numbered along the bottom
# axis and are separated by vertical dashed lines.
# Epoch plots are interactive (similar to :meth:`raw.plot()
# <mne.io.Raw.plot>`) and have many of the same interactive controls as
# :class:`~mne.io.Raw` plots. Horizontal and vertical scrollbars allow browsing
# through epochs or channels (respectively), and pressing :kbd:`?` when the
# plot is focused will show a help screen with all the available controls. See
# :ref:`tut-visualize-epochs` for more details (as well as other ways of
# visualizing epoched data).
#
#
# .. _tut-section-subselect-epochs:
#
# Subselecting epochs
# ^^^^^^^^^^^^^^^^^^^
#
# Now that we have our :class:`~mne.Epochs` object with our descriptive event
# labels added, we can subselect epochs easily using square brackets. For
# example, we can load all the "catch trials" where the stimulus was a face:

print(epochs['face'])

# %%
# We can also pool across conditions easily, thanks to how MNE-Python handles
# the ``/`` character in epoch labels (using what is sometimes called
# "tag-based indexing"):

# pool across left + right
print(epochs['auditory'])
assert len(epochs['auditory']) == (len(epochs['auditory/left']) +
                                   len(epochs['auditory/right']))
# pool across auditory + visual
print(epochs['left'])
assert len(epochs['left']) == (len(epochs['auditory/left']) +
                               len(epochs['visual/left']))

# %%
# You can also pool conditions by passing multiple tags as a list. Note that
# MNE-Python will not complain if you ask for tags not present in the object,
# as long as it can find *some* match: the below example is parsed as
# (inclusive) ``'right'`` **or** ``'bottom'``, and you can see from the output
# that it selects only ``auditory/right`` and ``visual/right``.

print(epochs[['right', 'bottom']])

# %%
# However, if no match is found, an error is returned:

try:
    print(epochs[['top', 'bottom']])
except KeyError:
    print('Tag-based selection with no matches raises a KeyError!')

# %%
# Selecting epochs by index
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :class:`~mne.Epochs` objects can also be indexed with integers, :term:`slices
# <slice>`, or lists of integers. This method of selection ignores event
# labels, so if you want the first 10 epochs of a particular type, you can
# select the type first, then use integers or slices:

print(epochs[:10])    # epochs 0-9
print(epochs[1:8:2])  # epochs 1, 3, 5, 7

print(epochs['buttonpress'][:4])            # first 4 "buttonpress" epochs
print(epochs['buttonpress'][[0, 1, 2, 3]])  # same as previous line

# %%
# Selecting, dropping, and reordering channels
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# You can use the :meth:`~mne.Epochs.pick`, :meth:`~mne.Epochs.pick_channels`,
# :meth:`~mne.Epochs.pick_types`, and :meth:`~mne.Epochs.drop_channels` methods
# to modify which channels are included in an :class:`~mne.Epochs` object. You
# can also use :meth:`~mne.Epochs.reorder_channels` for this purpose; any
# channel names not provided to :meth:`~mne.Epochs.reorder_channels` will be
# dropped. Note that these *channel* selection methods modify the object
# in-place (unlike the square-bracket indexing to select *epochs* seen above)
# so in interactive/exploratory sessions you may want to create a
# :meth:`~mne.Epochs.copy` first.

epochs_eeg = epochs.copy().pick_types(meg=False, eeg=True)
print(epochs_eeg.ch_names)

new_order = ['EEG 002', 'STI 014', 'EOG 061', 'MEG 2521']
epochs_subset = epochs.copy().reorder_channels(new_order)
print(epochs_subset.ch_names)

# %%

del epochs_eeg, epochs_subset

# %%
# Changing channel name and type
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# You can change the name or type of a channel using
# :meth:`~mne.Epochs.rename_channels` or :meth:`~mne.Epochs.set_channel_types`.
# Both methods take :class:`dictionaries <dict>` where the keys are existing
# channel names, and the values are the new name (or type) for that channel.
# Existing channels that are not in the dictionary will be unchanged.

epochs.rename_channels({'EOG 061': 'BlinkChannel'})

epochs.set_channel_types({'EEG 060': 'ecg'})
print(list(zip(epochs.ch_names, epochs.get_channel_types()))[-4:])

# %%

# let's set them back to the correct values before moving on
epochs.rename_channels({'BlinkChannel': 'EOG 061'})
epochs.set_channel_types({'EEG 060': 'eeg'})

# %%
# Selection in the time domain
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To change the temporal extent of the :class:`~mne.Epochs`, you can use the
# :meth:`~mne.Epochs.crop` method:

shorter_epochs = epochs.copy().crop(tmin=-0.1, tmax=0.1, include_tmax=True)

for name, obj in dict(Original=epochs, Cropped=shorter_epochs).items():
    print('{} epochs has {} time samples'
          .format(name, obj.get_data().shape[-1]))

# %%
# Cropping removed part of the baseline. When printing the
# cropped :class:`~mne.Epochs`, MNE-Python will inform you about the time
# period that was originally used to perform baseline correction by displaying
# the string "baseline period cropped after baseline correction":

print(shorter_epochs)

# %%
# However, if you wanted to *expand* the time domain of an :class:`~mne.Epochs`
# object, you would need to go back to the :class:`~mne.io.Raw` data and
# recreate the :class:`~mne.Epochs` with different values for ``tmin`` and/or
# ``tmax``.
#
# It is also possible to change the "zero point" that defines the time values
# in an :class:`~mne.Epochs` object, with the :meth:`~mne.Epochs.shift_time`
# method. :meth:`~mne.Epochs.shift_time` allows shifting times relative to the
# current values, or specifying a fixed time to set as the new time value of
# the first sample (deriving the new time values of subsequent samples based on
# the :class:`~mne.Epochs` object's sampling frequency).

# shift times so that first sample of each epoch is at time zero
later_epochs = epochs.copy().shift_time(tshift=0., relative=False)
print(later_epochs.times[:3])

# shift times by a relative amount
later_epochs.shift_time(tshift=-7, relative=True)
print(later_epochs.times[:3])

# %%

del shorter_epochs, later_epochs

# %%
# Note that although time shifting respects the sampling frequency (the spacing
# between samples), it does not enforce the assumption that there is a sample
# occurring at exactly time=0.
#
#
# Extracting data in other forms
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The :meth:`~mne.Epochs.get_data` method returns the epoched data as a
# :class:`NumPy array <numpy.ndarray>`, of shape ``(n_epochs, n_channels,
# n_times)``; an optional ``picks`` parameter selects a subset of channels by
# index, name, or type:

eog_data = epochs.get_data(picks='EOG 061')
meg_data = epochs.get_data(picks=['mag', 'grad'])
channel_4_6_8 = epochs.get_data(picks=slice(4, 9, 2))

for name, arr in dict(EOG=eog_data, MEG=meg_data, Slice=channel_4_6_8).items():
    print('{} contains {} channels'.format(name, arr.shape[1]))

# %%
# Note that if your analysis requires repeatedly extracting single epochs from
# an :class:`~mne.Epochs` object, ``epochs.get_data(item=2)`` will be much
# faster than ``epochs[2].get_data()``, because it avoids the step of
# subsetting the :class:`~mne.Epochs` object first.
#
# You can also export :class:`~mne.Epochs` data to :class:`Pandas DataFrames
# <pandas.DataFrame>`. Here, the :class:`~pandas.DataFrame` index will be
# constructed by converting the time of each sample into milliseconds and
# rounding it to the nearest integer, and combining it with the event types and
# epoch numbers to form a hierarchical :class:`~pandas.MultiIndex`. Each
# channel will appear in a separate column. Then you can use any of Pandas'
# tools for grouping and aggregating data; for example, here we select any
# epochs numbered 10 or less from the ``auditory/left`` condition, and extract
# times between 100 and 107 ms on channels ``EEG 056`` through ``EEG 058``
# (note that slice indexing within Pandas' :obj:`~pandas.DataFrame.loc` is
# inclusive of the endpoint):

df = epochs.to_data_frame(index=['condition', 'epoch', 'time'])
df.sort_index(inplace=True)
print(df.loc[('auditory/left', slice(0, 10), slice(100, 107)),
             'EEG 056':'EEG 058'])

del df

# %%
# See the :ref:`tut-epochs-dataframe` tutorial for many more examples of the
# :meth:`~mne.Epochs.to_data_frame` method.
#
#
# Loading and saving ``Epochs`` objects to disk
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :class:`~mne.Epochs` objects can be loaded and saved in the ``.fif`` format
# just like :class:`~mne.io.Raw` objects, using the :func:`mne.read_epochs`
# function and the :meth:`~mne.Epochs.save` method. Functions are also
# available for loading data that was epoched outside of MNE-Python, such as
# :func:`mne.read_epochs_eeglab` and :func:`mne.read_epochs_kit`.

epochs.save('saved-audiovisual-epo.fif', overwrite=True)
epochs_from_file = mne.read_epochs('saved-audiovisual-epo.fif', preload=False)

# %%
# The MNE-Python naming convention for epochs files is that the file basename
# (the part before the ``.fif`` or ``.fif.gz`` extension) should end with
# ``-epo`` or ``_epo``, and a warning will be issued if the filename you
# provide does not adhere to that convention.
#
# As a final note, be aware that the class of the epochs object is different
# when epochs are loaded from disk rather than generated from a
# :class:`~mne.io.Raw` object:

print(type(epochs))
print(type(epochs_from_file))

# %%
# In almost all cases this will not require changing anything about your code.
# However, if you need to do type checking on epochs objects, you can test
# against the base class that these classes are derived from:

print(all([isinstance(epochs, mne.BaseEpochs),
           isinstance(epochs_from_file, mne.BaseEpochs)]))

# %%
# Iterating over ``Epochs``
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Iterating over an :class:`~mne.Epochs` object will yield :class:`arrays
# <numpy.ndarray>` rather than single-trial :class:`~mne.Epochs` objects:

for epoch in epochs[:3]:
    print(type(epoch))

# %%
# If you want to iterate over :class:`~mne.Epochs` objects, you can use an
# integer index as the iterator:

for index in range(3):
    print(type(epochs[index]))
