# -*- coding: utf-8 -*-
"""
.. _epochs-intro-tutorial:

Working with epoched data
=========================

.. include:: ../../tutorial_links.inc

This tutorial covers creating epoched data from continuous data, subselecting
epochs from an Epochs object, and loading and saving Epochs objects to disk.
"""

###############################################################################
# :term:`Epoched data <epochs>` is data that has been divided into (possibly
# discontinuous) chunks of equal duration. Epoching is a common step in most
# event-related analysis pipelines, so naturally the :class:`~mne.Epochs`
# object plays an important role in most MNE-Python workflows.
#
# :class:`~mne.Epochs` objects have many similarities with :class:`~mne.io.Raw`
# objects, including:
#
# - they can be loaded from and saved to disk in ``.fif`` format, and their
#   data can be exported to a NumPy array through a
#   :meth:`~mne.Epochs.get_data` method or to a Pandas DataFrame through a
#   :meth:`~mne.Epochs.to_data_frame` method
#
# - channel selection by index or name, including :meth:`~mne.Epochs.pick`,
#   :meth:`~mne.Epochs.pick_channels` and :meth:`~mne.Epochs.pick_types`
#   methods
#
# - projector manipulation through :meth:`~mne.Epochs.add_proj`,
#   :meth:`~mne.Epochs.del_proj`, and :meth:`~mne.Epochs.plot_projs_topomap`
#
# - :meth:`~mne.Epochs.copy`, :meth:`~mne.Epochs.crop`,
#   :meth:`~mne.Epochs.time_as_index`, :meth:`~mne.Epochs.filter`, and
#   :meth:`~mne.Epochs.resample` methods
#
# - :attr:`~mne.Epochs.times`, :attr:`~mne.Epochs.ch_names`,
#   :attr:`~mne.Epochs.proj`, and info attributes
#
# - built-in plotting methods :meth:`~mne.Epochs.plot`,
#   :meth:`~mne.Epochs.plot_psd`, and :meth:`~mne.Epochs.plot_psd_topomap`
#
# The example data we've been using thus far doesn't include pre-epoched
# data, so we'll load the continuous data and creating epochs based on the
# events in the :class:`~mne.io.Raw` object's STIM channels.

import os
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

###############################################################################
# As we saw in the :ref:`events-tutorial` tutorial, we can extract an events
# array from :class:`~mne.io.Raw` objects using :func:`mne.find_events`, and we
# can make referencing events and pooling across event types easier if we make
# an event dictionary:

events = mne.find_events(raw, stim_channel='STI 014')
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'button': 32}

###############################################################################
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
#
# Creating an :class:`~mne.Epochs` object
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The :class:`~mne.io.Raw` object and the events array are the bare minimum
# needed to create an :class:`~mne.Epochs` object, which we create with the
# :class:`mne.Epochs` class constructor. However, you will almost surely want
# to change some of the other default parameters. Here we'll change ``tmin``
# and ``tmax`` (the time relative to each event at which to start and end each
# epoch):

epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7)

###############################################################################
# You'll see from the output that:
#
# - all 320 events were used to create epochs
#
# - baseline correction was automatically applied (by default, baseline is
#   defined as the time span from ``tmin`` to ``0``, but can be customized with
#   the `` baseline`` parameter)
#
# - no additional metadata was provided
#
# - the projection operators present in the :class:`~mne.io.Raw` file were
#   copied over to the :class:`~mne.Epochs` object
#
# If we print the :class:`~mne.Epochs` object, we'll also see a note that the
# epochs are not copied into memory by default, and a count of the number of
# epochs created for each integer Event ID.

print(epochs)

###############################################################################
# Notice that the Event IDs are in quotes; since we didn't provide an event
# dictionary, the :class:`mne.Epochs` constructor created one automatically and
# used the string representation of the integer Event IDs as the dictionary
# keys. This is more clear when viewing the ``event_id`` attribute:

print(epochs.event_id)

###############################################################################
# This time let's pass ``preload=True`` and provide the event dictionary; our
# provided dictionary will get stored as the ``event_id`` attribute:

epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict,
                    preload=True)
print(epochs.event_id)

###############################################################################
# .. note::
#
#     If you forget to provide the event dictionary to the :class:`~mne.Epochs`
#     constructor, you can add it later by assigning to the ``event_id``
#     attribute::
#
#         epochs.event_id = event_dict
#
#
# .. _epoch-pooling:
#
# Subselecting Epochs
# ^^^^^^^^^^^^^^^^^^^
#
# Now that we have our :class:`~mne.Epochs` object with our descriptive event
# labels added, we can subselect epochs easily using square brackets. For
# example, we can load all the "catch trials" where the stimulus was a face:

print(epochs['face'])

###############################################################################
# We can also pool across conditions easily, thanks to how MNE-Python handles
# the ``/`` character in epoch labels:

# pool across left + right
print(epochs['auditory'])
assert len(epochs['auditory']) == (len(epochs['auditory/left']) +
                                   len(epochs['auditory/right']))
# pool across auditory + visual
print(epochs['left'])
assert len(epochs['left']) == (len(epochs['auditory/left']) +
                               len(epochs['visual/left']))

###############################################################################
# :class:`~mne.Epochs` objects can also be indexed with integers. For example,
# this selects the first 10 epochs (regardless of their event labels):

print(epochs[:10])

###############################################################################
# Loading and saving epoched data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :class:`~mne.Epochs` objects can be loaded and saved in the ``.fif`` format
# just like :class:`~mne.io.Raw` objects, using the :func:`mne.read_epochs`
# function and the :meth:`~mne.Epochs.save` method. Functions are also
# available for loading data that was epoched outside of MNE-Python, such as
# :func:`mne.read_epochs_eeglab` and :func:`mne.read_epochs_kit`.

epochs.save('saved-audiovisual-epo.fif')
epochs_from_file = mne.read_epochs('saved-audiovisual-epo.fif')

###############################################################################
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

###############################################################################
# In almost all cases this will not require changing anything about your code.
# However, if you need to do type checking on epochs objects, you can test
# against the base class that these classes are derived from:

print(all([isinstance(epochs, mne.BaseEpochs),
           isinstance(epochs_from_file, mne.BaseEpochs)]))
