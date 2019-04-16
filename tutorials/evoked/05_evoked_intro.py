# -*- coding: utf-8 -*-
"""
.. _evoked-intro-tutorial:

The Evoked data class
=====================

.. include:: ../../tutorial_links.inc

This tutorial covers creating Evoked objects from epoched data, basic plotting
of evoked data, and loading and saving Evoked objects to disk.
"""

###############################################################################
# We'll start by importing the modules we need, loading some example data, and
# epoching it:

import os
import numpy as np
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)
# get events
events = mne.find_events(raw, stim_channel='STI 014')
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'button': 32}
# make epochs
raw.info['bads'].append('EEG 007')
reject_criteria = dict(mag=4e-12, grad=4e-10, eeg=150e-6, eog=250e-6)
epochs = mne.Epochs(raw, events, event_id=event_dict, reject=reject_criteria,
                    preload=True)

###############################################################################
# :term:`Evoked data <evoked>` results from averaging several epochs together.
# Through averaging, random signal fluctuations are likely to "average out" to
# zero, allowing aspects of the signal that reflect neural/behavioral responses
# to the stimuli to stand out. Most commonly, evoked data are created from an
# :class:`~mne.Epochs` object, using the :meth:`~mne.Epochs.average` method.
# The :meth:`~mne.Epochs.average` method incorporates all epochs present in the
# :class:`~mne.Epochs` object, so to create different evoked objects for each
# condition, it is necessary to subselect the epochs first:

evoked_aud = epochs['auditory'].average()
evoked_vis = epochs['visual'].average()
print(evoked_aud)
print(evoked_vis)

###############################################################################
# One thing to notice right away is the *condition name* printed just after the
# "|" character in each object's summary representation. It tells us that there
# were multiple conditions (i.e., multiple event types, AKA multiple keys in
# the ``epochs.event_id`` dictionary) that were pooled when the epochs were
# averaged together, and it tells us the relative proportion of each condition
# comprising that average. This condition name is accessible through the
# :class:`~mne.Evoked` object's ``comment`` attribute:

print(evoked_aud.comment)

###############################################################################
# .. note::
#
#     If you want to equalize the number of epochs in each condition before
#     averaging, you can use :meth:`mne.Epochs.equalize_event_counts` (note
#     that :meth:`mne.Epochs.equalize_event_counts` operates in-place, so
#     consider using :meth:`~mne.Epochs.copy` first and assigning the result to
#     a new variable).
#
#
# Basic visualization of :class:`~mne.Evoked` objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Many of the attributes and methods we've seen on :class:`~mne.io.Raw` and
# :class:`~mne.Epochs` objects are also available for :class:`~mne.Evoked`
# objects (:attr:`~mne.Evoked.ch_names`, :meth:`~mne.Evoked.copy`,
# :meth:`~mne.Evoked.crop`, :meth:`~mne.Evoked.filter`,
# :meth:`~mne.Evoked.interpolate_bads`, :meth:`~mne.Evoked.pick_channels`,
# :meth:`~mne.Evoked.pick_types`, :meth:`~mne.Evoked.to_data_frame`, etc). Like
# :class:`~mne.io.Raw` and :class:`~mne.Epochs` objects, :class:`~mne.Evoked`
# objects also have a :meth:`~mne.Evoked.plot` method for quickly viewing the
# data (in this case, as butterfly plots):

evoked_aud.plot()
evoked_vis.plot()

###############################################################################
# You can see that separate subplots are created for each channel type. When
# created in an interactive Python session, these plots will be interactive;
# clicking on a sensor trace will show the channel name for that trace, and
# click-and-dragging a time span will pop open a new figure showing the average
# scalp topography for that time span. Several other ways of plotting evoked
# data are covered in :ref:`evoked-plotting-tutorial`.
#
# The :meth:`~mne.Epochs.average` method also has a ``picks`` parameter
# (allowing you to subselect specific channels or channel types):

epochs['auditory'].average('eeg').plot()
epochs['auditory'].average(picks=['EEG 001', 'EEG 002', 'EEG 003']).plot()

###############################################################################
# :meth:`~mne.Epochs.average` also has a ``method`` parameter determining how
# to combine the data from different epochs. The default ``method`` is "mean";
# "median" is also supported, or you can pass a callable function to ``method``
# (as long as it accepts a 3-dimensional array and aggregates along the first
# dimension; see the documentation of :meth:`~mne.Epochs.average` for details).
# For example, if we wanted the standard deviation across epochs, we could use
# ``numpy.std`` with the ``axis`` parameter set to ``0``:


def standard_deviation(epoch_data):
    return np.std(epoch_data, axis=0)

epochs['auditory'].average(method=standard_deviation).plot()

###############################################################################
# .. note::
#
#     A shortcut to writing the ``standard_deviation`` function above is the
#     ``partial`` function from the ``functools`` module. ``partial`` takes in
#     a function along with some of its parameters, and returns a version of
#     that function in which those parameters are "hard-coded". So instead of::
#
#         def standard_deviation(epoch_data):
#             return np.std(epoch_data, axis=0)
#
#     we could have written::
#
#         standard_deviation = partial(np.std, axis=0)
#
#
# Loading and saving :class:`~mne.Evoked` objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Like :class:`~mne.io.Raw` and :class:`~mne.Epochs` objects,
# :class:`~mne.Evoked` objects can be saved in the ``.fif`` format. The
# conventional filename for :class:`~mne.Evoked` objects is ``-ave.fif`` or
# ``-ave.fif.gz`` (reflecting the fact that :class:`~mne.Evoked` objects are
# *averages* of epochs).

mne.write_evokeds('saved-audvis-ave.fif', evoked_aud)

###############################################################################
# You can store multiple evoked "datasets" in one ``.fif`` file, by passing a
# list of :class:`~mne.Evoked` objects to the :func:`~mne.write_evokeds`
# function. Note, however, that the ``.fif`` format can only contain one copy
# of the measurement info fields, so it is not recommended to store
# :class:`~mne.Evoked` objects from different subjects / recordings in the same
# ``.fif`` file.

mne.write_evokeds('saved-audvis-ave.fif', [evoked_aud, evoked_vis])

###############################################################################
# When loading ``.fif`` files containing multiple datasets, the default is to
# load *all* datasets and return them as a list of :class:`~mne.Evoked`
# objects. However, it is possible to selectively load only the dataset(s) you
# want by specifying the ``condition`` parameter of :func:`~mne.read_evokeds`.
# ``condition`` can be an integer index (or list of integers) to select based
# on the order in which the :class:`~mne.Evoked` objects were saved in the
# ``.fif`` file, or a string (or list of strings) giving the condition name(s)
# you want to load.
#
# One potential pitfall is that :func:`mne.write_evokeds` sets condition names
# from the :class:`~mne.Evoked` object's ``comment`` attribute, which (as we
# saw above) may be rather unweildy if multiple conditions were averaged
# together. If you anticipate saving multiple :class:`~mne.Evoked` objects to a
# single ``.fif`` file, consider changing the ``comment`` attributes of each
# :class:`~mne.Evoked` object before saving, to make selective loading easier,
# as shown below.

print(f'The auto-generated condition names are: "{evoked_aud.comment}" and '
      f'"{evoked_vis.comment}"')
print('=' * 79)
# change condition names to make selective loading easier:
evoked_aud.comment = 'foo'
evoked_vis.comment = 'bar'
# write both evokeds to same file:
mne.write_evokeds('saved-audvis-ave.fif', [evoked_aud, evoked_vis])
# now load just one of them:
evoked_aud_from_file = mne.read_evokeds('saved-audvis-ave.fif',
                                        condition='foo')
print('=' * 79)
print(evoked_aud_from_file)
