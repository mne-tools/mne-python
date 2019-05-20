# -*- coding: utf-8 -*-
"""
Parsing events from raw data
============================

This tutorial describes how to read experimental events from raw recordings,
and how to convert between the two different representations of events within
MNE-Python (Events arrays and Annotations objects).

.. contents:: Page contents
   :local:
   :depth: 1

In the :ref:`introductory tutorial <overview-tut-events-section>` we saw an
example of reading experimental events from a :term:`"STIM" channel <stim
channel>`; here we'll discuss :term:`events` and :term:`annotations` more
broadly, give more detailed information about reading from STIM channels, and
give an example of reading events that are included in the data file as an
embedded array.

We'll begin by loading the Python modules we need, and loading the same
:ref:`example data <sample-dataset>` we used in the :doc:`introductory tutorial
<plot_introduction>`, but to save memory we'll crop the :class:`~mne.io.Raw`
object to just 60 seconds before loading it into RAM:
"""

import os
import numpy as np
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)
raw.crop(tmax=60).load_data()

###############################################################################
# The Events and Annotations data structures
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generally speaking, both the Events and :class:`~mne.Annotations` data
# structures serve the same purpose: they provide a mapping between times
# during an EEG/MEG recording and a description of what happened at those
# times. In other words, they associate a *when* with a *what*. The main
# differences are:
#
# 1. the Events data structure represents the *when* in terms of samples,
#    whereas the :class:`~mne.Annotations` data structure represents the *when*
#    in seconds.
# 2. the Events data structure represents the *what* as an integer "Event ID"
#    code, whereas the :class:`~mne.Annotations` data structure represents the
#    *what* as a string.
# 3. Events in an Event array do not have a duration (though it is possible to
#    represent duration with pairs of onset/offset events within an Events
#    array), whereas each element of an :class:`~mne.Annotations` object
#    necessarily includes a duration (though the duration can be zero if an
#    instantaneous event is desired).
# 4. Events are stored as an ordinary :class:`NumPy array <numpy.ndarray>`,
#    whereas :class:`~mne.Annotations` is a :class:`list`-like class defined in
#    MNE-Python.
#
#
# What is a STIM channel?
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Many EEG and MEG systems incorporate one or more recording channels that do
# not receive signals from a sensor; instead, those channels record voltages
# (usually short, rectangular DC pulses of fixed magnitudes sent from the
# experiment-controlling computer) that are time-locked to experimental events,
# such as the onset of a stimulus or a button-press response by the subject. In
# other cases, these pulses may not be strictly time-locked to an experimental
# event, but instead may occur in between trials to indicate the type of
# stimulus (or experimental condition) that is about to occur on the upcoming
# trial. In MNE-Python, those channels are referred to as :term:`"STIM"
# channels <stim channel>`.
#
# The DC pulses may be all on one STIM channel (in which case different
# experimental events or trial types are encoded as different voltage
# magnitudes), or they may be spread across several channels, in which case the
# channel(s) on which the pulse(s) occur can be used to encode different events
# or conditions. Even on systems with multiple STIM channels, there is often
# one channel that records a weighted sum of the other STIM channels, in such a
# way that voltage levels on that channel can be unambiguously decoded as
# particular event types. On older Neuromag systems (such as that used to
# record the sample data) this was typically the ``STI 014`` channel; on newer
# systems it is more commonly ``STI101``. You can see the STIM channels in the
# raw data file here:

raw.copy().pick_types(meg=False, stim=True).plot(start=3, duration=6)

###############################################################################
# Above, you can see that ``STI 014`` (the "sum channel") contains pulses of
# different magnitudes whereas pulses on other channels have consistent
# magnitudes. You can also see that every time there is a pulse on ``STI 001``
# through ``STI 006`` there is a corresponding pulse on ``STI 014``.
#
#
# Converting a STIM channel signal to an Events array
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# If your data has events recorded on a STIM channel, you can convert them into
# an events array using :func:`mne.find_events`. The sample number of the onset
# (or offset) of each pulse is recorded as the event time, the pulse magnitudes
# are converted into integers, and these pairs of sample numbers plus integer
# codes are stored in :class:`NumPy arrays <numpy.ndarray>` (usually called
# "the events array" or just "the events"). In its simplest form, the function
# requires only the :class:`~mne.io.Raw` object, and the name of the channel(s)
# from which to read events:

events = mne.find_events(raw, stim_channel='STI 014')
print(events[:5])  # show the first 5

###############################################################################
# .. sidebar:: The middle column of the Events array
#
#     MNE-Python events are actually *three* values: in between the sample
#     number and the integer event code is a value indicating what the event
#     code was on the immediately preceding sample. In practice, that value is
#     almost always `0`, but it can be used to detect the *endpoint* of an
#     event whose duration is longer than one sample. See the documentation of
#     :func:`mne.find_events` for more details.
#
# If you don't provide the name of a STIM channel, :func:`~mne.find_events`
# will first look for MNE-Python :doc:`config variables <plot_configuration>`
# for variables ``MNE_STIM_CHANNEL``, ``MNE_STIM_CHANNEL_1``, etc. If those are
# not found, channels ``STI 014`` and ``STI101`` are tried, followed by the
# first channel with type "STIM" present in ``raw.ch_names``. If you regularly
# work with data from several different MEG systems with different STIM channel
# names, setting the ``MNE_STIM_CHANNEL`` config variable may not be very
# useful, but for researchers whose data is all from a single system it can be
# a time-saver to configure that variable once and then forget about it.
#
# :func:`~mne.find_events` has several options, including options for aligning
# events to the onset or offset of the STIM channel pulses, setting the minimum
# pulse duration, and handling of consecutive pulses (with no return to zero
# between them). For example, you can effectively encode event duration by
# passing ``output='step'`` to :func:`mne.find_events`;see the documentation of
# :func:`~mne.find_events` for details. More information on working with events
# arrays (including how to plot, combine, load, and save event arrays) can be
# found in the tutorial :doc:`../raw/plot_events`.
#
#
# Reading embedded events as :class:`~mne.Annotations`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Some EEG/MEG systems generate files where events are stored in a separate
# data array rather than as pulses on one or more STIM channels. For example,
# the EEGLAB format stores events as a collection of arrays in the :file:`.set`
# file. When reading those files, MNE-Python will automatically convert the
# stored events into an :class:`~mne.Annotations` object and store it as the
# :attr:`~mne.io.Raw.annotations` attribute of the :class:`~mne.io.Raw` object:

testing_data_folder = mne.datasets.testing.data_path()
eeglab_raw_file = os.path.join(testing_data_folder, 'EEGLAB', 'test_raw.set')
eeglab_raw = mne.io.read_raw_eeglab(eeglab_raw_file)
print(eeglab_raw.annotations)

###############################################################################
# The core data within an :class:`~mne.Annotations` object is accessible
# through three of its attributes: ``onset``, ``duration``, and
# ``description``. Here we can see that there were 154 events stored in the
# EEGLAB file, they all had a duration of zero seconds, there were two
# different types of events, and the first event occurred about 1 second after
# the recording began:

print(len(eeglab_raw.annotations))
print(set(eeglab_raw.annotations.duration))
print(set(eeglab_raw.annotations.description))
print(eeglab_raw.annotations.onset[0])

###############################################################################
# More information on working with :class:`~mne.Annotations` objects, including
# how to add annotations to :class:`~mne.io.Raw` objects interactively, and how
# to plot, concatenate, load, save, and export :class:`~mne.Annotations`
# objects can be found in the tutorial :doc:`../raw/plot_annotating_raw`.
#
#
# Converting between Events arrays and Annotations objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Once your experimental events are read into MNE-Python (as either an Events
# array or an :class:`~mne.Annotations` object), you can easily convert between
# the two formats as needed. You might do this because, e.g., an Events array
# is needed for epoching continous data, or because you want to take advantage
# of the "annotation-aware" capability of some functions, that automatically
# omit spans of data if they overlap with certain annotations.
#
# To convert an :class:`~mne.Annotations` object to an Events array, use the
# function :func:`mne.events_from_annotations` on the :class:`~mne.io.Raw` file
# containing the annotations. This function will assign an integer Event ID to
# each unique element of ``raw.annotations.description``, and will return the
# mapping of descriptions to integer Event IDs along with the derived Event
# array. By default, one event will be created at the onset of each annotation;
# this can be modified via the ``chunk_duration`` parameter of
# :func:`~mne.events_from_annotations` to create equally spaced events within
# each annotation span. See the function documentation for further details.

events_from_annot, event_dict = mne.events_from_annotations(eeglab_raw)
print(event_dict)
print(events_from_annot[:5])

###############################################################################
# If you want to control which integers are mapped to each unique description
# value, you can pass a :class:`dict` specifying the mapping as the
# ``event_id`` parameter of :func:`~mne.events_from_annotations`; this
# :class:`dict` will be returned unmodified as the ``event_dict``.
#
# .. TODO add this when the other tutorial is nailed down:
#    Note that this ``event_dict`` can be used when creating
#    :class:`~mne.Epochs` from :class:`~mne.io.Raw` objects, as demonstrated
#    in :doc:`epoching_tutorial_whatever_its_name_is`.

custom_mapping = {'rt': 77, 'square': 42}
(events_from_annot,
 event_dict) = mne.events_from_annotations(raw, event_id=custom_mapping)
print(event_dict)
print(events_from_annot[:5])

###############################################################################
# To make the opposite conversion (from Events array to
# :class:`~mne.Annotations` object), you can create a mapping from integer
# Event ID to string descriptions, and use the :class:`~mne.Annotations`
# constructor to create the :class:`~mne.Annotations` object, and use the
# :meth:`~mne.io.Raw.set_annotations` method to add the annotations to the
# :class:`~mne.io.Raw` object:

mapping = {1: 'auditory/left', 2: 'auditory/right', 3: 'visual/left',
           4: 'visual/right', 5: 'smiley', 32: 'buttonpress'}
onsets = events[:, 0] * raw.info['sfreq']
durations = np.zeros_like(onsets)  # assumes instantaneous events
descriptions = [mapping[event_id] for event_id in events[:, 2]]
annot_from_events = mne.Annotations(onset=onsets, duration=durations,
                                    description=descriptions)
raw.set_annotations(annot_from_events)
