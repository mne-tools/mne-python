# -*- coding: utf-8 -*-
"""
.. _events-tutorial:

Working with events
===================

.. include:: ../../tutorial_links.inc

This tutorial describes event representation and how event arrays are used to
subselect data.
"""

###############################################################################
# As usual we'll start by importing the modules we need, and loading some
# example data:

import os
import numpy as np
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

###############################################################################
# Many EEG and MEG systems incorporate one or more recording channels that do
# not receive signals from a sensor; instead, those channels record voltages
# (usually short, rectangular DC pulses of fixed magnitudes sent from the
# experiment-controlling computer) that are time-locked to experimental events
# such as the onset of a stimulus, or a button-press response by the subject.
# In other cases, these pulses may not be strictly time-locked to an
# experimental event, but instead may occur in between trials to indicate the
# type of stimulus (or experimental condition) that is about to occur on the
# upcoming trial.
#
# These DC pulses may be all on one channel (in which case different
# experimental events or trial types are encoded as different voltage
# magnitudes), or they may be spread across several channels, in which case the
# channel(s) on which the pulse(s) occur can be used to encode different events
# or conditions.
#
# In MNE-Python, those channels are referred to as "STIM" channels, the sample
# number of the onset (or offset) of each pulse is recorded as the event time,
# the pulse magnitudes are converted into integers, and these pairs of sample
# numbers plus integer codes are stored in NumPy arrays.
#
# .. note::
#
#     MNE-Python events are actually *three* values: in between the sample
#     number and the integer event code is a third value indicating what the
#     event code was on the immediately preceding sample. In practice, that
#     value is almost always `0` and is seldom used.
#
# On systems with multiple STIM channels, there is often one channel that
# records a binary sum of the other STIM channels (i.e., it adds the first stim
# channel to 2 times the second stim channel, 4 times the third stim channel,
# 8 times the fourth stim channel, etc). On older systems this was typically
# the ``STI 014`` channel; on newer systems it is typically called ``STI101``.
# The example data has that kind of setup:

raw.copy().pick_types(meg=False, stim=True).plot(start=3)

###############################################################################
# Above, you can see that ``STI 014`` (the "sum channel") contains pulses of
# different magnitudes whereas pulses on other channels have consistent
# magnitudes. You can also see that every time there is a pulse on ``STI 001``
# through ``STI 006`` there is a corresponding pulse on ``STI 014``.
#
#
# Reading events from a STIM channel
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# If your data has events recorded on a STIM channel, you can convert them into
# an events array using :func:`mne.find_events`. In its simplest form, the
# function requires only the :class:`~mne.io.Raw` object, and the name of the
# channel(s) from which to read events:

events = mne.find_events(raw, stim_channel='STI 014')
print(events[:5])  # show the first 5

###############################################################################
# :func:`~mne.find_events` has several options, including options for aligning
# events to the onset or offset of the STIM channel pulses, setting the minimum
# pulse duration, and handling of consecutive pulses (with no return to zero
# between them). See the function documentation for details.
#
# .. note::
#
#     If you don't provide the name of a STIM channel, :func:`~mne.find_events`
#     will first look for MNE-Python :ref:`config variables <config-tutorial>`
#     for variables ``MNE_STIM_CHANNEL``, ``MNE_STIM_CHANNEL_1``, etc. If those
#     are not found, channels ``STI 014`` and ``STI101`` are tried, followed by
#     the first STIM channel present in ``raw.ch_names``. If you regularly work
#     with data from several different MEG systems with different STIM channel
#     names, setting the ``MNE_STIM_CHANNEL`` config variable may not be very
#     useful, but for most researchers (whose data is all from a single system)
#     it can be a time-saver.
#
#
# Reading and writing events to file
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Event arrays are NumPy array objects, so they could be saved to disk as
# binary ``.npy`` files using ``numpy.save``. However, MNE-Python provides
# convenience functions :func:`mne.read_events` and :func:`mne.write_events`
# for reading and writing event arrays as either text files (common file
# extensions are ``.eve``, ``.lst``, and ``.txt``) or binary ``.fif`` files.
# The example dataset includes the results of ``mne.find_events(raw)`` in a
# ``.fif`` file:

sample_data_events_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                       'sample_audvis_raw-eve.fif')
events_from_file = mne.read_events(sample_data_events_file)
assert np.array_equal(events, events_from_file)

###############################################################################
# When writing event arrays to disk, the format will be inferred from the file
# extension you provide. By convention, MNE-Python expects events files to
# either have an ``.eve`` extension or to have a file basename ending in "-eve"
# or "_eve" (e.g., :file:`my_experiment_eve.fif`), and will issue a warning if
# this convention is not respected.
#
#
# Subselecting and combining events
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The output of :func:`~mne.find_events` above (copied here) told us the number
# of events that were found, and the unique integer event IDs present:
#
# .. code-block:: none
#
#     320 events found
#     Event IDs: [ 1  2  3  4  5 32]
#
# If some of those events are not of interest, you can easily subselect events
# using :func:`mne.pick_events`, which has parameters ``include`` and
# ``exclude``. For example, in the sample data Event ID 32 corresponds to a
# subject button press, which could be excluded as:

events_no_button = mne.pick_events(events, exclude=32)

###############################################################################
# It is also possible to combine two Event IDs using :func:`mne.merge_events`;
# the following example will combine Event IDs 1, 2 and 3 into a single event
# labelled ``1``:

merged_events = mne.merge_events(events, [1, 2, 3], 1)
print(np.unique(merged_events[:, -1]))

###############################################################################
# Note, however, that merging events is not necessary if you simply want to
# pool trial types for analysis; the next section describes how MNE-Python uses
# *event dictionaries* to map integer Event IDs to more descriptive label
# strings.
#
#
# Mapping Event IDs to trial descriptors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# So far we've only been dealing with integer Event IDs, which are assigned
# based on DC voltage pulse magnitude (which is ultimately determined by the
# experimenter's choices about what signals to send to the STIM channels).
# Keeping track of which Event ID corresponds to which experimental condition
# can be cumbersome, and it is often desirable to pool experimental conditions
# during analysis. For example, the sample data we're working with has five
# Event IDs indicating stimulus type, and one for subject button presses:
#
# +----------+-------------------------------------------+
# | Event ID | Condition                                 |
# +==========+===========================================+
# | 1        | auditory stimulus to the left ear         |
# +----------+-------------------------------------------+
# | 2        | auditory stimulus to the right ear        |
# +----------+-------------------------------------------+
# | 3        | visual stimulus to the left visual field  |
# +----------+-------------------------------------------+
# | 4        | visual stimulus to the right visual field |
# +----------+-------------------------------------------+
# | 5        | face (catch trial)                        |
# +----------+-------------------------------------------+
# | 32       | subject button press                      |
# +----------+-------------------------------------------+
#
# If we wanted to pool all auditory trials, instead of merging Event IDs 1 and
# 2, we can create an *event dictionary* that encodes the desired pooling.
# Event dictionaries have trial descriptors as keys, and integer Event IDs as
# values; crucially, the keys may contain multiple trial descriptors separated
# by ``/`` characters for easy pooling. For example:

event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'button': 32}

###############################################################################
# Event dictionaries like this one are used when extracting epochs from
# continuous data, and the resulting :class:`~mne.Epochs` object allows pooling
# by requesting partial trial descriptors (i.e., requesting ``'auditory'`` will
# select all epochs with Event IDs 1 and 2; requesting ``'left'`` will select
# all epochs with Event IDs 1 and 3). An example of this is shown later, in the
# :ref:`epoch-pooling` section of the :ref:`epochs-intro-tutorial` tutorial.
#
#
# Plotting events
# ^^^^^^^^^^^^^^^
#
# Another use of event dictionaries is when plotting events, which can serve as
# a useful check that your event signals were properly sent to the STIM
# channel(s) and that MNE-Python has successfully found them. The function
# :func:`mne.viz.plot_events` will plot each event versus its sample number
# (or, if you provide the sampling frequency, it will plot them versus time in
# seconds). It can also account for the offset between sample number and sample
# index in some MEG systems, with the ``first_samp`` parameter. If an event
# dictionary is provided, it will be used to generate a legend:

fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'],
                          first_samp=raw.first_samp, event_id=event_dict)
fig.subplots_adjust(right=0.7)

###############################################################################
# Events can also be plotted alongside the :class:`~mne.io.Raw` object they
# were extracted from:

raw.plot(events=events, start=5, duration=10, color='gray',
         event_color={1: 'r', 2: 'g', 3: 'b', 4: 'm', 5: 'y', 32: 'k'})

###############################################################################
# Events versus annotations
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Events, like :class:`~mne.Annotations`, link a particular time in the
# :class:`~mne.io.Raw` object to a particular label. However, they differ in a
# few ways:
#
# - In an event array the label is an integer; in an annotation the label is a
#   string.
#
# - In an event array the time is recorded as a sample number; in an annotation
#   the time is recorded in seconds.
#
# - Every annotation has a duration (which can be zero); events do not have a
#   duration (although it is possible to represent temporal spans as pairs of
#   pulse onset/offset events, by passing ``output='step'`` to
#   :func:`mne.find_events`).
#
# You can convert annotations to events using the
# :func:`mne.events_from_annotations` function, which acts on
# :class:`~mne.io.Raw` objects that have annotations, and returns an event
# array as well as an event dictionary that gives the mapping from annotation
# descriptions to integer Event IDs:

annot = mne.Annotations(onset=[3, 5, 7], duration=[1, 0.5, 0.25],
                        description=['AAA', 'BBB', 'CCC'])
raw.set_annotations(annot)
events_from_annot, event_dict_from_annot = mne.events_from_annotations(raw)
print(events_from_annot)
print(event_dict_from_annot)

###############################################################################
# If you want to control the specific integers that get assigned to each
# annotation description, you can create your own event dictionary and pass it
# to :func:`~mne.events_from_annotations` (the same dictionary will be returned
# by the function along with the event array):

ev_id_dict = {'AAA': 77, 'BBB': 88, 'CCC': 99}
ev_ann, ev_dict_ann = mne.events_from_annotations(raw, event_id=ev_id_dict)
print(ev_ann)
print(ev_dict_ann)

###############################################################################
# By default, one event will be created at the onset of each annotation; this
# can be modified via the ``chunk_duration`` parameter of
# :func:`~mne.events_from_annotations` to create equally spaced events within
# each annotation span. See the function documentation for further details.
