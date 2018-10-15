"""
.. _tut_events_and_annotation_objects:

The **events** and :class:`Annotations <mne.Annotations>` data structures
=========================================================================

XXX bla

1. `find_events`, with event masking
2. `events` column values
3. `raw.first_samp`, what it is, how it changes with `.crop`, how to get event times relative to the start of the instance
4. `Annotations` and `orig_time`

Events and annotations are quite similar. This tutorial highlights their
differences and similitudes and tries to shade some light to which one is
preferred to use in different situations when using MNE.
Here follows both terms definition from the glossary.
We refer the reader to :ref:`sphx_glr_auto_examples_io_plot_read_events.py`
for a complete example in how to read, select and visualize **events**;
and ref:`marking_bad_segments` to know how :class:`mne.Annotations` are used to
mark bad segments of data.

    events
        Events correspond to specific time points in raw data; e.g.,
        triggers, experimental condition events, etc. MNE represents events with
        integers that are stored in numpy arrays of shape (n_events, 3). Such arrays
        are classically obtained from a trigger channel, also referred to as
        stim channel.

    annotations
        One annotation is defined by an onset, a duration and a string
        description. It can contain information about the experiments, but
        also details on signals marked by a human: bad data segments,
        sleep scores, sleep events (spindles, K-complex) etc.


What are events, annotations and how to load / interpret them
-------------------------------------------------------------
"""
# :ref:`sphx_glr_auto_tutorials_plot_epoching_and_averaging.py`

# XXX I should put somewhere which file formats use stim channel and which ones prefer annotation
# see agramfort comment:
#     bst is coming from a ctf system
#     so .ds files
#     that also have stim channels usually
#     annotations are from EEG files / vendors

import os.path as op
import numpy as np

import mne

# load the data
data_path = mne.datasets.sample.data_path()
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(fname)

# Specify event_id dictionary based on the experiment
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
            'Visual/Left': 3, 'Visual/Right': 4,
            'smiley': 5, 'button': 32}


# plot the events
color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c', 5: 'black', 32: 'blue'}
events = mne.find_events(raw)
print(events[:5])
mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp, color=color,
                    event_id=event_id)

# create some annotations
eog_events = mne.preprocessing.find_eog_events(raw)
n_blinks = len(eog_events)
# Center to cover the whole blink with full duration of 0.5s:
onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
duration = np.repeat(0.5, n_blinks)
annot = mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
                        orig_time=raw.info['meas_date'])
raw.set_annotations(annot)


print(raw.annotations)  # to get information about what annotations we have
raw.plot(events=eog_events)  # To see the annotated segments.

###############################################################################
# links 
# :ref:`sphx_glr_auto_tutorials_plot_epoching_and_averaging.py`
# :ref:`sphx_glr_download_auto_examples_io_plot_read_events.py`
# https://mne-tools.github.io/stable/auto_tutorials/plot_artifacts_correction_rejection.html?highlight=marking%20bad%20segments
# :class:`mne.Annotations` and :ref:`marking_bad_segments`. To see all the

###############################################################################
# Other

# ###############################################################################
# # To illustrate the events object, lets see its first values.
# #
# # an event is a triplet of (trigger, XXX
# print('Found %s events, first five:' % len(events))
# print(events[:5])

# ###############################################################################
# # The main usage until MNE-v.0.17 of Annotations was to mark bad segments in the data.
# # .. _marking_bad_segments:
# # https://mne-tools.github.io/stable/auto_tutorials/plot_visualize_raw.html#drawing-annotations
# #
# # lets load a version of the same example but this time use a modified version for illustration
# # purposes that lacks stim channel but has annotations encoding the events
# # (plus, some fake bad segments).
# fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw_with_annotations_no_events.fif')
# raw_with_annotations = mne.io.read_raw_fif(fname)

# # illustrate that `find_events` returns an empty list since there is no stim channel
# assert mne.find_evets(raw_with_annotations) == []
# raw_with_annotations.annotations

# # plot something XXXX

# # Specify colors and an event_id dictionary as before for the legend.
# event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
#             'Visual/Left': 3, 'Visual/Right': 4,
#             'smiley': 5, 'button': 32}
# color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c', 5: 'black', 32: 'blue'}

# events = mne.events_from_annotations(raw_with_annotations)
# mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp, color=color,
#                     event_id=event_id)

# ###############################################################################
# # annotations can be loaded and treated on their own.
# # fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw_with_no_annotations_no_events.fif')
# # annotations_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_annotations.fif')

# # raw_no_annotations_no_events = mne.io.read_raw_fif(fname)
# # assert mne.find_evets(raw_with_no_annotations_no_events) == []
# # assert not len(raw_with_no_annotations_no_events.annotations)

# # inspect the annotations
# # annot = mne.read_annotations(annotations_fname)
# # print(annot)
# # print(annot.orig_time)

# # observe that annotations on its own, cannot be converted into event and they need a raw like object
# # xxx = raw_no_annotations_no_events.copy()
# # xxx.set_annotations(annot)







 


