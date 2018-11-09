"""
.. _tut_events_and_annotation_objects:

The **events** and :class:`Annotations <mne.Annotations>` data structures
=========================================================================

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

# XXX I should put somewhere which file formats use stim channel and which ones prefer annotation
# see agramfort comment:
#     bst is coming from a ctf system
#     so .ds files
#     that also have stim channels usually
#     annotations are from EEG files / vendors

# - [ ] 1. `find_events`, with event masking
# - [ ] 2. `events` column values
# - [ ] 3. `raw.first_samp`, what it is, how it changes with `.crop`, how to get
#           event times relative to the start of the instance
# - [ ] 4. `Annotations` and `orig_time`

###############################################################################
# links 
# :ref:`sphx_glr_auto_tutorials_plot_epoching_and_averaging.py`
# :ref:`sphx_glr_download_auto_examples_io_plot_read_events.py`
# https://mne-tools.github.io/stable/auto_tutorials/plot_artifacts_correction_rejection.html?highlight=marking%20bad%20segments
# :class:`mne.Annotations` and :ref:`marking_bad_segments`. To see all the

import os.path as op
import numpy as np

import mne

# Define some helper functions & constants
def _get_blink_annotations(raw):
    eog_events = mne.preprocessing.find_eog_events(raw)
    n_blinks = len(eog_events)
    # Center to cover the whole blink with full duration of 0.5s:
    onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
    duration = np.repeat(0.5, n_blinks)
    return mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
                           orig_time=raw.info['meas_date'])

# Specify event_id dictionary based on the experiment
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
            'Visual/Left': 3, 'Visual/Right': 4,
            'smiley': 5, 'button': 32}

color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c', 5: 'black', 32: 'blue'}

# load the data
data_path = mne.datasets.sample.data_path()
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(fname)

# plot the events
events = mne.find_events(raw)
mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp, color=color,
                    event_id=event_id)

# create some annotations
annotated_blink_raw = raw.copy()
annot = _get_blink_annotations(annotated_blink_raw)
annotated_blink_raw.set_annotations(annot)

print(annotated_blink_raw.annotations)  # check the annotations
annotated_blink_raw.plot()  # plot the annotated raw


###############################################################################
# Annotations
#
# An important element of the :class:`mne.Annotations` is ``orig_time`` which
# is the time reference for the ``onset``. It is key to understand that when
# calling `raw.set_annotation`, the given annotations is copied and transformed
# so that `raw.annotations.orig_time` matches meas_date. (check
# :class:`mne.Annotations` documentation notes to see the expected behavior
# depending of `meas_date` and `orig_time`)

# empty_annot = mne.Annotations(onset=list(), duration=list(),
#                               description=list(), orig_time=None)


# XXXX This should not be done like that, I should be able to get something
# printable using MNE.
def _print_meas_date(stamp):
    if stamp is None:
        print('None')
    else:
        from datetime import datetime
        stamp = mne.annotations._handle_meas_date(stamp)
        print(datetime.utcfromtimestamp(stamp))

annot_none = mne.Annotations(onset=[0, 2, 9], duration=[0.5, 4, 0],
                             description=['AA', 'BB', 'CC'],
                             orig_time=None)
print('annotation without orig_time')
_print_meas_date(annot_none.orig_time)


annot_orig = mne.Annotations(onset=[22, 24, 31], duration=[0.5, 4, 0],
                             description=['AA', 'BB', 'CC'],
                             orig_time=1038942091.6760709)
print('annotation with orig_time')
_print_meas_date(annot_orig.orig_time)

print('raw.info[\'meas_date\']')
_print_meas_date(raw.info['meas_date'])


raw_a = raw.copy().crop(tmax=12).set_annotations(annot_none)
raw_b = raw.copy().crop(tmax=12).set_annotations(annot_orig)

# Plot both raw files side to side to see they are the same
#
# fig, axs = plt.subplots(1, 2, figsize=(15, 5))
# raw_a.plot(axes=axs[0])
# axs[0].set(title="using None")
# raw_b.plot(axes=axs[0])
# axs[0].set(title="using orig_time")
# plt.tight_layout()
# plt.show()
#
raw_a.plot()
raw_b.plot()

# show the new origin
print('raw_a.annotation.orig_time')
_print_meas_date(raw_a.annotations.orig_time)
print('raw_b.annotation.orig_time')
_print_meas_date(raw_b.annotations.orig_time)

print(raw_a.annotations.onset == raw_b.annotations.onset)











 


