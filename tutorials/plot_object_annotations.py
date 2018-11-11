"""
The **events** and :class:`Annotations <mne.Annotations>` data structures
=========================================================================

Events and annotations are quite similar. This tutorial highlights their
differences and similitudes and tries to shade some light to which one is
preferred to use in different situations when using MNE.
Here follows both terms definition from the :ref:`glossary`.

    events
        Events correspond to specific time points in raw data; e.g., triggers,
        experimental condition events, etc. MNE represents events with integers
        that are stored in numpy arrays of shape (n_events, 3). Such arrays are
        classically obtained from a trigger channel, also referred to as stim
        channel.

    annotations
        One annotation is defined by an onset, a duration and a string
        description. It can contain information about the experiments, but
        also details on signals marked by a human: bad data segments,
        sleep scores, sleep events (spindles, K-complex) etc.

They both can be seen as triplets where the first element answers to **when**
something happens and the last element refers to **what** is it. The
main differnce is that events the when is samples with respect to the first
sample and the what is an integer id; while in annotations the when is in
seconds with respect to an origin and the what is an arbitrary string.
The second element of the triplets have no direct relation between the two
structures. For the events case, the second element corresponds to id of the
previous active event. Whereas, the second element of the
:class:`mne.Annotations` is a float indicating its duration in seconds.

See :ref:`sphx_glr_auto_examples_io_plot_read_events.py`
for a complete example in how to read, select and visualize **events**;
and :ref:`sphx_glr_auto_tutorials_plot_artifacts_correction_rejection.py` to
know how :class:`mne.Annotations` are used to mark bad segments of data.

An example of events and annotations
------------------------------------

The following example shows the recorded events in `sample_audvis_raw.fif` and
marks bad segments due to eye blinks.
"""

import os.path as op
import numpy as np

import mne

# load the data
data_path = mne.datasets.sample.data_path()
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(fname)

# plot the events
events = mne.find_events(raw)

# Specify event_id dictionary based on the experiment
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
            'Visual/Left': 3, 'Visual/Right': 4,
            'smiley': 5, 'button': 32}
color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c', 5: 'black', 32: 'blue'}

mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp, color=color,
                    event_id=event_id)

# create some annotations
annotated_blink_raw = raw.copy()
eog_events = mne.preprocessing.find_eog_events(raw)
n_blinks = len(eog_events)
# Center to cover the whole blink with full duration of 0.5s:
onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
duration = np.repeat(0.5, n_blinks)
annot = mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
                        orig_time=raw.info['meas_date'])
annotated_blink_raw.set_annotations(annot)

annotated_blink_raw.plot()  # plot the annotated raw


###############################################################################
# Working with Annotations
# ------------------------
#
# An important element of the :class:`mne.Annotations` is ``orig_time`` which
# is the time reference for the ``onset``. It is key to understand that when
# calling `raw.set_annotation`, the given annotations is copied and transformed
# so that `raw.annotations.orig_time` matches the recording time of the raw
# object. (check :class:`mne.Annotations` documentation notes to see the
# expected behavior depending of `meas_date` and `orig_time`. Notice that
# `meas_date` is the :class:`Info <mne.Info>` attribute of the recording time.
# Find more in :ref:`sphx_glr_auto_tutorials_plot_info.py`)
#
# We'll now manipulate some simulated annotations objects.
#
# First let's create an annotation object without orig_time. It this case
# one assumes that the orig_time is the time of the first sample of data.
annot_none = mne.Annotations(onset=[0, 2, 9], duration=[0.5, 4, 0],
                             description=['foo', 'bar', 'foo'],
                             orig_time=None)
print(annot_none)


###############################################################################
# Now let's create annotation object with orig_time
annot_orig = mne.Annotations(onset=[22, 24, 31], duration=[0.5, 4, 0],
                             description=['foo', 'bar', 'foo'],
                             orig_time=1038942091.6760709)
print(annot_orig)

###############################################################################
# create two cropped copies of raw with the two previous annotations
raw_a = raw.copy().crop(tmax=12).set_annotations(annot_none)
raw_b = raw.copy().crop(tmax=12).set_annotations(annot_orig)

# plot the raw objects
raw_a.plot()
raw_b.plot()

###############################################################################
# show the annotations in the raw objects
print(raw_a.annotations)
print(raw_b.annotations)

###############################################################################
# show that the onsets are the same
print(raw_a.annotations.onset)
print(raw_b.annotations.onset)

###############################################################################
# It is possible to concatenate two annotations with the + like for lists.

annot = mne.Annotations(onset=[10], duration=[0.5],
                        description=['foobar'],
                        orig_time=1038942091.6760709)
annot = annot_orig + annot  # concatenation
print(annot)
