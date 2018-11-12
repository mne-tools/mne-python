"""
The **events** and :class:`Annotations <mne.Annotations>` data structures
=========================================================================

Events and :class:`Annotations <mne.Annotations>` are quite similar.
This tutorial highlights their differences and similarities, and tries to shed
some light on which one is preferred to use in different situations when using
MNE.

Here are the definitions from the :ref:`glossary`.

    events
        Events correspond to specific time points in raw data; e.g., triggers,
        experimental condition events, etc. MNE represents events with integers
        that are stored in numpy arrays of shape (n_events, 3). Such arrays are
        classically obtained from a trigger channel, also referred to as stim
        channel.

    annotations
        An annotation is defined by an onset, a duration, and a string
        description. It can contain information about the experiments, but
        also details on signals marked by a human: bad data segments,
        sleep scores, sleep events (spindles, K-complex) etc.

Both events and :class:`Annotations <mne.Annotations>` can be seen as triplets
where the first element answers to **when** something happens and the last
element refers to **what** it is.
The main difference is that events represent the onset in samples taking into
account the first sample value
(:attr:`raw.first_samp <mne.io.Raw.first_samp>`), and the description is
an integer value.
In contrast, :class:`Annotations <mne.Annotations>` represents the
``onset`` in seconds (relative to the reference ``orig_time``),
and the ``description`` is an arbitrary string.
There is no correspondence between the second element of events and
:class:`Annotations <mne.Annotations>`.
For events, the second element corresponds to the previous value on the
stimulus channel from which events are extracted. In practice, the second
element is therefore in most cases zero.
The second element of :class:`Annotations <mne.Annotations>` is a float
indicating its duration in seconds.

See :ref:`sphx_glr_auto_examples_io_plot_read_events.py`
for a complete example of how to read, select, and visualize **events**;
and :ref:`sphx_glr_auto_tutorials_plot_artifacts_correction_rejection.py` to
learn how :class:`Annotations <mne.Annotations>` are used to mark bad segments
of data.

An example of events and annotations
------------------------------------

The following example shows the recorded events in `sample_audvis_raw.fif` and
marks bad segments due to eye blinks.
"""

import os.path as op
import numpy as np

import mne

# Load the data
data_path = mne.datasets.sample.data_path()
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(fname)

# Plot the events
events = mne.find_events(raw)

# Specify event_id dictionary based on the experiment
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
            'Visual/Left': 3, 'Visual/Right': 4,
            'smiley': 5, 'button': 32}
color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c', 5: 'black', 32: 'blue'}

mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp, color=color,
                    event_id=event_id)

# Create some annotations specifying onset, duration and description
annotated_blink_raw = raw.copy()
eog_events = mne.preprocessing.find_eog_events(raw)
n_blinks = len(eog_events)
# Center to cover the whole blink with full duration of 0.5s:
onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
duration = np.repeat(0.5, n_blinks)
description = ['bad blink'] * n_blinks
annot = mne.Annotations(onset, duration, description,
                        orig_time=raw.info['meas_date'])
annotated_blink_raw.set_annotations(annot)

annotated_blink_raw.plot()  # plot the annotated raw


###############################################################################
# Working with Annotations
# ------------------------
#
# An important element of :class:`Annotations <mne.Annotations>` is
# ``orig_time`` which is the time reference for the ``onset``.
# It is key to understand that when calling
# :func:`raw.set_annotations <mne.io.Raw.set_annotations>`, the given
# annotations are copied and transformed so that
# :class:`raw.annotations.orig_time <mne.Annotations>`
# matches the recording time of the raw object.
# Refer to the documentation of :class:`Annotations <mne.Annotations>` to see
# the expected behavior depending on ``meas_date`` and ``orig_time``.
# Where ``meas_date`` is the recording time stored in
# :class:`Info <mne.Info>`.
# You can find more information about :class:`Info <mne.Info>` in
# :ref:`sphx_glr_auto_tutorials_plot_info.py`.
#
# We'll now manipulate some simulated annotations.
# The first annotations has ``orig_time`` set to ``None`` while the
# second is set to a chosen POSIX timestamp for illustration purposes.

###############################################################################

# Create an annotation object without orig_time
annot_none = mne.Annotations(onset=[0, 2, 9], duration=[0.5, 4, 0],
                             description=['foo', 'bar', 'foo'],
                             orig_time=None)
print(annot_none)

# Create an annotation object with orig_time
orig_time = '2002-12-03 19:01:31.676071'
annot_orig = mne.Annotations(onset=[22, 24, 31], duration=[0.5, 4, 0],
                             description=['foo', 'bar', 'foo'],
                             orig_time=orig_time)
print(annot_orig)

###############################################################################
# Now we create two raw objects, set the annotations and plot them to compare
# them.

# Create two cropped copies of raw with the two previous annotations
raw_a = raw.copy().crop(tmax=12).set_annotations(annot_none)
raw_b = raw.copy().crop(tmax=12).set_annotations(annot_orig)

# Plot the raw objects
raw_a.plot()
raw_b.plot()

# Show the annotations in the raw objects
print(raw_a.annotations)
print(raw_b.annotations)

# Show that the onsets are the same
print(raw_a.annotations.onset)
print(raw_b.annotations.onset)

###############################################################################
#
# Notice that for the case where ``orig_time`` is ``None``,
# one assumes that the orig_time is the time of the first sample of data.

raw_delta = (1 / raw.info['sfreq'])
print('raw.first_sample is {}'.format(raw.first_samp * raw_delta))
print('annot_none.onset[0] is {}'.format(annot_none.onset[0]))
print('raw_a.annotations.onset[0] is {}'.format(raw_a.annotations.onset[0]))

###############################################################################
#
# It is possible to concatenate two annotations with the + operator like for
# lists if both share the same ``orig_time``

annot = mne.Annotations(onset=[10], duration=[0.5],
                        description=['foobar'],
                        orig_time=orig_time)
annot = annot_orig + annot  # concatenation
print(annot)
