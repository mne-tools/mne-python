.. _annotations:

Annotations
###########

.. contents::
   :local:
   :depth: 2

MNE provides an :class:`mne.Annotations` class that can be used to mark segments
of raw data and to reject epochs that overlap with bad segments of data.
The annotations are automatically synchronized with raw data as long as the
timestamps of raw data and annotations are in sync.

The instances of annotations are created by providing a list of onsets and
offsets with descriptions for each segment. The onsets and offsets are marked
as seconds. ``onset`` refers to time from start of the data. ``offset`` is the
duration of the annotation. The instance of :class:`mne.Annotations` can be
added as an attribute of :class:`mne.io.Raw`.

    >>> eog_events = mne.preprocessing.find_eog_events(raw)
    >>> onset = raw.index_as_time(eog[:, 0]) - 0.25  # Center to cover the whole blink.
    >>> offset = np.repeat(0.5, len(onset))
    >>> annotations = mne.Annotations(onset, np.repeat(0.5, len(onset)), 'blink')
    >>> raw.annotations = annotations
    >>> raw.plot()  # To see the annotated segments.

As the data is epoched, all the epochs overlapping with segments whose
description starts with 'bad' are rejected by default. To turn rejection off,
use keyword argument ``reject_by_annotation=False`` when constructing
class:`mne.Epochs`. When working with neuromag data, the ``first_samp`` offset
of raw acquisition is also taken into account the same way as with event lists.
For more see class:`mne.Epochs` and :class:`mne.Annotations`.


