
Annotations
###########

.. contents::
   :local:
   :depth: 2

MNE provides an :func:`mne.Annotations` class that can be used to mark segments
of raw data and to reject epochs that overlap with bad segments of data.
The annotations are automatically synchronized with raw data as long as the
timestamps of raw data and annotations are in sync.

When working with neuromag data, the ``first_samp`` offset of raw acquisition
is also taken into account the same way as with event lists.
