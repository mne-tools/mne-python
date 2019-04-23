# -*- coding: utf-8 -*-
"""
.. _annotations-tutorial:

Annotating continuous data
==========================

.. include:: ../../tutorial_links.inc

This tutorial describes adding annotations to a Raw object, and how annotations
are used in later stages of data processing.
"""

###############################################################################
# As usual we'll start by importing the modules we need, loading some
# example data, and cropping it to save on memory:

import os
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
raw.crop(tmax=60).load_data()

###############################################################################
# :class:`~mne.Annotations` in MNE-Python are a way of storing short strings of
# information about temporal spans of a :class:`~mne.io.Raw` object. Below the
# surface, each annotation in an :class:`~mne.Annotations` object is just three
# pieces of information: an ``onset`` time (in seconds), a ``duration`` (also
# in seconds), and a ``description`` (a text string).
#
#
# Creating annotations programmatically
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# If you know in advance what spans of the :class:`~mne.io.Raw` object you want
# to annotate, :class:`~mne.Annotations` can be created programmatically, and
# you can even pass lists or arrays to the :class:`~mne.Annotations`
# constructor to annotate multiple spans at once:

my_annot = mne.Annotations(onset=[3, 5, 7], duration=[1, 0.5, 0.25],
                           description=['AAA', 'BBB', 'CCC'])
print(my_annot)

###############################################################################
# When you print an :class:`~mne.Annotations` object, you'll notice it has a
# property called ``orig_time``; this gives the time relative to which the
# annotation ``onset`` values are determined. In the example above
# ``orig_time`` is ``None`` (meaning we didn't set it).
#
# Once created, the annotations can be added to a :class:`~mne.io.Raw` object
# with the :meth:`~mne.io.Raw.set_annotations` method, and then accessed
# through the :attr:`~mne.io.Raw.annotations` attribute:

raw.set_annotations(my_annot)
print(raw.annotations)

###############################################################################
# Notice that now the ``orig_time`` has been updated to match the measurement
# date of the :class:`~mne.io.Raw` object (``raw.info['meas_date']``):
# ``2002-12-03 19:01:10.720100``. Below, you can also see that the annotation
# onsets have been adjusted to take ``raw.first_samp`` into account (see
# :ref:`time-as-index` for more info on ``raw.first_samp``):

print(my_annot.onset)
print(raw.annotations.onset)
print(my_annot.onset + raw.first_samp / raw.info['sfreq'])

###############################################################################
# If that's not what you want, you can set ``orig_time`` before you add the
# annotations object to :class:`~mne.io.Raw`, and the onsets won't get
# adjusted. When you create the annotations object you can set ``orig_time``
# using an `ISO 8601`_ formatted string; here we'll set it at 100 seconds later
# than ``raw.info['meas_date']``:

later_annot = mne.Annotations(onset=[3, 5, 7], duration=[1, 0.5, 0.25],
                              description=['DDD', 'EEE', 'FFF'],
                              orig_time='2002-12-03 19:02:50.720100')
raw2 = raw.copy().set_annotations(later_annot)
print(later_annot.onset)
print(raw2.annotations.onset)

###############################################################################
# .. note::
#
#     If your annotations fall outside the range of data times in the
#     :class:`~mne.io.Raw` object, the annotations outside the data range will
#     not be added to ``raw.annotations``, and a warning will be issued.
#
# Now that your annotations have been added to a :class:`~mne.io.Raw` object,
# you can see them when you visualize the :class:`~mne.io.Raw` object:

raw.plot(start=2, duration=6)

###############################################################################
# The three annotations appear as differently colored rectangles because they
# have different ``description`` values (which are printed along the top
# edge of the plot area). Notice also that colored spans appear in the small
# scroll bar at the bottom of the plot window, making it easy to quickly view
# where in a :class:`~mne.io.Raw` object the annotations are so you can easily
# browse through the data to find and examine them.
#
#
# Annotating :class:`~mne.io.Raw` objects interactively
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Annotations can also be added to a :class:`~mne.io.Raw` object interactively
# by clicking-and-dragging the mouse in the plot window. To do this, you must
# first enter "annotation mode" by pressing :kbd:`a` while the plot window is
# focused; this will bring up the annotation controls window:
#
# .. image:: ../../_static/annotation-controls.png
#    :alt: screenshot of MNE-Python annotation controls window
#
# The colored rings are clickable, and determine which existing label will be
# created by the next click-and-drag operation in the main plot window. New
# annotation descriptions can be added by typing the new description,
# clicking the :guilabel:`Add label` button; the new description will be added
# to the list of descriptions and automatically selected.
#
# During interactive annotation it is also possible to adjust the start and end
# times of existing annotations, by clicking-and-dragging on the left or right
# edges of the highlighting rectangle corresponding to that annotation.
#
# .. warning::
#
#     Calling :meth:`~mne.io.Raw.set_annotations` **replaces** any annotations
#     currently stored in the :class:`~mne.io.Raw` object, so be careful when
#     working with annotations that were created interactively (you could lose
#     a lot of work if you accidentally overwrite your interactive
#     annotations). A good safeguard is to run
#     ``interactive_annot = raw.annotations`` after you finish an interactive
#     annotation session, so that the annotations are stored in a separate
#     variable outside the :class:`~mne.io.Raw` object.
#
#
# How annotations affect preprocessing and analysis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You may have noticed that the description for new labels in the annotation
# controls window defaults to ``BAD_``. The reason for this is that annotation
# is often used to mark bad temporal spans of data (such as movement artifacts
# or environmental interference that cannot be removed in other ways such as
# :ref:`projection <ssp-tutorial>` or filtering). Several MNE-Python operations
# are "annotation aware" and will avoid using data that is annotated with a
# description that begins with "bad" or "BAD"; such operations typically have a
# boolean ``reject_by_annotation`` parameter. Examples of such operations are
# independent components analysis (:class:`mne.preprocessing.ICA`), functions
# for finding heartbeat and blink artifacts
# (:func:`~mne.preprocessing.find_ecg_events`,
# :func:`~mne.preprocessing.find_eog_events`), and creation of epoched data
# from continuous data (:class:`mne.Epochs`).
#
#
# Operations on :class:`~mne.Annotations` objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :class:`~mne.Annotations` objects can be combined by simply adding them with
# the ``+`` operator, as long as they share the same ``orig_time``:

new_annot = mne.Annotations(onset=3.75, duration=0.75, description='AAA')
raw.set_annotations(my_annot + new_annot)
raw.plot(start=2, duration=6)

###############################################################################
# Notice that it is possible to create overlapping annotations, even when they
# share the same description. This is *not* possible when annotating
# interactively; click-and-dragging to create a new annotation that overlaps
# with an existing annotation with the same description will cause the old and
# new annotations to be merged.
#
# Individual annotations can be accessed by indexing an
# :class:`~mne.Annotations` object, and subsets of the annotations can be
# achieved by either slicing or indexing with a list, tuple, or array of
# indices:

print(raw.annotations[0])       # just the first annotation
print(raw.annotations[:2])      # the first two annotations
print(raw.annotations[(3, 2)])  # the fourth and third annotations

###############################################################################
# You can also iterate over the annotations within an :class:`~mne.Annotations`
# object:

for ann in raw.annotations:
    descr = ann['description']
    start = ann['onset']
    end = ann['onset'] + ann['duration']
    print(f"'{descr}' goes from {start} to {end}")

###############################################################################
# Note that iterating, indexing and slicing :class:`~mne.Annotations` all
# return a copy, so changes to an indexed, sliced, or iterated element will not
# modify the original :class:`~mne.Annotations` object.

# later_annot WILL be changed, because we're modifying the first element of
# later_annot.onset directly:
later_annot.onset[0] = 99

# later_annot WILL NOT be changed, because later_annot[0] returns a copy
# before the 'onset' field is changed:
later_annot[0]['onset'] = 77

print(later_annot[0]['onset'])

###############################################################################
# :class:`~mne.Annotations` objects have a :meth:`~mne.Annotations.save` method
# which can write ``.fif``, ``.csv``, and ``.txt`` formats (the format to write
# is inferred from the file extension in the filename you provide). There is a
# corresponding :func:`~mne.read_annotations` function to load them from disk:

raw.annotations.save('saved-annotations.csv')
annot_from_file = mne.read_annotations('saved-annotations.csv')
print(annot_from_file)
