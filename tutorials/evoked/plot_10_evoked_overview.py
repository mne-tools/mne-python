"""
.. _tut-evoked-class:

The Evoked data structure: evoked/averaged data
===============================================

This tutorial covers the basics of creating and working with :term:`evoked`
data. It introduces the :class:`~mne.Evoked` data structure in detail,
including how to load, query, subselect, export, and plot data from an
:class:`~mne.Evoked` object. For info on creating an :class:`~mne.Evoked`
object from (possibly simulated) data in a :class:`NumPy array
<numpy.ndarray>`, see :ref:`tut_creating_data_structures`.

.. contents:: Page contents
   :local:
   :depth: 2

As usual we'll start by importing the modules we need:
"""

import os
import mne

###############################################################################
# Creating ``Evoked`` objects from ``Epochs``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :class:`~mne.Evoked` objects typically store an EEG or MEG signal that has
# been *averaged* over multiple :term:`epochs`, which is a common technique for
# estimating stimulus-evoked activity. The data in an :class:`~mne.Evoked`
# object are stored in an :class:`array <numpy.ndarray>` of shape
# ``(n_channels, n_times)`` (in contrast to an :class:`~mne.Epochs` object,
# which stores data of shape ``(n_epochs, n_channels, n_times)``). Thus to
# create an :class:`~mne.Evoked` object, we'll start by epoching some raw data,
# and then averaging together all the epochs from one condition:

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
events = mne.find_events(raw, stim_channel='STI 014')
# we'll skip the "face" and "buttonpress" conditions, to save memory:
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4}
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict,
                    preload=True)
evoked = epochs['auditory/left'].average()

del raw  # reduce memory usage

###############################################################################
# Basic visualization of ``Evoked`` objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can visualize the average evoked response for left-auditory stimuli using
# the :meth:`~mne.Evoked.plot` method, which yields a butterfly plot of each
# channel type:

evoked.plot()

###############################################################################
# Like the ``plot()`` methods for :meth:`Raw <mne.io.Raw.plot>` and
# :meth:`Epochs <mne.Epochs.plot>` objects,
# :meth:`evoked.plot() <mne.Evoked.plot>` has many parameters for customizing
# the plot output, such as color-coding channel traces by scalp location, or
# plotting the :term:`global field power <GFP>` alongside the channel traces.
# See :ref:`tut-visualize-evoked` for more information about visualizing
# :class:`~mne.Evoked` objects.
#
#
# Subselecting ``Evoked`` data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. sidebar:: Evokeds are not memory-mapped
#
#   :class:`~mne.Evoked` objects use a :attr:`~mne.Evoked.data` *attribute*
#   rather than a :meth:`~mne.Epochs.get_data` *method*; this reflects the fact
#   that the data in :class:`~mne.Evoked` objects are always loaded into
#   memory, never `memory-mapped`_ from their location on disk (because they
#   are typically *much* smaller than :class:`~mne.io.Raw` or
#   :class:`~mne.Epochs` objects).
#
#
# Unlike :class:`~mne.io.Raw` and :class:`~mne.Epochs` objects,
# :class:`~mne.Evoked` objects do not support selection by square-bracket
# indexing. Instead, data can be subselected by indexing the
# :attr:`~mne.Evoked.data` attribute:

print(evoked.data[:2, :3])  # first 2 channels, first 3 timepoints

###############################################################################
# To select based on time in seconds, the :meth:`~mne.Evoked.time_as_index`
# method can be useful, although beware that depending on the sampling
# frequency, the number of samples in a span of given duration may not always
# be the same (see the :ref:`time-as-index` section of the
# :ref:`tutorial about Raw data <tut-raw-class>` for details).
#
#
# Selecting, dropping, and reordering channels
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# By default, when creating :class:`~mne.Evoked` data from an
# :class:`~mne.Epochs` object, only the "data" channels will be retained:
# ``eog``, ``ecg``, ``stim``, and ``misc`` channel types will be dropped. You
# can control which channel types are retained via the ``picks`` parameter of
# :meth:`epochs.average() <mne.Epochs.average>`, by passing ``'all'`` to
# retain all channels, or by passing a list of integers, channel names, or
# channel types. See the documentation of :meth:`~mne.Epochs.average` for
# details.
#
# If you've already created the :class:`~mne.Evoked` object, you can use the
# :meth:`~mne.Evoked.pick`, :meth:`~mne.Evoked.pick_channels`,
# :meth:`~mne.Evoked.pick_types`, and :meth:`~mne.Evoked.drop_channels` methods
# to modify which channels are included in an :class:`~mne.Evoked` object.
# You can also use :meth:`~mne.Evoked.reorder_channels` for this purpose; any
# channel names not provided to :meth:`~mne.Evoked.reorder_channels` will be
# dropped. Note that *channel* selection methods modify the object in-place, so
# in interactive/exploratory sessions you may want to create a
# :meth:`~mne.Evoked.copy` first.

evoked_eeg = evoked.copy().pick_types(meg=False, eeg=True)
print(evoked_eeg.ch_names)

new_order = ['EEG 002', 'MEG 2521', 'EEG 003']
evoked_subset = evoked.copy().reorder_channels(new_order)
print(evoked_subset.ch_names)

###############################################################################
# Similarities among the core data structures
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :class:`~mne.Evoked` objects have many similarities with :class:`~mne.io.Raw`
# and :class:`~mne.Epochs` objects, including:
#
# - They can be loaded from and saved to disk in ``.fif`` format, and their
#   data can be exported to a :class:`NumPy array <numpy.ndarray>` (but through
#   the :attr:`~mne.Evoked.data` attribute, not through a ``get_data()``
#   method). :class:`Pandas DataFrame <pandas.DataFrame>` export is also
#   available through the :meth:`~mne.Evoked.to_data_frame` method.
#
# - You can change the name or type of a channel using
#   :meth:`evoked.rename_channels() <mne.Evoked.rename_channels>` or
#   :meth:`evoked.set_channel_types() <mne.Evoked.set_channel_types>`.
#   Both methods take :class:`dictionaries <dict>` where the keys are existing
#   channel names, and the values are the new name (or type) for that channel.
#   Existing channels that are not in the dictionary will be unchanged.
#
# - :term:`SSP projector <projector>` manipulation is possible through
#   :meth:`~mne.Evoked.add_proj`, :meth:`~mne.Evoked.del_proj`, and
#   :meth:`~mne.Evoked.plot_projs_topomap` methods, and the
#   :attr:`~mne.Evoked.proj` attribute. See :ref:`tut-artifact-ssp` for more
#   information on SSP.
#
# - Like :class:`~mne.io.Raw` and :class:`~mne.Epochs` objects,
#   :class:`~mne.Evoked` objects have :meth:`~mne.Evoked.copy`,
#   :meth:`~mne.Evoked.crop`, :meth:`~mne.Evoked.time_as_index`,
#   :meth:`~mne.Evoked.filter`, and :meth:`~mne.Evoked.resample` methods.
#
# - Like :class:`~mne.io.Raw` and :class:`~mne.Epochs` objects,
#   :class:`~mne.Evoked` objects have ``evoked.times``,
#   :attr:`evoked.ch_names <mne.Evoked.ch_names>`, and :class:`info <mne.Info>`
#   attributes.
#
#
# .. _tut-section-load-evk:
#
# Loading and saving ``Evoked`` data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Single :class:`~mne.Evoked` objects can be saved to disk with the
# :meth:`evoked.save() <mne.Evoked.save>` method. One difference between
# :class:`~mne.Evoked` objects and the other data structures is that multiple
# :class:`~mne.Evoked` objects can be saved into a single ``.fif`` file, using
# :func:`mne.write_evokeds`. The :ref:`example data <sample-dataset>`
# includes just such a ``.fif`` file: the data have already been epoched and
# averaged, and the file contains separate :class:`~mne.Evoked` objects for
# each experimental condition:

sample_data_evk_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis-ave.fif')
evokeds_list = mne.read_evokeds(sample_data_evk_file, verbose=False)
print(evokeds_list)
print(type(evokeds_list))

###############################################################################
# Notice that :func:`mne.read_evokeds` returned a :class:`list` of
# :class:`~mne.Evoked` objects, and each one has an ``evoked.comment``
# attribute describing the experimental condition that was averaged to
# generate the estimate:

for evok in evokeds_list:
    print(evok.comment)

###############################################################################
# If you want to load only some of the conditions present in a ``.fif`` file,
# :func:`~mne.read_evokeds` has a ``condition`` parameter, which takes either a
# string (matched against the comment attribute of the evoked objects on disk),
# or an integer selecting the :class:`~mne.Evoked` object based on the order
# it's stored in the file. Passing lists of integers or strings is also
# possible. If only one object is selected, the :class:`~mne.Evoked` object
# will be returned directly (rather than a length-one list containing it):

right_vis = mne.read_evokeds(sample_data_evk_file, condition='Right visual')
print(right_vis)
print(type(right_vis))

###############################################################################
# Above, when we created an :class:`~mne.Evoked` object by averaging epochs,
# baseline correction was applied by default when we extracted epochs from the
# `~mne.io.Raw` object (the default baseline period is ``(None, 0)``,
# which assured zero mean for times before the stimulus event). In contrast, if
# we plot the first :class:`~mne.Evoked` object in the list that was loaded
# from disk, we'll see that the data have not been baseline-corrected:

evokeds_list[0].plot(picks='eeg')

###############################################################################
# This can be remedied by either passing a ``baseline`` parameter to
# :func:`mne.read_evokeds`, or by applying baseline correction after loading,
# as shown here:

evokeds_list[0].apply_baseline((None, 0))
evokeds_list[0].plot(picks='eeg')

###############################################################################
# Notice that :meth:`~mne.Evoked.apply_baseline` operated in-place. Similarly,
# :class:`~mne.Evoked` objects may have been saved to disk with or without
# :term:`projectors <projector>` applied; you can pass ``proj=True`` to the
# :func:`~mne.read_evokeds` function, or use the :meth:`~mne.Evoked.apply_proj`
# method after loading.
#
#
# Combining ``Evoked`` objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# One way to pool data across multiple conditions when estimating evoked
# responses is to do so *prior to averaging* (recall that MNE-Python can select
# based on partial matching of ``/``-separated epoch labels; see
# :ref:`tut-section-subselect-epochs` for more info):

left_right_aud = epochs['auditory'].average()
print(left_right_aud)

###############################################################################
# This approach will weight each epoch equally and create a single
# :class:`~mne.Evoked` object. Notice that the printed representation includes
# ``(average, N=145)``, indicating that the :class:`~mne.Evoked` object was
# created by averaging across 145 epochs. In this case, the event types were
# fairly close in number:

left_aud = epochs['auditory/left'].average()
right_aud = epochs['auditory/right'].average()
print([evok.nave for evok in (left_aud, right_aud)])

###############################################################################
# However, this may not always be the case; if for statistical reasons it is
# important to average *the same number* of epochs from different conditions,
# you can use :meth:`~mne.Epochs.equalize_event_counts` prior to averaging.
#
# Another approach to pooling across conditions is to create separate
# :class:`~mne.Evoked` objects for each condition, and combine them afterward.
# This can be accomplished by the function :func:`mne.combine_evoked`, which
# computes a weighted sum of the :class:`~mne.Evoked` objects given to it. The
# weights can be manually specified as a list or array of float values, or can
# be specified using the keyword ``'equal'`` (weight each `~mne.Evoked` object
# by :math:`\frac{1}{N}`, where :math:`N` is the number of `~mne.Evoked`
# objects given) or the keyword ``'nave'`` (weight each `~mne.Evoked` object
# proportional to the number of epochs averaged together to create it):

left_right_aud = mne.combine_evoked([left_aud, right_aud], weights='nave')
assert left_right_aud.nave == left_aud.nave + right_aud.nave

###############################################################################
# Note that the ``nave`` attribute of the resulting `~mne.Evoked` object will
# reflect the *effective* number of averages, and depends on both the ``nave``
# attributes of the contributing `~mne.Evoked` objects and the weights at
# which they are combined. Keeping track of effective ``nave`` is important for
# inverse imaging, because ``nave`` is used to scale the noise covariance
# estimate (which in turn affects the magnitude of estimated source activity).
# See :ref:`minimum_norm_estimates` for more information (especially the
# :ref:`whitening_and_scaling` section). Note that `mne.grand_average` does
# *not* adjust ``nave`` to reflect effective number of averaged epochs; rather
# it simply sets ``nave`` to the number of *evokeds* that were averaged
# together. For this reason, it is best to use `mne.combine_evoked` rather than
# `mne.grand_average` if you intend to perform inverse imaging on the resulting
# :class:`~mne.Evoked` object.
#
#
# Other uses of ``Evoked`` objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Although the most common use of :class:`~mne.Evoked` objects is to store
# *averages* of epoched data, there are a couple other uses worth noting here.
# First, the method :meth:`epochs.standard_error() <mne.Epochs.standard_error>`
# will create an :class:`~mne.Evoked` object (just like
# :meth:`epochs.average() <mne.Epochs.average>` does), but the data in the
# :class:`~mne.Evoked` object will be the standard error across epochs instead
# of the average. To indicate this difference, :class:`~mne.Evoked` objects
# have a :attr:`~mne.Evoked.kind` attribute that takes values ``'average'`` or
# ``'standard error'`` as appropriate.
#
# Another use of :class:`~mne.Evoked` objects is to represent *a single trial
# or epoch* of data, usually when looping through epochs. This can be easily
# accomplished with the :meth:`epochs.iter_evoked() <mne.Epochs.iter_evoked>`
# method, and can be useful for applications where you want to do something
# that is only possible for :class:`~mne.Evoked` objects. For example, here
# we use the :meth:`~mne.Evoked.get_peak` method (which isn't available for
# :class:`~mne.Epochs` objects) to get the peak response in each trial:

for ix, trial in enumerate(epochs[:3].iter_evoked()):
    channel, latency, value = trial.get_peak(ch_type='eeg',
                                             return_amplitude=True)
    latency = int(round(latency * 1e3))  # convert to milliseconds
    value = int(round(value * 1e6))      # convert to µV
    print('Trial {}: peak of {} µV at {} ms in channel {}'
          .format(ix, value, latency, channel))

###############################################################################
# .. REFERENCES
#
# .. _`memory-mapped`: https://en.wikipedia.org/wiki/Memory-mapped_file
