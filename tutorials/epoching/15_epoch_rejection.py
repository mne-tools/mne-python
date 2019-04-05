# -*- coding: utf-8 -*-
"""
.. _epoch-rejection-tutorial:

Automated epoch rejection
=========================

.. include:: ../../tutorial_links.inc

This tutorial covers the mechanisms for automated rejection of epochs based on
annotations or signal properties.
"""

###############################################################################
# As usual we'll start by importing the modules we need, and loading some
# example data:

import os
import mne

# load raw
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)
# get events
events = mne.find_events(raw, stim_channel='STI 014')
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'button': 32}

###############################################################################
# Rejecting epochs based on signal amplitude
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# So far when we've created epoched data from continous data, we've generated
# one epoch for every event, regardless of what the signal looks like in that
# epoch and regardless of any annotations in our :class:`~mne.io.Raw` object.
# However, it is common to reject epochs on the basis of overly large signal
# fluctuations (on the assumption that they represent environmental or system
# noise, not genuine neural signal), and also common to reject epochs due to
# the absence of signal fluctuations (on the assumption that a flat or nearly
# flat signal indicates a defective sensor).
#
# The :class:`~mne.Epochs` constructor has parameters (``reject`` and ``flat``)
# for rejecting epochs based on overly large or overly small signals; both take
# python dictionaries that can provide different thresholds for different
# channel types. The reject values should be given in the units used in
# MNE-Python's internal representation of signals (i.e., teslas for
# magnetometers, teslas/meter for gradiometers, and volts for EEG & EOG; see
# :ref:`units` for other sensor types).

reject_criteria = dict(mag=4e-12,     # 4000 fT
                       grad=4e-10,    # 4000 fT/cm
                       eeg=50e-6,     # 50 μV
                       eog=250e-6)    # 250 μV

# create epochs
epochs = mne.Epochs(raw, events, event_id=event_dict, preload=True,
                    reject=reject_criteria)

###############################################################################
# As you can see, the :class:`~mne.Epochs` constructor prints a message each
# time an epoch is rejected (or "dropped") based on your rejection criteria.
# In this case we've ended up dropping all 320 events, leaving us with zero
# epochs! Clearly our rejection criteria are too stringent.

print(epochs)

###############################################################################
# To correct this, you should make the rejection criteria less stringent, but
# with that much output it can be hard to tell which channels or channel types
# are most problematic. To make this process easier, the method
# :meth:`~mne.Epochs.plot_drop_log` will show, for each channel, the percentage
# of epochs that were rejected because that channel exceeded the ``reject``
# criterion for its channel type.

epochs.plot_drop_log()

###############################################################################
# Sometimes, only 1 or 2 channels may be responsible for the majority of epoch
# rejections; in such cases it may be best to simply remove that channel by
# adding it to the list of bad channels
# (``raw.info['bads'].append('channel_to_remove')``) and then re-run the
# epoching of the continuous data.  In other cases (like this one) there are
# many channels that are exceeding threshold on 50% or more of the epochs,
# suggesting that at least one of our rejection criteria is too stringent.
# Since all of the worst channels appear to be EEG channels, let's adjust the
# EEG rejection criterion and re-run the epoching step:

reject_criteria.update(eeg=150e-6)
epochs = mne.Epochs(raw, events, event_id=event_dict, preload=True,
                    reject=reject_criteria)
epochs.plot_drop_log()

###############################################################################
# With this new threshold we're losing only 48 epochs (or 15%), which is much
# closer to an acceptable level of data loss. At this point there is one
# channel (``'EEG 007'``) that contributes to many more dropped epochs than the
# other channels listed; you might consider marking that channel as "bad" (and
# possibly interpolating it) to further reduce the number of dropped epochs,
# but bear in mind the trade-off between fewer dropped epochs (more data) and
# fewer channels (lower data rank).
#
# It is a good idea to make sure that the dropped epochs are not affecting any
# one experimental condition too severely; printing the resulting epochs object
# will give epoch counts for each event label, allowing you to easily see this
# information:

print(epochs)

###############################################################################
# It is also possible to perform these threshold-based rejections after the
# continuous data has already been epoched, using the
# :meth:`~mne.Epochs.drop_bad` method. This method takes the same parameters as
# the :class:`~mne.Epochs` constructor (``reject`` and ``flat``), and operates
# in-place:

epochs = mne.Epochs(raw, events, event_id=event_dict, preload=True)
print('=' * 70)  # to clearly separate the output of the two commands
epochs.drop_bad(reject=reject_criteria)


###############################################################################
# Notice above that rejection was performed when the epochs were created, but
# no epochs were rejected because no rejection criteria were supplied (hence
# the statement ``0 bad epochs dropped`` above the line of `=` signs). Only
# when :meth:`~mne.Epochs.drop_bad` is invoked does the rejection occur.
#
# :meth:`~mne.Epochs.drop_bad` can also be useful for cases where rejection
# thresholds were provided when the epochs were created, but rejection was not
# actually carried out because the epoched data were not loaded into memory
# at the time (i.e., when the ``preload`` parameter was ``False``).

# default is preload=False
epochs = mne.Epochs(raw, events, event_id=event_dict, reject=reject_criteria)
print('=' * 70)
epochs.drop_bad()

###############################################################################
# Rejecting epochs based on annotations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As discussed in the tutorial :ref:`annotations-tutorial`, many operations in
# MNE-Python have a ``reject_by_annotations`` parameter that will omit spans of
# the continuous data that are annotated with a label that begins with
# ``'bad'`` or ``'BAD'``. The :class:`~mne.Epochs` constructor is one of those
# operations, providing a way to omit spans of data regardless of whether they
# exceed a particular rejection threshold. For example, you might choose to
# omit spans of the data where alpha waves are prominent in the EEG channels,
# suggesting that the subject is slipping into sleep, or spans involving large
# movement artifacts. Here we'll simulate those annotations:

my_annot = mne.Annotations(onset=[15, 30], duration=[4, 1.5],
                           description=['bad_alpha', 'bad_movement_artifact'])
raw.set_annotations(my_annot)
epochs = mne.Epochs(raw, events, event_id=event_dict, preload=True,
                    reject_by_annotation=True)

###############################################################################
# The :meth:`~mne.Epochs.plot_drop_log` method also works for epochs rejected
# based on annotations:

epochs.plot_drop_log()

###############################################################################
# Finally, it is possible to employ both rejection methods simultaneously, by
# providing both a threshold dictionary and the boolean
# ``reject_by_annotation`` parameter:

epochs = mne.Epochs(raw, events, event_id=event_dict, preload=True,
                    reject_by_annotation=True, reject=reject_criteria)
epochs.plot_drop_log()

###############################################################################
# Manually dropping epochs
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# Another way of omitting epochs from further analysis is by manually dropping
# them after visual inspection. As mentioned in
# :ref:`plotting-epochs-tutorial`, the :meth:`~mne.Epochs.plot` method yields
# an interactive plot where you can click on individual epochs to mark them as
# "bad"; epochs marked in that way will be dropped automatically when the plot
# window is closed. Alternatively, if you know the indices of the epochs you
# want to drop, you can do so with the :meth:`~mne.Epochs.drop` method:

epochs.drop([17, 19, 23])
