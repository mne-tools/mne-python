# -*- coding: utf-8 -*-
"""
.. _setting-eeg-reference-tutorial:

Setting the EEG reference
=========================

.. include:: ../../tutorial_links.inc

This tutorial describes how to set an EEG reference in MNE-Python.
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
# Typically, one of the first steps in processing EEG recordings is to subtract
# a *reference signal* from each channel. Conceptually, the reference signal
# represents environmental or equipment noise that affects all sensors
# approximately equally, and by subtracting it from the signal at each
# electrode, what is left will be a less noisy representation of brain activity
# than the original raw signal.
#
# Sometimes the subtracted reference signal is the signal from a physical
# electrode (common reference electrode placements are the earlobe or the
# mastoid processes) or the average from a pair of such electrodes. Other
# times, the subtracted reference signal is the average of signals at all
# electrodes. MNE-Python supports all of these possibilities through the
# :meth:`~mne.io.Raw.set_eeg_reference` method:
#
# .. code-block:: python3
#
#     # use average of mastoid channels as reference
#     raw.set_eeg_reference(ref_channels=['M1', 'M2'])
#
#     # use a single channel reference (left earlobe)
#     raw.set_eeg_reference(ref_channels=['A1'])

# use the average of all channels as reference
raw.set_eeg_reference(ref_channels='average')

###############################################################################
# When setting ``ref_channels='average'``, MNE-Python will automatically
# exclude any channels listed in ``raw.info['bads']`` from contributing to the
# average reference.
#
# If using an average reference, it is possible to create the reference as a
# :term:`projector` rather than subtracting the reference from the data
# immediately. This has a couple of advantages:
#
# 1. it is possible to turn projectors on or off when plotting, so it is easy
#    to visualize the effect that the reference has on the data
#
# 2. if there are other unapplied projectors affecting the EEG channels (such
#    as SSP projectors for removing heartbeat or blink artifacts), EEG
#    referencing cannot be performed until those projectors are either applied
#    or removed; adding the EEG reference as a projector is not subject to that
#    constraint.
#
# .. note::
#
#     MNE-Python will automatically apply an average reference if EEG channels
#     are present and no reference strategy is specified. Thus if you are
#     loading partially-preprocessed data that has already had a reference
#     applied, you should set the reference to an empty list
#     (``raw.set_eeg_reference(ref_channels=[])``) to prevent MNE-Python from
#     subtracting a second average reference signal from your data.
#
#
# EEG reference for source modeling
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# If you plan to perform source modeling (either with EEG or combined EEG/MEG
# data), it is **strongly recommended** to use the average reference approach.
# The reason is that using a specific reference sensor (or even an average of a
# few sensors) spreads the forward model error for the reference sensor(s) into
# all sensors, effectively amplifying the importance of the reference sensor(s)
# when computing source estimates. In contrast, using the average of all EEG
# channels as reference spreads the forward modeling error evenly across
# channels, so no one channel is weighted more strongly during source
# estimation. See also this `FieldTrip FAQ on average referencing`_ for more
# information.
