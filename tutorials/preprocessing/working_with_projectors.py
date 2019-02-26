# -*- coding: utf-8 -*-
"""
.. _projectors-tutorial:

Working with projectors
=======================

This tutorial provides background information on :term:`projectors <projector>`
and describes some common use cases for projectors: setting an EEG reference,
heartbeats, blinks, environmental noise, etc. As usual we'll start by importing
the modules we need and loading some sample data:
"""

import os
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)

###############################################################################
# Background: projectors and projections
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# TODO: basic concepts
#
# TODO: orthogonal vs oblique
#
# TODO: a figure illustrating the concepts?
#
# Loading and saving projectors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# TODO: some narrative text
#
# TODO: discussion of PCA projectors that are part of the raw file
# (from empty room recording)

ecg_proj_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                             'sample_audvis_ecg-proj.fif')
ecg_projs = mne.read_proj(ecg_proj_file)
channel_layouts = [mne.find_layout(raw.info, ch_type=channel_type)
                   for channel_type in ('grad', 'mag', 'eeg')]
mne.viz.plot_projs_topomap(ecg_projs, layout=channel_layouts)

###############################################################################
# Getting and setting projectors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# TODO: some narrative text

raw2 = raw.copy()
raw2.add_proj(ecg_projs)
raw2.plot_projs_topomap()

###############################################################################
# You can access individual projectors by indexing into the ``projs`` field of
# ``raw.info`` (it's just a list of :class:`~mne.Projection` objects), and can
# plot individual projectors using the :meth:`~mne.Projection.plot_topomap`
# method on the :class:`~mne.Projection` object:

first_projector = raw.info['projs'][0]
first_projector.plot_topomap()

###############################################################################
# When to apply projectors
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# TODO: have they been applied? (:attr:`mne.io.Raw.proj`)
#
# TODO: when to apply projectors
#
# TODO: when projectors are applied automatically
#
# TODO: turning them on/off during `raw.plot()`

###############################################################################
# Setting the EEG reference
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
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

# use average of mastoid channels as reference
# raw.set_eeg_reference(ref_channels=['M1', 'M2'])

# use a single channel reference (left earlobe)
# raw.set_eeg_reference(ref_channels=['A1'])

# use the average of all channels as reference
raw.set_eeg_reference(ref_channels='average')

###############################################################################
# .. note::
#
#     MNE-Python will automatically apply an average reference if EEG channels
#     are present and no reference strategy is specified. Thus if you are
#     loading partially-preprocessed data that has already had a reference
#     applied, you should set the reference to an empty list
#     (``raw.set_eeg_reference(ref_channels=[])``) to prevent MNE-Python from
#     subtracting a second average reference signal from your data.
#
# The EEG reference can also be added as a projector rather than directly
# subtracting the reference signal from the other channels.
#
# TODO: the (dis)advantage of this is...
#
# TODO: the things to watch out for are...
#
# TODO: section on blinks.. others?
