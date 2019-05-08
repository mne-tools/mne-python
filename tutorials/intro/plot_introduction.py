# -*- coding: utf-8 -*-
"""

Introduction to  MEG and EEG data processing with MNE-Python
============================================================

This tutorial covers the basic EEG/MEG pipeline for event-related analysis:
loading data, epoching, averaging, plotting, and estimating cortical activity
from sensor data. It introduces the core MNE-Python data structures
:class:`~mne.io.Raw`, :class:`~mne.Epochs`, :class:`~mne.Evoked`, and
:class:`~mne.SourceEstimate`, and covers a lot of ground fairly quickly (at the
expense of depth). Subsequent tutorials address each of these topics in greater
detail. We begin by importing the necessary Python modules:
"""

import os
import numpy as np
import mne

###############################################################################
# Loading data
# ^^^^^^^^^^^^
#
# MNE-Python data structures are based around the FIF file format from
# Neuromag, but there are reader functions for :ref:`a wide variety of other
# data formats <data-formats>`. MNE-Python also has interfaces to a
# variety of :doc:`publicly available datasets <../../manual/datasets_index>`,
# which MNE-Python can download and manage for you.
#
# We'll start this tutorial by loading one of the example datasets (called
# ":ref:`sample-dataset`"), which contains EEG and MEG data from one subject
# performing an audiovisual experiment, along with structural MRI scans for
# that subject. The :func:`mne.datasets.sample.data_path` function will
# automatically download the dataset if it isn't found in one of the expected
# locations, then return the directory path to the dataset (see the
# documentation of :func:`~mne.datasets.sample.data_path` for a list of places
# it checks before downloading).

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=False)

###############################################################################
# .. note::
#
#     :func:`~mne.io.read_raw_fif` takes a ``preload`` parameter, which
#     determines whether the data will be copied into RAM or not. Some
#     operations (such as filtering) require that the data be preloaded, but it
#     is possible use ``preload=False`` and then copy raw data into memory
#     later using the :meth:`~mne.io.Raw.load_data` method if needed.
#
# By default, :func:`~mne.io.read_raw_fif` displays some information about the
# file it's loading; for example, here it tells us that there are three
# "projection items" in the file along with the recorded data; those are
# :term:`SSP projectors <projector>` for environmental noise, and are discussed
# in a later tutorial. In addition to the information displayed during loading,
# you can get a glimpse of the basic details of a :class:`~mne.io.Raw` object
# by printing it:
#
# .. TODO edit prev. paragraph when projectors tutorial is added: ...those are
#     discussed in the tutorial :ref:`projectors-tutorial`. (or whatever link)

print(raw)

###############################################################################
# :class:`~mne.io.Raw` objects also have several built-in plotting methods; in
# interactive Python sessions the basic :meth:`~mne.io.Raw.plot` method is
# interactive and allows scrolling, scaling, bad channel marking, annotation,
# projector toggling, etc.

raw.plot(duration=5, n_channels=30)

###############################################################################
# Preprocessing
# ^^^^^^^^^^^^^
#
# MNE-Python supports a variety of preprocessing approaches and techniques
# (maxwell filtering, signal-space projection, independent components analysis,
# filtering, downsampling, etc), see the full list of functions in the
# :mod:`mne.preprocessing` and :mod:`mne.filter` submodules. Here we'll quickly
# clean up our data by adding SSP projectors for eye movements and heartbeats;
# we'll also plot the projectors to show the field characteristics that we're
# projecting out:

# eye movement/blink projectors
kwargs = dict(n_grad=1, n_mag=1, n_eeg=1, average=True, no_proj=True)
eog_projs, eog_events = mne.preprocessing.compute_proj_eog(raw, **kwargs)

# heartbeat projectors
kwargs.update(n_eeg=0, reject=None)
ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, **kwargs)

raw.add_proj(eog_projs + ecg_projs)
mne.viz.plot_projs_topomap(eog_projs + ecg_projs, info=raw.info)

###############################################################################
# Detecting experimental events
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The sample dataset includes several "STIM" channels that recorded electrical
# signals sent from the stimulus delivery computer (as brief DC shifts /
# squarewave pulses). These pulses (often called "triggers") are used in this
# dataset to mark experimental events: stimulus onset, stimulus type, and
# participant response (button press). The individual STIM channels are
# combined in a binary weighted sum onto a single channel, such that voltage
# levels on that channel can be unambiguously decoded as a particular event
# type. On older Neuromag systems (such as that used to record the sample data)
# this summation channel was called ``STI 014``, so we can pass that channel
# name to the :func:`mne.find_events` function to recover the timing and
# identity of the stimulus events.

events = mne.find_events(raw, stim_channel='STI 014')
print(events[:5])  # show the first 5

###############################################################################
# The resulting events array is an ordinary 3-column :class:`NumPy array
# <numpy.ndarray>`, with sample number in the first column and integer event ID
# in the last column; the middle column is usually ignored. Rather than keeping
# track of integer event IDs, we can provide an *event dictionary* that maps
# the integer IDs to experimental conditions or events. In this dataset, the
# mapping looks like this:
#
# +----------+----------------------------------------------------------+
# | Event ID | Condition                                                |
# +==========+==========================================================+
# | 1        | auditory stimulus (tone) to the left ear                 |
# +----------+----------------------------------------------------------+
# | 2        | auditory stimulus (tone) to the right ear                |
# +----------+----------------------------------------------------------+
# | 3        | visual stimulus (checkerboard) to the left visual field  |
# +----------+----------------------------------------------------------+
# | 4        | visual stimulus (checkerboard) to the right visual field |
# +----------+----------------------------------------------------------+
# | 5        | smiley face (catch trial)                                |
# +----------+----------------------------------------------------------+
# | 32       | subject button press                                     |
# +----------+----------------------------------------------------------+

event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'smiley': 5, 'buttonpress': 32}

###############################################################################
# Event dictionaries like this one are used when extracting epochs from
# continuous data; the ``/`` character in the dictionary keys allows pooling
# across conditions by requesting partial condition descriptors (i.e.,
# requesting ``'auditory'`` will select all epochs with Event IDs 1 and 2;
# requesting ``'left'`` will select all epochs with Event IDs 1 and 3). An
# example of this is shown in the next section. There is also a convenient
# :func:`~mne.viz.plot_events` function for visualizing the distribution of
# events across the duration of the recording (to make sure event detection
# worked as expected):

fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'])
fig.subplots_adjust(right=0.7)  # make room for the legend

###############################################################################
# Epoching continuous data
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# The :class:`~mne.io.Raw` object and the events array are the bare minimum
# needed to create an :class:`~mne.Epochs` object, which we create with the
# :class:`mne.Epochs` class constructor. Here we'll also specify some data
# quality constraints: we'll reject any epoch where peak-to-peak signal
# amplitude is beyond reasonable limits for that channel type. This is done
# with a *rejection dictionary*:

reject_criteria = dict(mag=4e-12,     # 4000 fT
                       grad=4e-10,    # 4000 fT/cm
                       eeg=150e-6,    # 150 μV
                       eog=250e-6)    # 250 μV

###############################################################################
# We'll also pass the event dictionary as the ``event_id`` parameter (so we can
# work with easy-to-pool event labels instead of the integer event IDs), and
# specify ``tmin`` and ``tmax`` (the time relative to each event at which to
# start and end each epoch). Finally, since we didn't preload the
# :class:`~mne.io.Raw` data, we'll tell the :class:`~mne.Epochs` constructor to
# load the epoched data into memory:

epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.5,
                    reject=reject_criteria, preload=True)

###############################################################################
# Next we'll pool across left/right stimulus presentations so we can compare
# auditory versus visual responses. To avoid biasing our signals to the
# left or right, we'll use :meth:`~mne.Epochs.equalize_event_counts` first to
# randomly sample epochs from each condition to match the number of epochs
# present in the condition with the fewest good epochs.

conds_we_care_about = ['auditory/left', 'auditory/right',
                       'visual/left', 'visual/right']
epochs.equalize_event_counts(conds_we_care_about)  # this operates in-place
aud_epochs = epochs['auditory']
vis_epochs = epochs['visual']
del raw, epochs  # free up memory

###############################################################################
# Like :class:`~mne.io.Raw` objects, :class:`~mne.Epochs` objects also have a
# number of built-in plotting methods. One is :meth:`~mne.Epochs.plot_image`,
# which shows each epoch as one row of an image map, with color representing
# signal magnitude; the average evoked response and the sensor location are
# shown below the image:

aud_epochs.plot_image(picks=['MEG 1332', 'EEG 021'])

###############################################################################
# Estimating evoked responses
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now that we have our conditions in ``aud_epochs`` and ``vis_epochs``, we can
# get an estimate of evoked responses to auditory versus visual stimuli by
# averaging together the epochs in each condition. This is as simple as calling
# the :meth:`~mne.Epochs.average` method on the :class:`~mne.Epochs` object,
# and then using a function from the :mod:`mne.viz` module to compare the
# global field power for each sensor type of the two :class:`~mne.Evoked`
# objects:

aud_evoked = aud_epochs.average()
vis_evoked = vis_epochs.average()

mne.viz.plot_compare_evokeds(dict(auditory=aud_evoked, visual=vis_evoked),
                             show_legend='upper left',
                             show_sensors='upper right')

###############################################################################
# We can also get a more detailed view of each :class:`~mne.Evoked` object
# using other plotting methods such as :meth:`~mne.Evoked.plot_joint` or
# :meth:`~mne.Evoked.plot_topomap`. Here we'll examine just the EEG channels,
# and see the classic auditory evoked N100-P200 pattern over dorso-frontal
# electrodes, then plot scalp topographies at some additional arbitrary times:

aud_evoked.plot_joint(picks='eeg')
aud_evoked.plot_topomap(times=[0., 0.08, 0.1, 0.12, 0.2], ch_type='eeg')

##############################################################################
# Evoked objects can also be combined to show contrasts between conditions,
# using the :func:`mne.combine_evoked` function. A simple difference can be
# generated by negating one of the :class:`~mne.Evoked` objects passed into the
# function. We'll then plot the difference wave at each sensor using
# :meth:`~mne.Evoked.plot_topo`:

evoked_diff = mne.combine_evoked([aud_evoked, -vis_evoked], weights='equal')
evoked_diff.pick_types('mag').plot_topo(color='r', legend=False)

##############################################################################
# Time-frequency analysis
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# The :mod:`mne.time_frequency` submodule provides implementations of several
# algorithms to compute time-frequency representations, power spectral density,
# and cross-spectral density. Here, for example, we'll compute for the
# auditory epochs the induced power at different frequencies and times, using
# Morlet wavelets:

frequencies = np.arange(7, 30, 3)
power = mne.time_frequency.tfr_morlet(aud_epochs, n_cycles=2, return_itc=False,
                                      freqs=frequencies, decim=3)
power.plot(['MEG 1332'])

##############################################################################
# Inverse modeling
# ^^^^^^^^^^^^^^^^
#
# Finally, we can estimate the cortical origins of the evoked activity by
# projecting the sensor data into this subject's :term:`source space`.
# MNE-Python supports lots of ways of doing this (minimum-norm estimation,
# dipole fitting, beamformers, etc); here we'll use minimum-norm estimation
# (MNE) to generate a continuous cortical activation map. To do this we'll
# need the inverse operator for this subject (the sample data includes one
# that's been pre-computed), and we'll need to specify a regularization
# parameter:

inverse_operator_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                     'sample_audvis-meg-oct-6-meg-inv.fif')
inv_operator = mne.minimum_norm.read_inverse_operator(inverse_operator_file)
# set the regularization parameter (λ²)
signal_to_noise_ratio = 3.
regularization_param = 1. / signal_to_noise_ratio ** 2
# generate the STC
source_time_course = mne.minimum_norm.apply_inverse(vis_evoked, inv_operator,
                                                    regularization_param,
                                                    method='MNE')
# plot
source_time_course.plot(initial_time=0.1, hemi='split', views=['lat', 'med'])

##############################################################################
# The rest of the tutorials have *much more detail* on each of these topics (as
# well as many other capabilities of MNE-Python not mentioned here:
# connectivity analysis, encoding/decoding models, lots more visualization
# options, etc). Read on to learn more!

# sphinx_gallery_thumbnail_number = 9
