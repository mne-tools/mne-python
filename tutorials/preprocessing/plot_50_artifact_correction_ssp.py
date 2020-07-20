# -*- coding: utf-8 -*-
"""
.. _tut-artifact-ssp:

Repairing artifacts with SSP
============================

This tutorial covers the basics of signal-space projection (SSP) and shows
how SSP can be used for artifact repair; extended examples illustrate use
of SSP for environmental noise reduction, and for repair of ocular and
heartbeat artifacts.

.. contents:: Page contents
   :local:
   :depth: 2

We begin as always by importing the necessary Python modules. To save ourselves
from repeatedly typing ``mne.preprocessing`` we'll directly import a couple
functions from that submodule:
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)

###############################################################################
# .. note::
#     Before applying SSP (or any artifact repair strategy), be sure to observe
#     the artifacts in your data to make sure you choose the right repair tool.
#     Sometimes the right tool is no tool at all — if the artifacts are small
#     enough you may not even need to repair them to get good analysis results.
#     See :ref:`tut-artifact-overview` for guidance on detecting and
#     visualizing various types of artifact.
#
#
# What is SSP?
# ^^^^^^^^^^^^
#
# Signal-space projection (SSP) :footcite:`UusitaloIlmoniemi1997` is a
# technique for removing noise from EEG
# and MEG signals by :term:`projecting <projector>` the signal onto a
# lower-dimensional subspace. The subspace is chosen by calculating the average
# pattern across sensors when the noise is present, treating that pattern as
# a "direction" in the sensor space, and constructing the subspace to be
# orthogonal to the noise direction (for a detailed walk-through of projection
# see :ref:`tut-projectors-background`).
#
# The most common use of SSP is to remove noise from MEG signals when the noise
# comes from environmental sources (sources outside the subject's body and the
# MEG system, such as the electromagnetic fields from nearby electrical
# equipment) and when that noise is *stationary* (doesn't change much over the
# duration of the recording). However, SSP can also be used to remove
# biological artifacts such as heartbeat (ECG) and eye movement (EOG)
# artifacts. Examples of each of these are given below.
#
#
# Example: Environmental noise reduction from empty-room recordings
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The :ref:`example data <sample-dataset>` was recorded on a Neuromag system,
# which stores SSP projectors for environmental noise removal in the system
# configuration (so that reasonably clean raw data can be viewed in real-time
# during acquisition). For this reason, all the :class:`~mne.io.Raw` data in
# the example dataset already includes SSP projectors, which are noted in the
# output when loading the data:

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)

###############################################################################
# The :ref:`example data <sample-dataset>` also includes an "empty room"
# recording taken the same day as the recording of the subject. This will
# provide a more accurate estimate of environmental noise than the projectors
# stored with the system (which are typically generated during annual
# maintenance and tuning). Since we have this subject-specific empty-room
# recording, we'll create our own projectors from it and discard the
# system-provided SSP projectors (saving them first, for later comparison with
# the custom ones):

system_projs = raw.info['projs']
raw.del_proj()
empty_room_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                               'ernoise_raw.fif')
empty_room_raw = mne.io.read_raw_fif(empty_room_file)

###############################################################################
# Notice that the empty room recording itself has the system-provided SSP
# projectors in it — we'll remove those from the empty room file too.

empty_room_raw.del_proj()

###############################################################################
# Visualizing the empty-room noise
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's take a look at the spectrum of the empty room noise. We can view an
# individual spectrum for each sensor, or an average (with confidence band)
# across sensors:

for average in (False, True):
    empty_room_raw.plot_psd(average=average, dB=False, xscale='log')

###############################################################################
# Creating the empty-room projectors
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We create the SSP vectors using :func:`~mne.compute_proj_raw`, and control
# the number of projectors with parameters ``n_grad`` and ``n_mag``. Once
# created, the field pattern of the projectors can be easily visualized with
# :func:`~mne.viz.plot_projs_topomap`. We include the parameter
# ``vlim='joint'`` so that the colormap is computed jointly for all projectors
# of a given channel type; this makes it easier to compare their relative
# smoothness. Note that for the function to know the types of channels in a
# projector, you must also provide the corresponding :class:`~mne.Info` object:

# sphinx_gallery_thumbnail_number = 3
empty_room_projs = mne.compute_proj_raw(empty_room_raw, n_grad=3, n_mag=3)
mne.viz.plot_projs_topomap(empty_room_projs, colorbar=True, vlim='joint',
                           info=empty_room_raw.info)

###############################################################################
# Notice that the gradiometer-based projectors seem to reflect problems with
# individual sensor units rather than a global noise source (indeed, planar
# gradiometers are much less sensitive to distant sources). This is the reason
# that the system-provided noise projectors are computed only for
# magnetometers. Comparing the system-provided projectors to the
# subject-specific ones, we can see they are reasonably similar (though in a
# different order) and the left-right component seems to have changed
# polarity.

fig, axs = plt.subplots(2, 3)
for idx, _projs in enumerate([system_projs, empty_room_projs[3:]]):
    mne.viz.plot_projs_topomap(_projs, axes=axs[idx], colorbar=True,
                               vlim='joint', info=empty_room_raw.info)

###############################################################################
# Visualizing how projectors affect the signal
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We could visualize the different effects these have on the data by applying
# each set of projectors to different copies of the :class:`~mne.io.Raw` object
# using :meth:`~mne.io.Raw.apply_proj`. However, the :meth:`~mne.io.Raw.plot`
# method has a ``proj`` parameter that allows us to *temporarily* apply
# projectors while plotting, so we can use this to visualize the difference
# without needing to copy the data. Because the projectors are so similar, we
# need to zoom in pretty close on the data to see any differences:

mags = mne.pick_types(raw.info, meg='mag')
for title, projs in [('system', system_projs),
                     ('subject-specific', empty_room_projs[3:])]:
    raw.add_proj(projs, remove_existing=True)
    fig = raw.plot(proj=True, order=mags, duration=1, n_channels=2)
    fig.subplots_adjust(top=0.9)  # make room for title
    fig.suptitle('{} projectors'.format(title), size='xx-large', weight='bold')

###############################################################################
# The effect is sometimes easier to see on averaged data. Here we use an
# interactive feature of :func:`mne.Evoked.plot_topomap` to turn projectors on
# and off to see the effect on the data. Of course, the interactivity won't
# work on the tutorial website, but you can download the tutorial and try it
# locally:

events = mne.find_events(raw, stim_channel='STI 014')
event_id = {'auditory/left': 1}

# NOTE: appropriate rejection criteria are highly data-dependent
reject = dict(mag=4000e-15,     # 4000 fT
              grad=4000e-13,    # 4000 fT/cm
              eeg=150e-6,       # 150 µV
              eog=250e-6)       # 250 µV

# time range where we expect to see the auditory N100: 50-150 ms post-stimulus
times = np.linspace(0.05, 0.15, 5)

epochs = mne.Epochs(raw, events, event_id, proj='delayed', reject=reject)
fig = epochs.average().plot_topomap(times, proj='interactive')

###############################################################################
# Plotting the ERP/F using ``evoked.plot()`` or ``evoked.plot_joint()`` with
# and without projectors applied can also be informative, as can plotting with
# ``proj='reconstruct'``, which can reduce the signal bias introduced by
# projections (see :ref:`tut-artifact-ssp-reconstruction` below).
#
# Example: EOG and ECG artifact repair
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Visualizing the artifacts
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As mentioned in :ref:`the ICA tutorial <tut-artifact-ica>`, an important
# first step is visualizing the artifacts you want to repair. Here they are in
# the raw data:

# pick some channels that clearly show heartbeats and blinks
regexp = r'(MEG [12][45][123]1|EEG 00.)'
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
raw.plot(order=artifact_picks, n_channels=len(artifact_picks))

###############################################################################
# Repairing ECG artifacts with SSP
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# MNE-Python provides several functions for detecting and removing heartbeats
# from EEG and MEG data. As we saw in :ref:`tut-artifact-overview`,
# :func:`~mne.preprocessing.create_ecg_epochs` can be used to both detect and
# extract heartbeat artifacts into an :class:`~mne.Epochs` object, which can
# be used to visualize how the heartbeat artifacts manifest across the sensors:

ecg_evoked = create_ecg_epochs(raw).average()
ecg_evoked.plot_joint()

###############################################################################
# Looks like the EEG channels are pretty spread out; let's baseline-correct and
# plot again:

ecg_evoked.apply_baseline((None, None))
ecg_evoked.plot_joint()

###############################################################################
# To compute SSP projectors for the heartbeat artifact, you can use
# :func:`~mne.preprocessing.compute_proj_ecg`, which takes a
# :class:`~mne.io.Raw` object as input and returns the requested number of
# projectors for magnetometers, gradiometers, and EEG channels (default is two
# projectors for each channel type).
# :func:`~mne.preprocessing.compute_proj_ecg` also returns an :term:`events`
# array containing the sample numbers corresponding to the onset of each
# detected heartbeat.

projs, events = compute_proj_ecg(raw, n_grad=1, n_mag=1, n_eeg=1, reject=None)

###############################################################################
# The first line of output tells us that
# :func:`~mne.preprocessing.compute_proj_ecg` found three existing projectors
# already in the :class:`~mne.io.Raw` object, and will include those in the
# list of projectors that it returns (appending the new ECG projectors to the
# end of the list). If you don't want that, you can change that behavior with
# the boolean ``no_proj`` parameter. Since we've already run the computation,
# we can just as easily separate out the ECG projectors by indexing the list of
# projectors:

ecg_projs = projs[3:]
print(ecg_projs)

###############################################################################
# Just like with the empty-room projectors, we can visualize the scalp
# distribution:

mne.viz.plot_projs_topomap(ecg_projs, info=raw.info)

###############################################################################
# Since no dedicated ECG sensor channel was detected in the
# :class:`~mne.io.Raw` object, by default
# :func:`~mne.preprocessing.compute_proj_ecg` used the magnetometers to
# estimate the ECG signal (as stated on the third line of output, above). You
# can also supply the ``ch_name`` parameter to restrict which channel to use
# for ECG artifact detection; this is most useful when you had an ECG sensor
# but it is not labeled as such in the :class:`~mne.io.Raw` file.
#
# The next few lines of the output describe the filter used to isolate ECG
# events. The default settings are usually adequate, but the filter can be
# customized via the parameters ``ecg_l_freq``, ``ecg_h_freq``, and
# ``filter_length`` (see the documentation of
# :func:`~mne.preprocessing.compute_proj_ecg` for details).
#
# .. TODO what are the cases where you might need to customize the ECG filter?
#    infants? Heart murmur?
#
# Once the ECG events have been identified,
# :func:`~mne.preprocessing.compute_proj_ecg` will also filter the data
# channels before extracting epochs around each heartbeat, using the parameter
# values given in ``l_freq``, ``h_freq``, ``filter_length``, ``filter_method``,
# and ``iir_params``. Here again, the default parameter values are usually
# adequate.
#
# .. TODO should advice for filtering here be the same as advice for filtering
#    raw data generally? (e.g., keep high-pass very low to avoid peak shifts?
#    what if your raw data is already filtered?)
#
# By default, the filtered epochs will be averaged together
# before the projection is computed; this can be controlled with the boolean
# ``average`` parameter. In general this improves the signal-to-noise (where
# "signal" here is our artifact!) ratio because the artifact temporal waveform
# is fairly similar across epochs and well time locked to the detected events.
#
# To get a sense of how the heartbeat affects the signal at each sensor, you
# can plot the data with and without the ECG projectors:


raw.del_proj()
for title, proj in [('Without', empty_room_projs), ('With', ecg_projs)]:
    raw.add_proj(proj, remove_existing=False)
    fig = raw.plot(order=artifact_picks, n_channels=len(artifact_picks))
    fig.subplots_adjust(top=0.9)  # make room for title
    fig.suptitle('{} ECG projectors'.format(title), size='xx-large',
                 weight='bold')

###############################################################################
# Finally, note that above we passed ``reject=None`` to the
# :func:`~mne.preprocessing.compute_proj_ecg` function, meaning that all
# detected ECG epochs would be used when computing the projectors (regardless
# of signal quality in the data sensors during those epochs). The default
# behavior is to reject epochs based on signal amplitude: epochs with
# peak-to-peak amplitudes exceeding 50 µV in EEG channels, 250 µV in EOG
# channels, 2000 fT/cm in gradiometer channels, or 3000 fT in magnetometer
# channels. You can change these thresholds by passing a dictionary with keys
# ``eeg``, ``eog``, ``mag``, and ``grad`` (though be sure to pass the threshold
# values in volts, teslas, or teslas/meter). Generally, it is a good idea to
# reject such epochs when computing the ECG projectors (since presumably the
# high-amplitude fluctuations in the channels are noise, not reflective of
# brain activity); passing ``reject=None`` above was done simply to avoid the
# dozens of extra lines of output (enumerating which sensor(s) were responsible
# for each rejected epoch) from cluttering up the tutorial.
#
# .. note::
#
#     :func:`~mne.preprocessing.compute_proj_ecg` has a similar parameter
#     ``flat`` for specifying the *minimum* acceptable peak-to-peak amplitude
#     for each channel type.
#
# While :func:`~mne.preprocessing.compute_proj_ecg` conveniently combines
# several operations into a single function, MNE-Python also provides functions
# for performing each part of the process. Specifically:
#
# - :func:`mne.preprocessing.find_ecg_events` for detecting heartbeats in a
#   :class:`~mne.io.Raw` object and returning a corresponding :term:`events`
#   array
#
# - :func:`mne.preprocessing.create_ecg_epochs` for detecting heartbeats in a
#   :class:`~mne.io.Raw` object and returning an :class:`~mne.Epochs` object
#
# - :func:`mne.compute_proj_epochs` for creating projector(s) from any
#   :class:`~mne.Epochs` object
#
# See the documentation of each function for further details.
#
#
# Repairing EOG artifacts with SSP
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Once again let's visualize our artifact before trying to repair it. We've
# seen above the large deflections in frontal EEG channels in the raw data;
# here is how the ocular artifacts manifests across all the sensors:

eog_evoked = create_eog_epochs(raw).average()
eog_evoked.apply_baseline((None, None))
eog_evoked.plot_joint()

###############################################################################
# Just like we did with the heartbeat artifact, we can compute SSP projectors
# for the ocular artifact using :func:`~mne.preprocessing.compute_proj_eog`,
# which again takes a :class:`~mne.io.Raw` object as input and returns the
# requested number of projectors for magnetometers, gradiometers, and EEG
# channels (default is two projectors for each channel type). This time, we'll
# pass ``no_proj`` parameter (so we get back only the new EOG projectors, not
# also the existing projectors in the :class:`~mne.io.Raw` object), and we'll
# ignore the events array by assigning it to ``_`` (the conventional way of
# handling unwanted return elements in Python).

eog_projs, _ = compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=1, reject=None,
                                no_proj=True)

###############################################################################
# Just like with the empty-room and ECG projectors, we can visualize the scalp
# distribution:

mne.viz.plot_projs_topomap(eog_projs, info=raw.info)

###############################################################################
# Now we repeat the plot from above (with empty room and ECG projectors) and
# compare it to a plot with empty room, ECG, and EOG projectors, to see how
# well the ocular artifacts have been repaired:

for title in ('Without', 'With'):
    if title == 'With':
        raw.add_proj(eog_projs)
    fig = raw.plot(order=artifact_picks, n_channels=len(artifact_picks))
    fig.subplots_adjust(top=0.9)  # make room for title
    fig.suptitle('{} EOG projectors'.format(title), size='xx-large',
                 weight='bold')

###############################################################################
# Notice that the small peaks in the first to magnetometer channels (``MEG
# 1411`` and ``MEG 1421``) that occur at the same time as the large EEG
# deflections have also been removed.
#
#
# Choosing the number of projectors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In the examples above, we used 3 projectors (all magnetometer) to capture
# empty room noise, and saw how projectors computed for the gradiometers failed
# to capture *global* patterns (and thus we discarded the gradiometer
# projectors). Then we computed 3 projectors (1 for each channel type) to
# capture the heartbeat artifact, and 3 more to capture the ocular artifact.
# How did we choose these numbers? The short answer is "based on experience" —
# knowing how heartbeat artifacts typically manifest across the sensor array
# allows us to recognize them when we see them, and recognize when additional
# projectors are capturing something else other than a heartbeat artifact (and
# thus may be removing brain signal and should be discarded).
#
# .. _tut-artifact-ssp-reconstruction:
#
# Visualizing SSP sensor-space bias via signal reconstruction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# .. sidebar:: SSP reconstruction
#
#     Internally, the reconstruction is performed by effectively using a
#     minimum-norm source localization to a spherical source space with the
#     projections accounted for, and then projecting the source-space data
#     back out to sensor space.
#
# Because SSP performs an orthogonal projection, any spatial component in the
# data that is not perfectly orthogonal to the SSP spatial direction(s) will
# have its overall amplitude reduced by the projection operation. In other
# words, SSP typically introduces some amount of amplitude reduction bias in
# the sensor space data.
#
# When performing source localization of M/EEG data, these projections are
# properly taken into account by being applied not just to the M/EEG data
# but also to the forward solution, and hence SSP should not bias the estimated
# source amplitudes. However, for sensor space analyses, it can be useful to
# visualize the extent to which SSP projection has biased the data. This can be
# explored by using ``proj='reconstruct'`` in evoked plotting functions, for
# example via :meth:`evoked.plot() <mne.Evoked.plot>`:

evoked = epochs.average()
# Apply the average ref first:
# It's how we typically view EEG data, and here we're really just interested
# in the effect of the EOG+ECG SSPs
evoked.del_proj().set_eeg_reference(projection=True).apply_proj()
evoked.add_proj(ecg_projs).add_proj(eog_projs)
fig, axes = plt.subplots(3, 3, figsize=(8, 6))
for ii in range(3):
    axes[ii, 0].get_shared_y_axes().join(*axes[ii])
for pi, proj in enumerate((False, True, 'reconstruct')):
    evoked.plot(proj=proj, axes=axes[:, pi], spatial_colors=True)
    if pi == 0:
        for ax in axes[:, pi]:
            parts = ax.get_title().split('(')
            ax.set(ylabel=f'{parts[0]} ({ax.get_ylabel()})\n'
                          f'{parts[1].replace(")", "")}')
    axes[0, pi].set(title=f'proj={proj}')
    axes[0, pi].texts = []
plt.setp(axes[1:, :].ravel(), title='')
plt.setp(axes[:, 1:].ravel(), ylabel='')
plt.setp(axes[:-1, :].ravel(), xlabel='')
mne.viz.tight_layout()

###############################################################################
# Note that here the bias in the EEG and magnetometer channels is reduced by
# the reconstruction. This suggests that the application of SSP has slightly
# reduced the amplitude of our signals in sensor space, but that it should not
# bias the amplitudes in source space.
#
# References
# ^^^^^^^^^^
#
# .. footbibliography::
