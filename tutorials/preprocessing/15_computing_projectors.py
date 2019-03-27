# -*- coding: utf-8 -*-
"""
.. _computing-projectors-tutorial:

Computing projectors
====================

.. include:: ../../tutorial_links.inc

This tutorial covers computing SSP projectors to reduce heartbeat and eye
movement artifacts or environmental noise.
"""

###############################################################################
# So far we've worked with projectors that were already included in the
# :class:`~mne.io.Raw` object (the "empty room" projectors), and projectors
# loaded from a separate ``.fif`` file (ECG projectors). Here we'll go through
# the steps to compute those projectors ourselves. We'll go through the process
# of computing environmental noise projectors from empty room recordings, show
# how to use dedicated ECG or EOG sensors for heartbeat and eyeblink artifact
# removal, and see how to use regular EEG/MEG sensor channels when ECG or EOG
# sensors are not available. As usual we'll start by importing the modules we
# need, and loading some example data:

import os
import numpy as np
import matplotlib.pyplot as plt
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

###############################################################################
# SSP projectors from empty room data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# *TODO why does the file :file:`ernoise_raw.fif` already contain projectors?*
#
# *TODO link to :ref:`Show noise levels from empty room data` example*
#
#
# Heartbeat (ECG) projectors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# MNE-Python provides several functions for detecting and removing heartbeats
# from EEG and MEG data. The most straightforward to use is
# :func:`~mne.preprocessing.compute_proj_ecg`, which takes a
# :class:`~mne.io.Raw` object as input and returns the requested number of
# projectors for magnetometers, gradiometers, and EEG channels (default is two
# projectors for each channel type).
# :func:`~mne.preprocessing.compute_proj_ecg` also returns an :term:`events`
# array containing the sample numbers corresponding to the onset of each
# detected heartbeat.

projs, events = mne.preprocessing.compute_proj_ecg(raw, n_grad=1, n_mag=1,
                                                   n_eeg=0, reject=None)

###############################################################################
# The first line of output tells us that
# :func:`~mne.preprocessing.compute_proj_ecg` found three existing projectors
# already in the :class:`~mne.io.Raw` object, and will include those in the
# list of projectors that it returns (appending the new ECG projectors to the
# end of the list). If you don't want that, you can change that behavior with
# the boolean ``no_proj`` parameter. Since we've already run the computation,
# we can just as easily separate out the ECG projectors by indexing the list of
# projectors:

ecg_projs = projs[-2:]
print(ecg_projs)

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
# *TODO what are the cases where you might need to customize the ECG filter?
# infants? Heart murmur?*
#
# Once the ECG events have been identified,
# :func:`~mne.preprocessing.compute_proj_ecg` will also filter the data
# channels before extracting epochs around each heartbeat, using the parameter
# values given in ``l_freq``, ``h_freq``, ``filter_length``, ``filter_method``,
# and ``iir_params``. By default, the filtered epochs will be averaged together
# before the projection is computed; this can be controlled with the boolean
# ``average`` parameter.
#
# - *TODO crossreference to filtering tutorial and filtering background
#   discussion*
# - *TODO what is the (dis)advantage of **not** averaging before projection?*
# - *TODO should advice for filtering here be the same as advice for filtering
#   raw data generally? (e.g., keep high-pass very low to avoid peak shifts?
#   what if your raw data is already filtered?)*
#
# To get a sense of how the heartbeat affects the signal at each sensor, you
# can plot the difference between the data with and without the ECG projector
# applied. Here we'll do that with both a butterfly plot and an imagemap:

# get magnetometers with/without ECG projector as separate objects
mags = raw.copy().pick_types(meg='mag').apply_proj()
mags_ecg = raw.copy().pick_types(meg='mag').add_proj(ecg_projs).apply_proj()
# compute the difference between magnetometers with/without ECG projector
stop = int(raw.time_as_index(2))
ecg_diff = (mags.get_data(stop=stop) - mags_ecg.get_data(stop=stop))
# convert from teslas to femtoteslas
ecg_diff /= 1e-15
# initialize figure
fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[2, 1]),
                        sharex=True)
# imagemap
axs[0].imshow(ecg_diff, aspect='auto')
axs[0].set_ylabel('Channel index')
# butterfly plot
axs[1].plot(ecg_diff.T, color='k', linewidth=0.5, alpha=0.2)
axs[1].set_ylabel('flux density (fT)')
# change x-axis ticks to time instead of sample number
time_ticks = np.linspace(0, 2, 5)
sampling_frequency = raw.info['sfreq']
axs[1].set_xticks([int(tick * sampling_frequency) for tick in time_ticks])
axs[1].set_xticklabels(map(str, time_ticks))
axs[1].set_xlabel('Time (s)')
fig.tight_layout()

###############################################################################
# Finally, note that above we passed ``reject=None`` to the
# :func:`~mne.preprocessing.compute_proj_ecg` function, meaning that all
# detected ECG epochs would be used when computing the projectors (regardless
# of signal quality in the data sensors during those epochs). The default
# behavior is to reject epochs based on signal amplitude: epochs with
# peak-to-peak amplitudes exceeding 50 μV in EEG channels, 250 μV in EOG
# channels, 200 pT/m in gradiometer channels, or 3 pT in magnetometer channels.
# You can change these thresholds by passing a dictionary with keys ``eeg``,
# ``eog``, ``mag``, and ``grad`` (though be sure to pass the threshold values
# in volts, teslas, or teslas/meter). Generally, it is a good idea to reject
# such epochs when computing the ECG projectors (since presumably the
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
# - *TODO add link to how-to example for extracting and plotting the ECG signal
#   itself*
#
#
# Eye movement (EOG) projectors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The construction of EOG projectors is very similar to the procedure shown
# above for ECG projectors. The function
# :func:`~mne.preprocessing.compute_proj_eog` has similar parameters to
# :func:`~mne.preprocessing.compute_proj_ecg`, though the defaults differ
# slightly:
#
# - The default rejection threshold for EEG channels is 500 μV instead of 50 μV
#   (reflecting the fact that EEG sensors are much more strongly affected by
#   eye movements than by heartbeats), and the default rejection threshold for
#   the EOG channel is infinite.
#
# - The default epoch length for EOG is shorter (400 ms instead of 600 ms),
#   reflecting the faster time course of eye movements versus heartbeats.
#
# - The default filter characteristics are different (bandpass between 1 and 10
#   Hz for identifying eye movements versus bandpass between 5 and 35 Hz for
#   identifying heartbeats), reflecting the different waveform characteristics
#   of these artifacts.
#
# - :func:`~mne.preprocessing.compute_proj_eog` will not reconstruct a virtual
#   artifact channel from magnetometers, so either there must be an EOG channel
#   present in the :class:`~mne.io.Raw` object, or you must provide an
#   alternate channel in the ``ch_name`` parameter to use when identifying eye
#   movements. Frontal EEG channels (such as electrodes Fp1 or Fp2 in the
#   `10-20 system <ten_twenty>`_) are often an acceptable choice when no EOG
#   sensors were used during data acquisition.
#
# Despite those differences, the call to
# :func:`~mne.preprocessing.compute_proj_eog` and its output should look
# familiar, though here we'll refrain from passing ``reject=False`` so you can
# see what the rejection output looks like:

eog_projs, eog_events = mne.preprocessing.compute_proj_eog(raw, n_grad=1,
                                                           n_mag=1, n_eeg=1,
                                                           no_proj=True)

###############################################################################
# Here we see that 46 blinks were detected, 6 of the blink epochs were rejected
# prior to computing the projectors that will remove the blink artifact from
# the data channels, and that the rejection was due to noise in MEG channels
# 1421 and 1411.
#
# Just like with ECG, there are complementary functions that perform individual
# parts of the EOG analysis steps that are bundled in
# :func:`~mne.preprocessing.compute_proj_eog`:
#
# - :func:`mne.preprocessing.find_eog_events` for detecting eye movements in a
#   :class:`~mne.io.Raw` object and returning a corresponding :term:`events`
#   array
#
# - :func:`mne.preprocessing.create_eog_epochs` for detecting eye movements in
#   a :class:`~mne.io.Raw` object and returning an :class:`~mne.Epochs` object
#
# - :func:`mne.compute_proj_epochs` for creating projector(s) from any
#   :class:`~mne.Epochs` object
#
# See the documentation of those functions for further details.
#
# *TODO: add crossref to how-to example of plotting the EOG artifacts
# themselves*
