# -*- coding: utf-8 -*-
"""
.. _computing-projectors-tutorial:

Computing projectors
====================

.. include:: ../../tutorial_links.inc

So far we've worked with projectors that were already included in the
:class:`~mne.io.Raw` object, and projectors loaded from a separate ``.fif``
file. Here we'll go through the steps to compute those projectors yourself.
This tutorial covers three examples for creating projectors using SSP:

- Environmental noise projectors from empty room data
- Heartbeat projectors (from MEG or ECG channels)
- Blink projectors (from EEG or EOG channels)

As usual we'll start by importing the modules we need, and loading some example
data:
"""

import os
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

###############################################################################
# SSP projectors from empty room data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# *TODO why does the file ``ernoise_raw.fif`` already contain projectors?
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
# :func:`~mne.preprocessing.compute_proj_ecg` also returns an
# :class:`~mne.Events` array containing the sample numbers corresponding to the
# onset each detected heartbeat.

projs, events = mne.preprocessing.compute_proj_ecg(raw, n_grad=1, n_mag=1,
                                                   n_eeg=0, reject=None)

###############################################################################
# The first line of output tells us that
# :func:`~mne.preprocessing.compute_proj_ecg` found three existing projectors
# already in the :class:`~mne.io.Raw` object, and will include those in the
# list of projectors that it returns (appending the new ECG projectors to the
# end of the list). If you don't want that, you can change that behavior with
# the boolean ``no_proj`` parameter:

ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, n_grad=1,
                                                           n_mag=1, n_eeg=0,
                                                           reject=None,
                                                           no_proj=True)
print(ecg_projs)

###############################################################################
# Since no dedicated ECG sensor channel was detected in the
# :class:`~mne.io.Raw` object, by default
# :func:`~mne.preprocessing.compute_proj_ecg` uses the magnetometers to
# estimate the ECG signal (as stated on the third line of output). You can also
# supply the ``ch_name`` parameter to restrict which channel to use for ECG
# artifact detection; this is most useful when you had an ECG sensor but it is
# not labeled as such in the :class:`~mne.io.Raw` file.
#
# The next few lines of the output describe the filter used to isolate ECG
# events. The default settings are usually adequate, but the filter can be
# customized via the parameters ``ecg_l_freq``, ``ecg_h_freq``, and
# ``filter_length`` (see the docstring for details).
#
# *TODO what are the cases where you might need to customize the ECG filter?
# infants? Heart murmur?*
#
# Once the ECG events have been identified,
# :func:`~mne.preprocessing.compute_proj_ecg` will filter the data channels
# before extracting epochs around each heartbeat, using the parameter values
# given in ``l_freq``, ``h_freq``, ``filter_length``, ``filter_method``, and
# ``iir_params``. By default, the filtered epochs will be averaged together
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
# can plot the topographic distribution of each projector, using the
# :class:`~mne.Projection` object's :meth:`~mne.Projection.plot_topomap`
# method:

for proj in ecg_projs:
    proj.plot_topomap()

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
#     Note that there is a similar parameter ``flat`` for specifying the
#     *minimum* acceptable peak-to-peak amplitude for each channel type.
#
# While :func:`~mne.preprocessing.compute_proj_ecg` conveniently combines
# several operations into a single function, MNE-Python also provides functions
# for performing each part of the process. Specifically:
#
# - :func:`mne.preprocessing.find_ecg_events` for detecting heartbeats in a
#   :class:`~mne.io.Raw` object and returning a corresponding
#   :class:`~mne.Events` object
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
# - the default rejection threshold for EEG channels is 500 μV instead of 50 μV
#   (reflecting the fact that EEG sensors are much more strongly affected by
#   eye movements than by heartbeats), and the default rejection threshold for
#   the EOG channel is infinite
#
# - the default epoch length for EOG is shorter (400 ms instead of 600 ms),
#   reflecting the faster time course of eye movements versus heartbeats
#
# - the default filter characteristics are different (bandpass between 1 and 10
#   Hz for identifying eye movements versus bandpass between 5 and 35 Hz for
#   identifying heartbeats), reflecting the different waveform characteristics
#   of these artifacts
#
#
# - :func:`~mne.preprocessing.compute_proj_eog` will not reconstruct a virtual
#   artifact channel from magnetometers, so either there must be an EOG channel
#   present in the :class:`~mne.io.Raw` object, or you must provide an
#   alternate channel in the ``ch_name`` parameter to use when identifying eye
#   movements. Frontal EEG channels (such as electrodes Fp1 or Fp2 in the
#   `10-20 system <ten_twenty>`_) are often an acceptable choice when no EOG
#   sensors were used during recording.
#
# Despite those differences, the call to
# :func:`~mne.preprocessing.compute_proj_eog` and its output should look
# familiar, though here we'll refrain from passing ``reject=False`` so you can
# see what the rejection output looks like:

eog_projs, eog_events = mne.preprocessing.compute_proj_eog(raw, n_grad=1,
                                                           n_mag=1, n_eeg=1,
                                                           no_proj=True)

###############################################################################
# Here we see that 46 blinks were detected, and 6 of the blink epochs were
# rejected prior to computing the projectors that will remove the blink
# artifact from the data channels, and that the rejection was due to noise in
# MEG channels 1421 and 1411.
#
# Just like with ECG, there are complementary functions that perform individual
# parts of the EOG analysis steps that are bundled in
# :func:`~mne.preprocessing.compute_proj_eog`:
#
# - :func:`mne.preprocessing.find_eog_events` for detecting eye movements in a
#   :class:`~mne.io.Raw` object and returning a corresponding
#   :class:`~mne.Events` object
#
# - :func:`mne.preprocessing.create_eog_epochs` for detecting eye movements in
#   a :class:`~mne.io.Raw` object and returning an :class:`~mne.Epochs` object
#
# - :func:`mne.compute_proj_epochs` for creating projector(s) from any
#   :class:`~mne.Epochs` object
#
# See the documentation of those functions for further details.
