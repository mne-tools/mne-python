"""
.. _tut_artifacts_detect:

Artifacts Detection
===================

This tutorial discusses a couple of major artifacts that most analyses
have to deal with and demonstrates how to detect them.

"""

import mne
from mne.datasets import sample
from mne.preprocessing import create_ecg_epochs, create_eog_epochs

# getting some data ready
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)


###############################################################################
# Low frequency drifts and line noise

(raw.copy().pick_types(meg='mag')
           .del_proj(0)
           .plot(duration=60, n_channels=100, remove_dc=False))

###############################################################################
# we see high amplitude undulations in low frequencies, spanning across tens of
# seconds

raw.plot_psd(fmax=250)

###############################################################################
# On MEG sensors we see narrow frequency peaks at 60, 120, 180, 240 Hz,
# related to line noise.
# But also some high amplitude signals between 25 and 32 Hz, hinting at other
# biological artifacts such as ECG. These can be most easily detected in the
# time domain using MNE helper functions
#
# See :ref:`tut_artifacts_filter`.

###############################################################################
# ECG
# ---
#
# finds ECG events, creates epochs, averages and plots

average_ecg = create_ecg_epochs(raw).average()
print('We found %i ECG events' % average_ecg.nave)
average_ecg.plot_joint()

###############################################################################
# we can see typical time courses and non dipolar topographies
# not the order of magnitude of the average artifact related signal and
# compare this to what you observe for brain signals

###############################################################################
# EOG
# ---

average_eog = create_eog_epochs(raw).average()
print('We found %i EOG events' % average_eog.nave)
average_eog.plot_joint()

###############################################################################
# Knowing these artifact patterns is of paramount importance when
# judging about the quality of artifact removal techniques such as SSP or ICA.
# As a rule of thumb you need artifact amplitudes orders of magnitude higher
# than your signal of interest and you need a few of such events in order
# to find decompositions that allow you to estimate and remove patterns related
# to artifacts.
#
# Consider the following tutorials for correcting this class of artifacts:
#     - :ref:`tut_artifacts_filter`
#     - :ref:`tut_artifacts_correct_ica`
#     - :ref:`tut_artifacts_correct_ssp`
