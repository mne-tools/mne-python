# -*- coding: utf-8 -*-
"""
.. _computing-projectors-tutorial:

Computing projectors yourself
=============================

.. include:: ../../tutorial_links.inc

So far we've worked with projectors that were already included in the
:class:`~mne.io.Raw` object, and projectors loaded from a separate ``.fif``
file. Here we'll go through the steps to compute those projectors yourself.
This tutorial covers five examples for creating projectors using SSP:

- Environmental noise projectors from empty room data
- Heartbeat projectors from MEG data
- Heartbeat projectors from a dedicated ECG sensor
- Blink projectors from an EEG channel
- Blink projectors from dedicated EOG sensors

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
# TODO
#
#
# Heartbeat projectors from a dedicated ECG sensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# TODO
#
#
# Heartbeat projectors from MEG data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# TODO, based on
#
# - https://mne-tools.github.io/dev/auto_examples/preprocessing/plot_find_ecg_artifacts.html
# - https://mne-tools.github.io/dev/auto_tutorials/plot_artifacts_correction_ssp.html
#
#
# Blink projectors from dedicated EOG sensors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# TODO, based on
#
# - https://mne-tools.github.io/dev/auto_examples/preprocessing/plot_find_eog_artifacts.html
# - https://mne-tools.github.io/dev/auto_tutorials/plot_artifacts_correction_ssp.html
#
#
# Blink projectors from an EEG channel
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# TODO
#
