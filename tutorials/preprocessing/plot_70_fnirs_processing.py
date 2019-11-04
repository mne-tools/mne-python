"""
.. _tut-fnirs-processing:

Preprocessing fNIRS data
========================

This tutorial covers how to convert fNIRS data from raw measurements to
HbO/HbR.

.. contents:: Page contents
   :local:
   :depth: 2

Here we will work with the :ref:`fNIRS motor data <fnirs-motor-dataset>`.
"""
# sphinx_gallery_thumbnail_number = 3

import os
import mne

fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()
raw_intensity.plot()

###############################################################################
# Converting from raw intensity to optical density
# ------------------------------------------------
#
# The first thing we should do is convert from raw intensity values ...

raw_od = mne.preprocessing.optical_density(raw_intensity)
raw_od.plot()

###############################################################################
# Converting from optical density to hemoglobin
# ---------------------------------------------
#
# Next we Beer-Lambert ...

raw_fnirs = mne.preprocessing.beer_lambert_law(raw_od)
raw_fnirs.plot()
