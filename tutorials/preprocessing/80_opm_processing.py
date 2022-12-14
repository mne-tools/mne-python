# -*- coding: utf-8 -*-
"""
.. _tut-opm-processing:

==========================================================
Preprocessing optically pumped magnetometer (OPM) MEG data
==========================================================

This tutorial covers preprocessing steps that are specific to :term:`OPM`
MEG data. OPMs use a different sensing technology than traditional
:term:`SQUID` MEG systems, which leads to several important differences for
analysis:

- They are sensitive to :term:`DC` magnetic fields
- Sensor layouts can vary by participant and recording session due to flexible
  sensor placement
- Devices are typically not fixed in place, so the position of the sensors
  relative to the room (and through the DC fields) can change over time

We will cover some of these considerations here by processing the
:ref:`UCL OPM auditory dataset <ucl-opm-auditory-dataset>`
:footcite:`SeymourEtAl2022`
"""

# %%

import mne

opm_data_folder = mne.datasets.ucl_opm_auditory.data_path()
opm_file = (opm_data_folder / 'sub-001' / 'ses-001' / 'meg' /
            'sub-001_ses-001_task-aef_run-001_meg.bin')
# For now we are going to assume the device and head coordinate frames are
# identical (even though this is incorrect), so we pass verbose='error' for now
raw = mne.io.read_raw_fil(opm_file, verbose='error')

# %%
# Examining raw data
# ------------------
#
# First, let's look at the raw data, noting that there are huge fluctuations:

raw.plot(scalings=dict(mag=1e-11), n_channels=30, duration=30)

# $$
# References
# ----------
# .. footbibliography::
