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
import matplotlib.pyplot as plt

from numpy import log10, sqrt

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
# First, let's look at the raw data, noting that there are large fluctuations
# in the sub 1 Hz band. In some cases the range of fields a single sensor
# sensor is as much as 600 pT.

picks = mne.pick_types(raw.info, meg=True)

data, time = raw[picks, :]
dataDS = data[:, :-300:300]
timeDS = time[:-300:300]

plt.figure()
plt.plot(timeDS, dataDS.T - dataDS.mean(axis=1))
plt.grid()
plt.ylim((-5e-10, 5e-10))
plt.xlim((0, 400))
plt.title('No Preprocessing')


# %%
# Denoising: Regressing via reference sensors
# -------------------------------------------
#
# The simplest method for reducing the low frequency drift in the data is to
# use a set of reference sensors away from the scalp, which only sample the
# ambient fields in the room. An advantage of this method is that no prior
# knowldge of the locations of the sensors is required, however it assumes that
# the reference sensors experience the same interference as scalp recordings.
#
# To do this in our current dataset we require a bit of housekeeping.
# There are a set of channels beginning with the name "Flux" which do not
# contain any evironmental data, these need to be set to as bad channels.
#
# For now we are only interested in removing artefacts seen below 2 Hz, so we
# initially low-pass filter the good reference channels in this dataset prior
# to regression
#
# Looking at the processed data, we see there has been a large reduction in the
# low frequency drift, but there are still periods where the drift has not been
# entirely removed. The likely cause of this is that the spatial profile of the
# interference is dynamic, so performing a single regression over the entire
# experiment is not the most effective approach.

# set flux channels to bad
raw2 = raw.copy()
bads = mne.pick_channels_regexp(raw2.ch_names, regexp='Flux.')
raw2.info['bads'].extend([raw2.info['chs'][ii]['ch_name'] for ii in bads])

# filter and regress
raw2.load_data()
raw2.filter(None, 5, picks='ref_meg', method='iir')
raw2, _ = mne.preprocessing.regress_artifact(raw2,
                                             picks='meg',
                                             picks_artifact='ref_meg'
                                             )
# plot
data, _ = raw2[picks, :]
dataDS = data[:, :-300:300]
plt.figure()
plt.plot(timeDS, dataDS.T - dataDS.mean(axis=1))
plt.grid()
plt.ylim((-5e-10, 5e-10))
plt.xlim((0, 400))
plt.title('After Reference Regression')

# %%
# Comparing denoising methods
# ---------------------------
#
# Differing denoising methods will have differing levels of performance across
# different parts of the spectrum. One way to evaluate the performance of a
# denoising step is to calculate the power spectrum of the dataset before and
# after processing. We will use metric called the shielding factor to summarise
# the values. Positive shielding factors indicate a reduction in power, whilst
# negative means in increase.

spec_pre = raw.compute_psd(fmax=20, n_fft=10000, average='median')
spec_post = raw2.compute_psd(fmax=20, n_fft=10000, average='median')

psd_pre = sqrt(spec_pre[:])
psd_post = sqrt(spec_post[:])

shielding = 20 * log10(psd_pre / psd_post)

plt.figure()
plt.plot(spec_post.freqs, shielding.T)
plt.grid()
plt.xlim((0, 20))
plt.title('Reference Regression: Shielding')


# %%
# References
# ----------
# .. footbibliography::
