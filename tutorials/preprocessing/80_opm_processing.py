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

import matplotlib.pyplot as plt
import numpy as np

import mne

opm_data_folder = mne.datasets.ucl_opm_auditory.data_path()
opm_file = (opm_data_folder / 'sub-001' / 'ses-001' / 'meg' /
            'sub-001_ses-001_task-aef_run-001_meg.bin')
# For now we are going to assume the device and head coordinate frames are
# identical (even though this is incorrect), so we pass verbose='error' for now
raw = mne.io.read_raw_fil(opm_file, verbose='error')
raw.crop(120, 240).load_data()  # crop for speed

# %%
# Examining raw data
# ------------------
#
# First, let's look at the raw data, noting that there are large fluctuations
# in the sub 1 Hz band. In some cases the range of fields a single channel
# reports is as much as 600 pT across this experiment.

picks = mne.pick_types(raw.info, meg=True)

amp_scale = 1e12  # T->pT
stop = len(raw.times) - 300
step = 300
data_ds, time_ds = raw[picks[::5], :stop]
data_ds, time_ds = data_ds[:, ::step] * amp_scale, time_ds[::step]

fig, ax = plt.subplots(constrained_layout=True)
plot_kwargs = dict(lw=1, alpha=0.5)
ax.plot(time_ds, data_ds.T - np.mean(data_ds, axis=1), **plot_kwargs)
ax.grid(True)
set_kwargs = dict(ylim=(-500, 500), xlim=time_ds[[0, -1]],
                  xlabel='Time (s)', ylabel='Amplitude (pT)')
ax.set(title='No preprocessing', **set_kwargs)

# %%
# Denoising: Regressing via reference sensors
# -------------------------------------------
#
# The simplest method for reducing low frequency drift in the data is to
# use a set of reference sensors away from the scalp, which only sample the
# ambient fields in the room. An advantage of this method is that no prior
# knowldge of the locations of the sensors is required. However, it assumes
# that the reference sensors experience the same interference as scalp
# recordings.
#
# To do this in our current dataset, we require a bit of housekeeping.
# There are a set of channels beginning with the name "Flux" which do not
# contain any evironmental data, these need to be set to as bad channels.
#
# For now we are only interested in removing artefacts seen below 5 Hz, so we
# initially low-pass filter the good reference channels in this dataset prior
# to regression
#
# Looking at the processed data, we see there has been a large reduction in the
# low frequency drift, but there are still periods where the drift has not been
# entirely removed. The likely cause of this is that the spatial profile of the
# interference is dynamic, so performing a single regression over the entire
# experiment is not the most effective approach.

# set flux channels to bad
bad_picks = mne.pick_channels_regexp(raw.ch_names, regexp='Flux.')
raw.info['bads'].extend([raw.ch_names[ii] for ii in bad_picks])

# compute the PSD for later using 1 Hz resolution
psd_kwargs = dict(fmax=20, n_fft=int(round(raw.info['sfreq'])))
psd_pre = raw.compute_psd(**psd_kwargs)

# filter and regress
raw.filter(None, 5, picks='ref_meg')
regress = mne.preprocessing.EOGRegression(picks, picks_artifact='ref_meg')
regress.fit(raw)
regress.apply(raw, copy=False)

# plot
data_ds, _ = raw[picks[::5], :stop]
data_ds = data_ds[:, ::step] * amp_scale

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(time_ds, data_ds.T - np.mean(data_ds, axis=1), **plot_kwargs)
ax.grid(True, ls=':')
ax.set(title='After reference regression', **set_kwargs)

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

# psd_pre was computed above before regression
psd_post = raw.compute_psd(**psd_kwargs)
shielding = 10 * np.log10(psd_pre[:] / psd_post[:])

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(psd_post.freqs, shielding.T, **plot_kwargs)
ax.grid(True, ls=':')
ax.set(xticks=psd_post.freqs)
ax.set(xlim=(0, 20), title='Reference regression shielding',
       xlabel='Frequency (Hz)', ylabel='Shielding (dB)')

# %%
# Filtering nuisance signals
# ---------------------------------
#
# Having regressed much of the high-amplitude, low-frequency interference, we
# can now look to filtering the remnant nuisance signals. The motivation for
# filtering after regression (rather than before) is to minimise any filter
# artefacts generated when removing such high-amplitude interfece (compared
# to the neural signals we are interested in).
#
# We are going to remove the 50 Hz mains signal with a notch filter,
# followed by a bandpass filter between 1 and 48 Hz. From here it becomes clear
# that the variance in our signal has been reduced from 100s of pT to 10s of
# pT instead.

# notch
raw.notch_filter(np.arange(50, 251, 50))
# bandpass
raw.filter(1, 48, picks='meg')
# plot
data_ds, _ = raw[picks[::5], :stop]
data_ds = data_ds[:, ::step] * amp_scale
fig, ax = plt.subplots(constrained_layout=True)
plot_kwargs = dict(lw=1, alpha=0.5)
ax.plot(time_ds, data_ds.T - np.mean(data_ds, axis=1), **plot_kwargs)
ax.grid(True)
set_kwargs = dict(ylim=(-500, 500), xlim=time_ds[[0, -1]],
                  xlabel='Time (s)', ylabel='Amplitude (pT)')
ax.set(title='After regression and filtering', **set_kwargs)

# %%
# Generating an evoked response
# ----------------------------
#
# With the data preprocessed, it is now possible to see an auditory evoked
# response at the sensor level.

events = mne.find_events(raw, min_duration=0.1)
epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=0.4, baseline=(-0.1, 0.))
evoked = epochs.average()
evoked.plot()

# %%
# References
# ----------
# .. footbibliography::
