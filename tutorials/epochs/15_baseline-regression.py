# -*- coding: utf-8 -*-
"""
.. _ex-baseline-regression:

====================================================================
Regression-based baseline correction
====================================================================

Traditional baseline correction simply adds or subtracts a scalar amount from every
timepoint in an epoch, such that the mean value during the defined baseline period
is zero. Two difficulties are usually discussed in traditional baseline correction:
 - which baseline time window to choose
 - no baseline differences between conditions 
The method introduced by :footcite:`Alday2019`, allows to address the second part.
 By including the baseline interval as a regressor in a general linear model,
 we allow the data to determine the amount of baseline correction needed.
We demonstrate the alternative regression-based method recommended, where the
*strength of the effect* of the baseline period is allowed to vary across each
timepoint of the epoch.

(adapted from code by Phillip Alday: https://osf.io/85bjq)
"""
# Authors: Carina Forster
# Email: carinaforster0611@gmail.com
#
# License: BSD-3-Clause

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample

# %%

# we load the example dataset for one subject

data_path = sample.data_path()

subjects_dir = data_path / 'subjects'
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'

# load variables

tmin, tmax = -0.2, 0.5
lowpass, highpass = 40, 0.1
baseline_tmin, baseline_tmax = None, 0

# we select a single channel of interest

ch = "EEG 021"  # we combine both hemispheres, therefore a central electrode was selected for analysis
# https://mne.tools/stable/auto_tutorials/intro/10_overview.html#sphx-glr-auto-tutorials-intro-10-overview-py
# %% load the raw data for one subject

raw = mne.io.read_raw_fif(raw_fname, preload=True)

# we store the events

events = mne.find_events(raw)

# we are only interested in EEG data

raw.pick_types(stim=False, eog=False, eeg=True)

# %%
# we start by bandpass filtering the data

raw_filtered = raw.copy().filter(highpass, lowpass)

# %% We epoch the data

# here we merge visual and auditory events from both hemispheres

events = mne.merge_events(events, [1, 2], 1)
events = mne.merge_events(events, [3, 4], 2)

# we have 2 events of interest

event_id = {'auditory': 1, 'visual': 2}

epochs = mne.Epochs(raw_filtered,
                    events, event_id,
                    tmin=tmin, tmax=tmax,
                    reject=dict(eeg=150e-6),
                    flat=dict(eeg=5e-6),
                    baseline=None,
                    preload=True)

del raw_filtered  # free up memory

# %% we average the epochs per condition and baseline correct the data

auditory_baseline = (epochs[epochs.events[:, 2] == 1].copy()
                     .average()
                     .apply_baseline((baseline_tmin, baseline_tmax)))

visual_baseline = (epochs[epochs.events[:, 2] == 2].copy()
                   .average()
                   .apply_baseline((baseline_tmin, baseline_tmax)))

# %% traditional baseline correction

fig = mne.viz.plot_compare_evokeds({"auditory": auditory_baseline,
                                   "visual": visual_baseline},
                                   picks=[ch],
                                   invert_y=True,
                                   title="traditional baseline correction",
                                   show_sensors=False, legend=True,
                                   colors=dict(auditory="k", visual="r"),
                                   truncate_yaxis=False)

# let's look at the topography of the contrast auditory - visual evoked response

diff = mne.combine_evoked([auditory_baseline, visual_baseline], [1, -1])

diff.plot_topomap(times=[0., 0.05, 0.1, 0.2, 0.3, 0.4], ch_type='eeg')

# %% regression based baseline correction

# we crop the data in the desired baseline time window and save the data

# baseline_epochs = epochs.copy().crop(baseline_tmin, baseline_tmax)

baseline_epochs = epochs.copy().pick_channels([ch]).crop(baseline_tmin, baseline_tmax)

epoch_data = baseline_epochs.get_data()

# here we take the mean over the last axis (time samples)

baseline = epoch_data.mean(axis=-1).squeeze() * 1e6

# we convert to millivolt to avoid very small volt values that
# are hard for the regression model to fit

# here we set up the design matrix, each column is a regressor and
# ech row is a epoch

design_matrix = np.stack([epochs.events[:, 2] == 1, epochs.events[:, 2] == 2,
                          baseline, baseline * (epochs.events[:, 2] == 2)]).T

# shape should be number of epochs per subject x number of regressor (so in that example 286 x 4)

# fit regression model

# we fit the model for all channels so that we can look at the topography later

regmodel = mne.stats.linear_regression(epochs, design_matrix,
                                       names=["auditory", "visual",
                                              "baseline",
                                              "baseline:visual"])

regbaseline_aud = regmodel['auditory'].beta
regbaseline_vis = regmodel['visual'].beta

# %% plot regression output

plt.plot(regmodel['baseline'].beta.pick_channels([ch]).get_data().squeeze())
plt.show()

fig = mne.viz.plot_compare_evokeds({'auditory': regbaseline_aud, 'visual': regbaseline_vis},
                                   invert_y=True,
                                   picks=ch,
                                   title="baseline correction based on regression",
                                   show_sensors=False,
                                   colors=dict(auditory="k", visual="r"),
                                   truncate_yaxis=False)

# let's look at the topography of the contrast auditory - visual evoked response

regressed_diff = mne.combine_evoked([regbaseline_aud, regbaseline_vis], [1, -1])

regressed_diff.plot_topomap(times=[0., 0.05, 0.1, 0.2, 0.3, 0.4], ch_type='eeg')
