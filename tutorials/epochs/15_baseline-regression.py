# -*- coding: utf-8 -*-
"""
.. _ex-baseline-regression:

====================================================================
Regression-based baseline correction
====================================================================

Traditional baseline correction simply adds or subtracts a scalar amount from
every timepoint in an epoch, such that the mean value during the defined
baseline period is zero. Two difficulties are usually discussed in traditional
baseline correction:
 - which baseline time window to choose
 - no baseline differences between conditions

The method introduced by :footcite:`Alday2019`, allows to account for the fact
that there might be baseline differences between conditions that the we are
not aware of. By including the baseline interval as a regressor in a
general linear model, we allow the data to determine the amount of baseline
correction needed.
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

# first we define variables for epoching and preprocessing of the data

tmin, tmax = -0.2, 0.5
lowpass, highpass = 40, 0.1
baseline_tmin, baseline_tmax = None, 0

# we select a single channel of interest
# we combine both hemispheres and therefore select a central electrode
# https://mne.tools/stable/auto_tutorials/intro/10_overview.html#sphx-glr-auto-tutorials-intro-10-overview-py

ch = "EEG 021"

# now we load the audiovisual example dataset for one subject

data_path = sample.data_path()

subjects_dir = data_path / 'subjects'
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'

# %% here we load the raw data for one subject

raw = mne.io.read_raw_fif(raw_fname, preload=True)

# we check for events saved while data data collection

events = mne.find_events(raw)

# here we pick only EEG channels

raw.pick_types(stim=False, eog=False, eeg=True)

# %% Basic preprocessing

# we start by bandpass filtering the data

raw_filtered = raw.copy().filter(highpass, lowpass)

# %% Epoching

# here we merge visual and auditory events from both hemispheres

events = mne.merge_events(events, [1, 2], 1)  # auditory events
events = mne.merge_events(events, [3, 4], 2)  # visual events

# we store the events of interest in a dictionary

event_id = {'auditory': 1, 'visual': 2}

# now we epoch the data based on the selected events, tmin and tmax
# Note that we don't baseline correct the epochs by specifying baseline=None
# we clean the data minimally by rejecting channels with very high or low
# amplitudes

epochs = mne.Epochs(raw_filtered,
                    events, event_id,
                    tmin=tmin, tmax=tmax,
                    reject=dict(eeg=150e-6),
                    flat=dict(eeg=5e-6),
                    baseline=None,
                    preload=True)

# here we delete the raw data to free up memory

del raw_filtered

# %% traditional baseline correction

# first let's baseline correct the data as we usually would do
# we average over all epochs and subtract the condition specific baseline
# for both events of interest

auditory_baseline = (epochs[epochs.events[:, 2] == 1].copy()
                     .average()
                     .apply_baseline((baseline_tmin, baseline_tmax)))

visual_baseline = (epochs[epochs.events[:, 2] == 2].copy()
                   .average()
                   .apply_baseline((baseline_tmin, baseline_tmax)))

# %% plot evoked data baseline corrected the traditional way

# here we choose a baseline window of 200 miliseconds
# feel free to try out different baseline windows

fig = mne.viz.plot_compare_evokeds({"auditory": auditory_baseline,
                                   "visual": visual_baseline},
                                   picks=[ch],
                                   title="traditional baseline correction",
                                   show_sensors=False, legend=True,
                                   truncate_yaxis=False,
                                   colors=dict(auditory="b", visual="r"))

# we can see that there is a lot going on in the baseline and the baseline
# activity seems to differ between conditions

# Next we subtract the visual evoked response from the auditory evoked
# response and plot the difference topography of the poststimulus window

diff = mne.combine_evoked([auditory_baseline, visual_baseline], [1, -1])

diff.plot_topomap(times=[0., 0.05, 0.1, 0.2, 0.3, 0.4], ch_type='eeg')

# we can see that the biggest difference is around 300 miliseconds with a
# dipole in central channels

# %% regression-based baseline correction

# let's now try out the regression based baseline correction approach

# therefore we need to crop the data in the desired baseline time window
# and save it as a numpy array

# baseline_epochs = epochs.copy().crop(baseline_tmin, baseline_tmax)

baseline_epochs = epochs.copy().pick_channels([ch]).crop(baseline_tmin,
                                                         baseline_tmax)

epoch_data = baseline_epochs.get_data()

# here we take the mean over the last axis (time samples), get rid of the 0
# dimension and multiply the data with 1000 000 to get micro volt for fitting
# purposes

baseline = epoch_data.mean(axis=-1).squeeze() * 1e6

# next we set up the design matrix
# each row is an epoch and each column is a regressor: here we define
# 4 regressor: auditory events, visual events, baseline and the interaction
# between the baseline and the conditions

design_matrix = np.stack([epochs.events[:, 2] == 1, epochs.events[:, 2] == 2,
                          baseline, baseline * (epochs.events[:, 2] == 2)]).T

# resulting shape of the design matrix should be number of epochs per subject
# x number of regressor (so in that example 286 epochs x 4 regressor)

# finally we fit the regression model

# we add the epochs for both conditions, the design matrix and the names for
# the regressors included in the model

regmodel = mne.stats.linear_regression(epochs, design_matrix,
                                       names=["auditory", "visual",
                                              "baseline",
                                              "baseline:visual"])

# after fitting is done, we can extract the beta values for each condition

regbaseline_aud = regmodel['auditory'].beta
regbaseline_vis = regmodel['visual'].beta

# %% plot regression output

plt.plot(regmodel['baseline'].beta.pick_channels([ch]).get_data().squeeze())
plt.show()

fig = mne.viz.plot_compare_evokeds({'auditory': regbaseline_aud,
                                    'visual': regbaseline_vis},
                                   invert_y=True,
                                   picks=ch,
                                   title="baseline correction "
                                         "based on regression",
                                   show_sensors=False,
                                   colors=dict(auditory="k", visual="r"),
                                   truncate_yaxis=False)

# let's look at the topography of the contrast auditory-visual evoked response

regressed_diff = mne.combine_evoked([regbaseline_aud, regbaseline_vis],
                                    [1, -1])

regressed_diff.plot_topomap(times=[0., 0.05, 0.1, 0.2, 0.3, 0.4],
                            ch_type='eeg')
