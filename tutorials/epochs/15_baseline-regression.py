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
We demonstrate a alternative regression-based method, which allows the
*strength of the effect* of the baseline period to vary across each
timepoint of the epoch.

"""

# Authors: Carina Forster
# Email: carinaforster0611@gmail.com

# License: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.datasets import sample

# %% define variables

# first we define variables needed to epoch and preprocess the data

tmin, tmax = -0.2, 0.5
lowpass, highpass = 40, 0.1
baseline_tmin, baseline_tmax = None, 0  # None takes the first timepoint

# we select a single central electrode as we combine activity from both
# hemispheres

ch = "EEG 021"

# now we load the audiovisual example dataset for one subject

data_path = sample.data_path()

subjects_dir = data_path / 'subjects'
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'

# %% load data

raw = mne.io.read_raw_fif(raw_fname, preload=True)

# we check for events saved while data data collection

events = mne.find_events(raw)

# next we pick only EEG channels

raw.pick_types(stim=False, eog=False, eeg=True)

# %% preprocess data

# we start by bandpass filtering the data

raw_filtered = raw.copy().filter(highpass, lowpass)

# %% create epochs

# here we merge visual and auditory events from both hemispheres

events = mne.merge_events(events, [1, 2], 1)  # auditory events
events = mne.merge_events(events, [3, 4], 2)  # visual events

# we store the events of interest in a dictionary

event_id = {'auditory': 1, 'visual': 2}

# now we epoch the data for auditory and visual events
# Note that we don't baseline correct the epochs by specifying baseline=None
# we minimally clean the data by rejecting channels with very high or low
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

# %% baseline correct the data the traditional way

# first let's baseline correct the data as we usually do
# we average over all epochs and subtract the condition specific baseline
# for auditory and visual events

trad_aud = (epochs[epochs.events[:, 2] == 1].copy()
            .average()
            .apply_baseline((baseline_tmin, baseline_tmax)))

trad_vis = (epochs[epochs.events[:, 2] == 2].copy()
            .average()
            .apply_baseline((baseline_tmin, baseline_tmax)))

# %% plot evoked data baseline corrected the traditional way

# here we choose a baseline window of 200 milli seconds
# feel free to try out different baseline windows

fig = mne.viz.plot_compare_evokeds({"auditory": trad_aud,
                                   "visual": trad_vis},
                                   picks=[ch],
                                   title="traditional baseline correction",
                                   show_sensors=False, legend=True,
                                   truncate_yaxis=False,
                                   colors=dict(auditory="b", visual="r"))

# we can see that there is a lot going on in the baseline and the baseline
# activity seems to differ between conditions

# %% baseline correct data using a regression-based approach

# let's now try out the regression-based baseline correction approach

# therefore we need to crop the data in the desired baseline time window
# and save the data as a numpy array

# baseline_epochs = epochs.copy().crop(baseline_tmin, baseline_tmax)

baseline_epochs = epochs.copy().pick_channels([ch]).crop(baseline_tmin,
                                                         baseline_tmax)

epoch_data = baseline_epochs.get_data()

# here we take the mean over the last axis (time samples), get rid of the 0
# dimension and multiply the data with 1e6 to convert to micro volt for fitting
# purposes (very small values are not easy to fit for a regression model)

baseline = epoch_data.mean(axis=-1).squeeze() * 1e6

# next we set up the design matrix
# each row is an epoch and each column is a regressor: here we define
# 4 regressors: auditory events, visual events, baseline and the interaction
# between the baseline and the visual condition

design_matrix = np.stack([epochs.events[:, 2] == 1, epochs.events[:, 2] == 2,
                          baseline, baseline * (epochs.events[:, 2] == 2)]).T

# resulting shape of the design matrix should be number of epochs per subject
# x number of regressors (here: 286 epochs x 4 regressors)

# finally we fit the regression model

# we add the epochs for both conditions, the design matrix and the names for
# the regressors included in the model

reg_model = mne.stats.linear_regression(epochs, design_matrix,
                                        names=["auditory", "visual",
                                               "baseline",
                                               "baseline:visual"])

# after fitting is done, we extract the beta values for each condition

reg_aud = reg_model['auditory'].beta
reg_vis = reg_model['visual'].beta

# %% plot evoked data baseline corrected with a regression approach

fig = mne.viz.plot_compare_evokeds({'auditory': reg_aud,
                                    'visual': reg_vis},
                                   picks=ch,
                                   title="baseline correction "
                                         "based on regression",
                                   show_sensors=False,
                                   colors=dict(auditory="b", visual="r"),
                                   truncate_yaxis=False)

# %% compare sensor topography for both approaches

# Finally, let's compare the topographies for the traditional and regression
# approach

# here we calculate the difference between both conditions for the traditional
# and the regression approach

diff_traditional = mne.combine_evoked([trad_aud, trad_vis],
                                      [1, -1])

diff_regression = mne.combine_evoked([reg_aud, reg_vis],
                                     [1, -1])

# %% plot topography of auditory-visual contrast with traditional baseline
# correction

diff_traditional.plot_topomap(times=[0.1, 0.2, 0.3, 0.4],
                              ch_type='eeg', title="traditional "
                                                   "baseline correction")
# %% plot topography of auditory-visual contrast with regression-based baseline
# correction

diff_regression.plot_topomap(times=[0.1, 0.2, 0.3, 0.4],
                             ch_type='eeg', title="regression-based"
                                                  " baseline correction")

# we can see that the biggest difference is around 300 ms after stimulus onset
# with a strong dipole in central channels, this difference is much less
# pronounced in the regression-based baseline correction approach

# %% compare both approaches in one plot

# let's compare the traditional with the regression approach
# therefore we plot the difference between the auditory and the
# visual condition with traditional and regression-based baseline correction

fig = mne.viz.plot_compare_evokeds({'traditional': diff_traditional,
                                    'regression': diff_regression},
                                   picks=ch,
                                   title="difference evoked potential",
                                   show_sensors=False,
                                   colors=dict(traditional="magenta",
                                               regression="orange"),
                                   truncate_yaxis=False)

# %% plot baseline regressor

# eventually, we can check the impact of the baseline and interaction
# regressor over time

reg_baseline = reg_model['baseline'].beta
reg_interaction = reg_model['baseline:visual'].beta

# we choose the beta values for the central channel we selected
# therefore we extract the channel index

ch_index = raw.info.ch_names.index(ch)

# here we plot the beta values over time

plt.plot(reg_baseline.times, reg_baseline.get_data()[ch_index],
         color='darkgreen')
plt.xlabel('Time (s)')
plt.ylabel('Volt')
plt.title('"baseline" beta values over time')

# we can see that the baseline beta values are decreasing over time,
# therefore early evoked potentials are "more" baseline corrected than late
# potentials

# %% plot interaction regressor

# Next we plot the interaction beta values over time

plt.plot(reg_interaction.times, reg_interaction.get_data()[ch_index],
         color='darkblue')
plt.xlabel('Time (s)')
plt.ylabel('Volt')
plt.title('"interaction" beta values over time')
plt.show()

# the interaction beta weights are rather small with a strong, negative peak
# around 500 ms post stimulus

# %%
# References
# ==========
# .. footbibliography::
