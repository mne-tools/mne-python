# -*- coding: utf-8 -*-
"""
.. _ex-baseline_regression:

====================================================================
Baseline correction using generalized linear mixed models
====================================================================

By including the baseline interval as a regressor in a generalized
linear model, we allow the data to determine how much baseline
correction is necessary for the specific epoch. This approach can
increase statistical power (Alday et al., 2019)
"""
# Authors: Carina Forster
#
# License: BSD-3-Clause

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample

# %%

#we load example data for one subject

data_path = sample.data_path()

subjects_dir = data_path / 'subjects'
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'
tmin, tmax = -0.200, 0.500
event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3, 'visual/right': 4}
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.pick_types(stim=True, eog=True, exclude=(), eeg=True)
lowpass = 40
highpass = 0.5
baseline_tmin = -0.2
baseline_tmax = 0
ch = "EEG 021"
sfreq = raw.info['sfreq']

# how many channels?

n_ch = len(raw.ch_names)

# %%
# we start by bandpass filtering the data

raw_filtered = raw.copy().filter(highpass, lowpass,
                                l_trans_bandwidth='auto',
                                h_trans_bandwidth='auto',
                                filter_length='auto',
                                method='fir',
                                fir_window='hamming',
                                fir_design='firwin',
                                phase='zero')


baseline = None # no baseline correction
reject_criteria = dict(eeg=150e-6) #  we use just absolute treshold and not peak-to-peak
flat = dict(eeg=5e-6)
detrend = 0 # DC offset

# %% We add events and epoch the data

events = mne.find_events(raw_filtered)

epochs = mne.Epochs(raw_filtered,
                            events, event_id,
                            tmin=tmin, tmax=tmax,
                            reject=reject_criteria,
                            flat=flat,
                            baseline=baseline,
                            detrend=detrend,
                            preload=True,
                            on_missing="ignore" # ignore not having every single conditions
                            )
epochs.drop_bad()

conds_we_care_about = ['auditory/left', 'auditory/right',
                       'visual/left', 'visual/right']

epochs.equalize_event_counts(conds_we_care_about)  # this operates in-place

# 70 epochs per condition

aud_epochs = epochs['auditory']
vis_epochs = epochs['visual']

all_epochs = mne.concatenate_epochs([aud_epochs, vis_epochs])

del raw_filtered # free up memory

# %% We store the evoked data for each condition in a dictionary separately for baseline
# and non baseline corrected data

# combine auditory and visual epochs in a list

epochs_combined = [aud_epochs, vis_epochs]

evokeds = dict(visual=[],
               auditory=[])

evokeds_baseline = dict(visual=[],
               auditory=[])

for cond,epoch in zip(evokeds, epochs_combined):
        evokeds[cond].append(epoch.average())
        evokeds_baseline[cond].append(epoch.copy().apply_baseline((baseline_tmin, baseline_tmax)).average())

# %% no baseline correction
# no CI because only one subject

fig = mne.viz.plot_compare_evokeds(evokeds,
                             picks=[ch],
                             invert_y=True, ci=0.95,
                             title="no baseline correction",
                             show_sensors=False, legend=True,
                             colors=dict(auditory="k",visual="r"),
                             truncate_yaxis=False)

evokeds_diff = dict(diff=[mne.combine_evoked(e, [-1, 1]) for e in
                          zip(evokeds["visual"], evokeds["auditory"])])

fig = mne.viz.plot_compare_evokeds(evokeds_diff,
                             picks=[ch],
                             invert_y=True, ci=0.95,
                             title="no baseline correction",
                             show_sensors=False, legend=True,
                             truncate_yaxis=False)

evokeds_diff['diff'][0].plot_topomap(times=[0., 0.1, 0.2, 0.3, 0.4], ch_type='eeg')

# %% traditional baseline correction

print(baseline_tmax, baseline_tmax)

fig = mne.viz.plot_compare_evokeds(evokeds_baseline,
                             picks=[ch],
                             invert_y=True, ci=0.95,
                             title="traditional baseline correction",
                             show_sensors=False, legend=True,
                             colors=dict(auditory="k",visual="r"),
                             truncate_yaxis=False)

evokeds_diff = dict(diff=[mne.combine_evoked(e,[-1, 1]) for e in
                          zip(evokeds_baseline["visual"],evokeds_baseline["auditory"])])

fig = mne.viz.plot_compare_evokeds(evokeds_diff,
                             picks=[ch],
                             invert_y=True, ci=0.95,
                             title="traditional baseline correction",
                             show_sensors=False, legend=True,
                             truncate_yaxis=False)

evokeds_diff['diff'][0].plot_topomap(times=[0., 0.1, 0.2, 0.3, 0.4], ch_type='eeg')

# %% regression based baseline correction

baseline_epochs = all_epochs.copy().pick_channels([ch]).crop(baseline_tmin, baseline_tmax)
epoch_data = baseline_epochs.get_data()

# this is dependent on a 500 Hz sampling rate
# and epochs cropped to 100ms pre-stimulus
# such that the first 50 samples correspond to the target 100ms prestimulus baseline
#baseline_window = int(sfreq*abs(baseline_tmin))

baseline = epoch_data.mean(axis=-1).squeeze() * 1e6  # convert to µV

auditory = ((all_epochs.events[:, 2] == 1) | (all_epochs.events[:, 2] == 2))
visual = ((all_epochs.events[:, 2] == 3) | (all_epochs.events[:, 2] == 4))

#set up design matrix

design_matrix = np.stack([auditory, visual, baseline, baseline*visual]).T

# shape should be number of epochs per subject x number of regressors (so in that example 280 x 4)

# fit regression model
# all_epochs should have 280 epochs

# we fit the model for one channel only

epochs_ch = all_epochs.pick_channels([ch])

regmodel = mne.stats.linear_regression(epochs_ch, design_matrix,
                                               names=["auditory", "visual",
                                                      "baseline",
                                                      "baseline:visual"])

# check why?
#<ipython-input-67-6077694f2bba>:1: RuntimeWarning: Fitting linear model to non-data or bad channels. Check picking
#  regmodel = mne.stats.linear_regression(all_epochs, design_matrix,

# %% interpret regression output

evokeds_regression = dict(visual=[],
               auditory=[])

for cond in evokeds_regression:
    evokeds_regression[cond].append(regmodel[cond].beta)
    if cond == "visual":
        bs = mne.combine_evoked([regmodel['baseline'].beta,
                                 regmodel['baseline:visual'].beta],
                                [1, 1])
    else:
        bs = regmodel['baseline'].beta

# %% plot regression output

fig = mne.viz.plot_compare_evokeds(evokeds_regression,
                             invert_y=True,
                             picks=ch,
                             title="baseline correction based on regression",
                             show_sensors=False,
                             colors=dict(auditory="k",visual="r"),
                             truncate_yaxis=False)

evokeds_regressed_diff =  dict(diff=[mne.combine_evoked(epoch,[-1, 1]) for epoch in
                          zip(evokeds_regression["visual"],evokeds_regression["auditory"])])

fig = mne.viz.plot_compare_evokeds(evokeds_regressed_diff,
                             picks=ch,
                             invert_y=True, ci=0.95,
                             title="baseline correction based on regression",
                             show_sensors=False,legend=True,
                             truncate_yaxis=False)

evokeds_regressed_diff['diff'][0].plot_topomap(times=[0., 0.1, 0.2, 0.3, 0.4], ch_type='eeg')