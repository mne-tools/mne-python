# -*- coding: utf-8 -*-
"""
.. _ex-baseline_regression:

====================================================================
Baseline correction using generalized linear mixed models
====================================================================

By including the baseline interval as a regressor in a generalized
linear model, we allow the data to determine how much baseline
correction is necessary for the specific epoch. This approach can
increase statistical power (Alday et al., 2019).
Alday et al. provided a script at osf: https://osf.io/85bjq
"""
# Authors: Carina Forster
#
# License: BSD-3-Clause

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample

# %%

#we load the example dataset for one subject

data_path = sample.data_path()

subjects_dir = data_path / 'subjects'
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'

#define epoch length

tmin, tmax = None, 0.500

# %% load the raw data for one subject

raw = mne.io.read_raw_fif(raw_fname, preload=True)

# we are only interested in EEG data

raw.pick_types(stim=True, eog=True, exclude=(), eeg=True)

# we filter the data (similar to Alday et al.)

lowpass = 40
highpass = 0.5 # Alday et al used 0.1

# baseline window

baseline_tmin = -0.2
baseline_tmax = 0

# we select one channel of interest

ch = "EEG 021"  # we combine both hemispheres, therefore a central electrode
# https://mne.tools/stable/auto_tutorials/intro/10_overview.html#sphx-glr-auto-tutorials-intro-10-overview-py

sfreq = raw.info['sfreq']

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

# very minimal cleaning

reject_criteria = dict(eeg=150e-6)
flat = dict(eeg=5e-6)

# %% We add events and epoch the data

events = mne.find_events(raw_filtered)

# merge visual and auditory events from both hemispheres
# can I do this in one line?

events = mne.merge_events(events, [1, 2], 1)
events = mne.merge_events(events, [3, 4], 2)

# we have 2 events of interest

event_id = {'auditory': 1, 'visual': 2}

epochs = mne.Epochs(raw_filtered,
                    events, event_id,
                    tmin=tmin, tmax=tmax,
                    reject=reject_criteria,
                    flat=flat,
                    baseline=None,
                    preload=True)

# kick out bad epochs

epochs.drop_bad()  # this operates in-place

epochs.interpolate_bads(reset_bads=True)  # avoids warning when fitting regression

epochs.equalize_event_counts()  # this operates in-place, 143 epochs per condition

del raw_filtered  # free up memory

# %% we average the epochs per condition and baseline correct the data

auditory_baseline = epochs[epochs.events[:, 2] == 1].copy().\
    apply_baseline((baseline_tmin, baseline_tmax)).average()

visual_baseline = epochs[epochs.events[:, 2] == 2].copy().\
    apply_baseline((baseline_tmin, baseline_tmax)).average()

# %% traditional baseline correction

fig = mne.viz.plot_compare_evokeds({"auditory": auditory_baseline, "visual": visual_baseline},
                                   picks=[ch],
                                   invert_y=True,
                                   title="traditional baseline correction",
                                   show_sensors=False, legend=True,
                                   colors=dict(auditory="k", visual="r"),
                                   truncate_yaxis=False)

# let's look at the topography of the contrast auditory - visual evoked response

diff = mne.combine_evoked([auditory_baseline, visual_baseline], [1, -1])

diff.plot_topomap(times=[0., 0.05, 0.1, 0.2, 0.3, 0.4], ch_type='eeg')

# maybe check a different channel?

diff.plot_topo()

# %% regression based baseline correction

baseline_epochs = epochs.copy().pick_channels([ch]).crop(baseline_tmin, baseline_tmax)
epoch_data = baseline_epochs.get_data()

baseline = epoch_data.mean(axis=-1).squeeze() * 1e6

# we convert to millivolt to make is easier for the regression to fit the data

# set up design matrix

design_matrix = np.stack([epochs.events[:, 2] == 1, epochs.events[:, 2] == 2,
                          baseline, baseline*(epochs.events[:, 2] == 2)]).T

# shape should be number of epochs per subject x number of regressor (so in that example 286 x 4)

# fit regression model

# epochs should have 286 epochs

# we fit the model for all channels

regmodel = mne.stats.linear_regression(epochs, design_matrix,
                                               names=["auditory", "visual",
                                                      "baseline",
                                                      "baseline:visual"])

regbaseline_aud = regmodel['auditory'].beta
regbaseline_vis = regmodel['visual'].beta

# %% plot regression output

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
