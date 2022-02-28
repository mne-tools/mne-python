# -*- coding: utf-8 -*-
"""
.. _ex-linear-regression-raw:

========================================
Regression on continuous data (rER[P/F])
========================================

This demonstrates how rER[P/F]s - regressing the continuous data - is a
generalisation of traditional averaging. If all preprocessing steps
are the same, no overlap between epochs exists, and if all
predictors are binary, regression is virtually identical to traditional
averaging.
If overlap exists and/or predictors are continuous, traditional averaging
is inapplicable, but regression can estimate effects, including those of
continuous predictors.

rERPs are described in:
Smith, N. J., & Kutas, M. (2015). Regression-based estimation of ERP
waveforms: II. Non-linear effects, overlap correction, and practical
considerations. Psychophysiology, 52(2), 169-189.
"""
# Authors: Jona Sassenhagen <jona.sassenhagen@gmail.de>
#
# License: BSD-3-Clause

# %%

import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.stats.regression import linear_regression_raw

# Load and preprocess data
data_path = sample.data_path()
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname)
raw.pick_types(meg='grad', stim=True, eeg=False).load_data()
raw.filter(1, None, fir_design='firwin')  # high-pass

# Set up events
events = mne.find_events(raw)
event_id = {'Aud/L': 1, 'Aud/R': 2}
tmin, tmax = -.1, .5

# regular epoching
picks = mne.pick_types(raw.info, meg=True)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, reject=None,
                    baseline=None, preload=True, verbose=False)

# rERF
evokeds = linear_regression_raw(raw, events=events, event_id=event_id,
                                reject=None, tmin=tmin, tmax=tmax)
# linear_regression_raw returns a dict of evokeds
# select conditions similarly to mne.Epochs objects

# plot both results, and their difference
cond = "Aud/L"
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
params = dict(spatial_colors=True, show=False, ylim=dict(grad=(-200, 200)),
              time_unit='s')
epochs[cond].average().plot(axes=ax1, **params)
evokeds[cond].plot(axes=ax2, **params)
contrast = mne.combine_evoked([evokeds[cond], epochs[cond].average()],
                              weights=[1, -1])
contrast.plot(axes=ax3, **params)
ax1.set_title("Traditional averaging")
ax2.set_title("rERF")
ax3.set_title("Difference")
plt.show()
