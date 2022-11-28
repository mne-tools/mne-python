# -*- coding: utf-8 -*-
"""
=======================================
Reduce EOG artifacts through regression
=======================================

Reduce artifacts by regressing the EOG channels onto the rest of the channels
and then subtracting the EOG signal.

This is a quick example to show the most basic application of the technique.
See the :ref:`tutorial <tut-artifact-regression>` for a more thorough
explanation that demonstrates more advanced approaches.
"""

# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD (3-clause)

# %%
# Import packages and load data
# -----------------------------
#
# We begin as always by importing the necessary Python modules and loading some
# data, in this case the :ref:`MNE sample dataset <sample-dataset>`.

import mne
from mne.datasets import sample
from mne.preprocessing import EOGRegression
from matplotlib import pyplot as plt

print(__doc__)

data_path = sample.data_path()
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'

# Read raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
events = mne.find_events(raw, 'STI 014')

# Highpass filter to eliminate slow drifts
raw.filter(0.3, None, picks='all')

# %%
# Perform regression and remove EOG
# ---------------------------------

# Fit the regression model
weights = EOGRegression().fit(raw)
raw_clean = weights.apply(raw, copy=True)

# Show the filter weights in a topomap
weights.plot()

# %%
# Before/after comparison
# -----------------------
# Let's compare the signal before and after cleaning with EOG regression. This
# is best visualized by extracting epochs and plotting the evoked potential.

tmin, tmax = -0.1, 0.5
event_id = {'visual/left': 3, 'visual/right': 4}
evoked_before = mne.Epochs(raw, events, event_id, tmin, tmax,
                           baseline=(tmin, 0)).average()
evoked_after = mne.Epochs(raw_clean, events, event_id, tmin, tmax,
                          baseline=(tmin, 0)).average()

# Create epochs after EOG correction
epochs_after = mne.Epochs(raw_clean, events, event_id, tmin, tmax,
                          baseline=(tmin, 0))
evoked_after = epochs_after.average()

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 7),
                       sharex=True, sharey='row')
evoked_before.plot(axes=ax[:, 0], spatial_colors=True)
evoked_after.plot(axes=ax[:, 1], spatial_colors=True)
fig.subplots_adjust(top=0.905, bottom=0.09, left=0.08, right=0.975,
                    hspace=0.325, wspace=0.145)
fig.suptitle('Before --> After')
