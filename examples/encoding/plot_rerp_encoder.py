"""
======================================================
Regression on continuous event-related data (rER[P/F])
======================================================

This demonstrates how rER[P/F]s - regressing the continuous data - is a
generalisation of traditional averaging. If all preprocessing steps
are the same, no overlap between epochs exists, and all predictors are binary,
regression is virtually identical to traditional averaging.

If overlap exists and/or predictors are continuous, traditional averaging
is inapplicable, but regression can still estimate effects.

rER[P/F]s are described in:

    Smith, N. J., & Kutas, M. (2015). Regression-based estimation of ERP
    waveforms: II. Non-linear effects, overlap correction, and practical
    considerations. Psychophysiology, 52(2), 169-189.
"""
# Authors: Jona Sassenhagen <jona.sassenhagen@gmail.de>
#          Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.encoding.model import EventRelatedRegressor

# Load and preprocess data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True).pick_types(
    meg='grad', stim=True, eeg=False).filter(1, None, method='iir')

# Set up events
events = mne.find_events(raw)
event_id = {'Aud/L': 1, 'Aud/R': 2}
tmin, tmax = -.1, .5

# Regular epoching
picks = mne.pick_types(raw.info, meg=True)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, reject=None,
                    baseline=None, preload=True, verbose=False)

# rERF
rerp = EventRelatedRegressor(raw, events, est='cholesky', event_id=event_id,
                             tmin=tmin, tmax=tmax, remove_outliers=True)
rerp.fit()

# The EventRelatedRegressor object can returns a dict of evokeds
evokeds = rerp.to_evoked()

# Plot both results, and their difference
cond = "Aud/L"
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
params = dict(spatial_colors=True, show=False, ylim=dict(grad=(-200, 200)))
epochs[cond].average().plot(axes=ax1, **params)
evokeds[cond].plot(axes=ax2, **params)
(evokeds[cond] - epochs[cond].average()).plot(axes=ax3, **params)
ax1.set_title("Traditional averaging")
ax2.set_title("rERF")
ax3.set_title("Difference")
plt.show()
