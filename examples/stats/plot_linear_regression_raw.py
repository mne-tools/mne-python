"""
========================================
Regression on continuous data (rER[P/F])
========================================

This demonstrates how rERPs/regressing the continuous data is a
generalisation of traditional averaging. If all preprocessing steps
are the same and if no overlap between epochs exists and if all
predictors are binary, regression is virtually identical to traditional
averaging.
If overlap exists and/or predictors are continuous, traditional averaging
is inapplicable, but regression can estimate, including those of
continuous predictors.

rERPs are described in:
Smith, N. J., & Kutas, M. (2015). Regression-based estimation of ERP
waveforms: II. Non-linear effects, overlap correction, and practical
considerations. Psychophysiology, 52(2), 169-189.
"""
# Authors: Jona Sassenhagen <jona.sassenhagen@gmail.de>
#
# License: BSD (3-clause)

%matplotlib inline
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.stats.regression import linear_regression_raw

# Preprocess data
data_path = sample.data_path()
# Load and filter data, set up epochs
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

# Use just the first 4 minutes to save memory
raw = mne.io.read_raw_fif(raw_fname, preload=True).filter(
  1, None, method='iir').apply_proj()

picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')

events = mne.find_events(raw)
event_id = {'Aud/L': 1, 'Aud/R': 2}
tmin, tmax = -.1, .5

# regular epoching
picks = mne.pick_types(raw.info, meg=True)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, reject=None,
                    baseline=None, preload=True, picks=picks,
                    verbose=False)

# rERF
evokeds = linear_regression_raw(raw, events=events, event_id=event_id,
                                reject=None, tmin=tmin, tmax=tmax,
                                picks=picks)
# linear_regression_raw returns a dict of evokeds
# select conditions similarly to mne.Epochs objects

# plot both results, and their difference
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
