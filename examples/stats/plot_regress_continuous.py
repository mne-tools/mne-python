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

import matplotlib.pyplot as plt

import mne
from mne.datasets import spm_face
from mne.stats.regression import linear_regression_raw

# Preprocess data
data_path = spm_face.data_path()
# Load and filter data, set up epochs
raw_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D_raw.fif'

raw = mne.io.Raw(raw_fname, preload=True)  # Take first run

picks = mne.pick_types(raw.info, meg=True, exclude='bads')
raw.filter(1, 45, method='iir')

events = mne.find_events(raw, stim_channel='UPPT001')
event_id = dict(faces=1, scrambled=2)
tmin, tmax = -.1, .5

raw.pick_types(meg=True)

# regular epoching
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, reject=None,
                    baseline=None, preload=True, verbose=False, decim=4)

# rERF
evokeds = linear_regression_raw(raw, events=events, event_id=event_id,
                                reject=None, tmin=tmin, tmax=tmax, 
                                decim=4)

# plot both results
cond = "faces"
fig, (ax1, ax2) = plt.subplots(1, 2)
epochs[cond].average().plot(axes=ax1)
evokeds[cond].plot(axes=ax2)  # linear_regression_raw returns a dict of evokeds
ax1.set_title("Traditional averaging")
ax2.set_title("rERF")
