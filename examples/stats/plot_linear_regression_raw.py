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

Note. This example is based on new code which may still not be
memory-optimized. Be careful when working with a small computer.

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
raw_path = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D.ds'

# Use just the first 4 minutes to save memory
raw = mne.io.read_raw_ctf(raw_path).crop(0, 240, copy=False).load_data()

picks = mne.pick_types(raw.info, meg=True, exclude='bads')
raw.filter(1, 45, method='iir')

events = mne.find_events(raw, stim_channel='UPPT001')
event_id = dict(faces=1, scrambled=2)
tmin, tmax = -.1, .5

# regular epoching
picks = mne.pick_types(raw.info, meg=True)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, reject=None,
                    baseline=None, preload=True, picks=picks,
                    verbose=False, decim=4)

# rERF
evokeds = linear_regression_raw(raw, events=events, event_id=event_id,
                                reject=None, tmin=tmin, tmax=tmax,
                                decim=4, picks=picks)
# linear_regression_raw returns a dict of evokeds
# select conditions similarly to mne.Epochs objects

# plot both results
cond = "faces"
fig, (ax1, ax2) = plt.subplots(1, 2)
epochs[cond].average().plot(axes=ax1, show=False)
evokeds[cond].plot(axes=ax2, show=False)
ax1.set_title("Traditional averaging")
ax2.set_title("rERF")
plt.show()
