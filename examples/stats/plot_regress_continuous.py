"""
=========================================
Regression on continuous data (rER[P/F])
=========================================

This demonstrates how rERPs/regressing the continuous data is a
generalisation of traditional averaging. If all preprocessing steps
are the same and if no overlap between epochs exists and if all
predictors are binary, regression is virtually identical to traditional
averaging.

If overlap exists and/or predictors are continuous, traditional averaging
is inapplicable, but regression can be separate effects, including those of
continuous predictors.

References
-------------
See Smith, N. J., & Kutas, M. (2015). Regression-based estimation of ERP
waveforms: II. Non-linear effects, overlap correction, and practical
considerations. Psychophysiology, 52(2), 169-189.

Authors: Jona Sassenhagen <jona.sassenhagen@gmail.de>
         Marijn van Vliet <w.m.vanvliet@gmail.com>
        Denis A. Engemann <denis.engemann@gmail.com>

License: BSD (3-clause)
"""

import numpy as np

import mne
from mne.datasets import spm_face
from mne.stats.regression import regress_continuous

mne.set_log_level(False)

# Preprocess data
data_path = spm_face.data_path()
# Load and filter data, set up epochs
raw_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces%d_3D_raw.fif'

raw = mne.io.Raw(raw_fname % 1, preload=True)  # Take first run

picks = mne.pick_types(raw.info, meg=True, exclude='bads')
raw.filter(1, 45, method='iir')

events = mne.find_events(raw, stim_channel='UPPT001')
event_id = {"faces": 1, "scrambled": 2}
tmin, tmax = -.1, .5

raw.pick_types(meg=True)

# regular epoching
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(-.1, .2), preload=True, verbose=False)

# rERF
evoked_dict = regress_continuous(raw, events=events, event_id=event_id,
                                 reject=False, tmin=tmin, tmax=tmax)

# plot both results
cond = "faces"
print("traditional ERF:")
epochs[cond].average().plot()
print("rERF:")
evoked_dict[cond].plot()

# check if results are virtually identical
# (as they should be, in the case of no overlap)
print("Are the two methods virtually identical? ",
      np.allclose(epochs[cond].average().data*1e+15,
                  evoked_dict[cond].data*1e+15))
