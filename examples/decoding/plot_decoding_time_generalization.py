"""
========================================================
Decoding sensor space data with over-time generalization
========================================================

This example runs the analysis computed in:

Jean-Remi King, Alexandre Gramfort, Aaron Schurger, Lionel Naccache
and Stanislas Dehaene, "Two distinct dynamic modes subtend the detection of
unexpected sounds", PLOS ONE, 2013

The idea is to learn at one time instant and assess if the decoder
can predict accurately over time.
"""
print(__doc__)

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import spm_face
from mne.decoding import time_generalization

data_path = spm_face.data_path()

###############################################################################
# Load and filter data, set up epochs

raw_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces%d_3D_raw.fif'

raw = mne.io.Raw(raw_fname % 1, preload=True)  # Take first run
raw.append(mne.io.Raw(raw_fname % 2, preload=True))  # Take second run too

picks = mne.pick_types(raw.info, meg=True, exclude='bads')
raw.filter(1, 45, method='iir')

events = mne.find_events(raw, stim_channel='UPPT001')
event_id = {"faces": 1, "scrambled": 2}
tmin, tmax = -0.1, 0.5

# Set up pick list
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=True, eog=True,
                       ref_meg=False, exclude='bads')

# Read epochs
decim = 4  # decimate to make the example faster to run
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=None, preload=True,
                    reject=dict(mag=1.5e-12), decim=decim)

epochs_list = [epochs[k] for k in event_id]
mne.epochs.equalize_epoch_counts(epochs_list)

###############################################################################
# Run decoding

# Compute Area Under the Curver (AUC) Receiver Operator Curve (ROC) score
# of time generalization. A perfect decoding would lead to AUCs of 1.
# Chance level is at 0.5.
# The default classifier is a linear SVM (C=1) after feature scaling.
scores = time_generalization(epochs_list, clf=None, cv=5, scoring="roc_auc",
                             shuffle=True, n_jobs=2)

###############################################################################
# Now visualize
times = 1e3 * epochs.times  # convert times to ms

plt.figure()
plt.imshow(scores, interpolation='nearest', origin='lower',
           extent=[times[0], times[-1], times[0], times[-1]],
           vmin=0.1, vmax=0.9, cmap='RdBu_r')
plt.xlabel('Times Test (ms)')
plt.ylabel('Times Train (ms)')
plt.title('Time generalization (%s vs. %s)' % tuple(event_id.keys()))
plt.axvline(0, color='k')
plt.axhline(0, color='k')
plt.colorbar()

plt.figure()
plt.plot(times, np.diag(scores), label="Classif. score")
plt.axhline(0.5, color='k', linestyle='--', label="Chance level")
plt.axvline(0, color='r', label='stim onset')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('ROC classification score')
plt.title('Decoding (%s vs. %s)' % tuple(event_id.keys()))
plt.show()
