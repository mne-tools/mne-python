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
#          Denis Engemann <denis.engemann@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.decoding import time_generalization
from mne.utils import create_slices
from functools import partial

data_path = sample.data_path()

###############################################################################
# Load and filter data, set up epochs
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
events_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
# read file
raw = mne.io.Raw(raw_fname, preload=True)
# pick MEG channels only
picks = mne.pick_types(raw.info, meg=True, exclude='bads')

# band pass filtering signals: time generalization is here applied with an evoked signals.
raw.filter(1, 30, method='iir')

# get events
events = mne.read_events(events_fname)
event_id  = {'AudL': 1, 'VisL': 3, 'AudR': 2, 'VisR': 4}
event_id_train = ['AudL', 'VisL']
event_id_generalize = ['AudR', 'VisR']
tmin, tmax = -0.1, 0.5

# Read epochs
decim = 3  # decimate to make the example faster to run
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
	                picks=picks, baseline=None, preload=True,
	                reject=dict(mag=5e-12), decim=decim)

# select training condition
epochs_list = [epochs[k] for k in event_id_train]
mne.epochs.equalize_epoch_counts(epochs_list) 

# select  generalization condition
epochs_list_generalize = [epochs[k] for k in event_id_generalize]
mne.epochs.equalize_epoch_counts(epochs_list_generalize) 

###############################################################################
# 1. classic decoding across time example
# 
# Compute Area Under the Curver (AUC) Receiver Operator Curve (ROC) score
# of time generalization. A perfect decoding would lead to AUCs of 1.
# Chance level is at 0.5.
# The default classifier is a linear SVM (C=1) after feature scaling.

# Setup time generalization parameters
# train_times = partial(create_slices,width=2) # width of the temporal window, in time sample
# test_times = partial(create_slices,width=2)

# results = time_generalization(epochs_list,
# 	                          epochs_list_generalize=epochs_list_generalize,
# 	                          train_times=train_times,
# 	                          clf=None, cv=5, scoring="roc_auc",
# 	                          shuffle=True, n_jobs=1)


# 2. Generalization across time
train_times = partial(create_slices,width=2) # width of the temporal window, in time sample

results = time_generalization(epochs_list,
	                          epochs_list_generalize=epochs_list_generalize,
	                          train_times=train_times,
	                          clf=None, cv=5, scoring="roc_auc",
	                          shuffle=True, n_jobs=1)

scores = results['scores']
scores_generalize = results['scores_generalize']

###############################################################################

# convert times to ms (note that for windows' width > 1, time 
#                      corresponds to the latest time sample)
train_times = 1e3 * epochs.times[list(s.stop for s in results['train_times'])]
test_times = 1e3 * epochs.times[list(s.stop for s in results['test_times'][0])]

# Now visualize
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax1, ax2, ax3, ax4 = ax.T.flatten()

ax1.plot(train_times, np.diag(scores), label="Classif. score")
ax1.axhline(0.5, color='k', linestyle='--', label="Chance level")
ax1.axvline(0, color='r', label='Stim onset')
ax1.set_ylim(0,1)
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('ROC classification score')
ax1.set_title('Decoding (%s vs. %s)' % tuple(event_id_train))

ax2.plot(train_times, np.diag(scores_generalize), label="Classif. score")
ax2.axhline(0.5, color='k', linestyle='--', label="Chance level")
ax2.axvline(0, color='r', label='Stim onset')
ax2.set_ylim(0,1)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('ROC classification score')
ax2.set_title('Decoding (%s vs. %s)' % tuple(event_id_generalize))
ax2.legend(loc='best')

im3 = ax3.imshow(scores, interpolation='nearest', origin='lower',
	extent=[test_times[0], test_times[-1], train_times[0], train_times[-1]],
	vmin=0., vmax=1.)
ax3.set_xlabel('Times Test (ms)')
ax3.set_ylabel('Times Train (ms)')
ax3.set_title('Time generalization (%s vs. %s)' % tuple(event_id_train))
ax3.axvline(0, color='k')
ax3.axhline(0, color='k')
plt.colorbar(im3, ax=ax3)

im4 = ax4.imshow(scores_generalize, interpolation='nearest', origin='lower',
	extent=[test_times[0], test_times[-1], train_times[0], train_times[-1]],
	vmin=0., vmax=1.)
ax4.set_xlabel('Times Test (ms)')
ax4.set_ylabel('Times Train (ms)')
ax4.set_title('Time generalization (%s vs. %s)' % tuple(event_id_generalize))
ax4.axvline(0, color='k')
ax4.axhline(0, color='k')
plt.colorbar(im4, ax=ax4)

mne.viz.tight_layout(fig=fig)
plt.show()
