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
from functools import partial
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from mne.decoding import time_generalization
from mne.utils import create_slices

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
event_id_gen = ['AudR', 'VisR']
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
epochs_list_generalize = [epochs[k] for k in event_id_gen]
mne.epochs.equalize_epoch_counts(epochs_list_generalize) 

###############################################################################
# Compute Area Under the Curver (AUC) Receiver Operator Curve (ROC) score
# of time generalization. A perfect decoding would lead to AUCs of 1.
# Chance level is at 0.5.
# The default classifier is a linear SVM (C=1) after feature scaling.

# Setup time generalization parameters
# 1. Classic decoding over time
# 1.1. Setup sliding window parameters
train_slices = partial(create_slices, width=2)
# 1.2. Run decoding
results = time_generalization(epochs_list,
							  train_slices=train_slices,
	                          generalization='diagonal',
	                          clf=None, cv=5, scoring="roc_auc",
	                          shuffle=True, n_jobs=5)
# 1.3 Retrieve results
scores = results['scores']
# Note that time corresponds to the *start* of the classifier. Larger window 
# width imply that  later time points will be used by each classifier.
train_times = 1e3 * results['train_times']

# 1.4 Visualize
# Vizualize
def plot_decode(ax, scores, time, event_id):
	ax.plot(time, np.diag(scores), label="Classif. score")
	ax.axhline(0.5, color='k', linestyle='--', label="Chance level")
	ax.axvline(0, color='r', label='Stim onset')
	ax.set_ylim(0,1)
	ax.set_xlabel('Time (ms)')
	ax.set_ylabel('ROC classification score')
	ax.set_title('Decoding (%s vs. %s)' % tuple(event_id))
	ax.legend(loc='best')

def plot_time_gen(ax, plt, scores, train_time, test_time, event_id):
	im = ax.imshow(scores, interpolation='nearest', origin='lower',
	extent=[test_times[0], test_times[-1], train_times[0], train_times[-1]],
	vmin=0., vmax=1.)
	ax.set_xlabel('Times Test (ms)')
	ax.set_ylabel('Times Train (ms)')
	ax.set_title('Time generalization (%s vs. %s)' % tuple(event_id))
	ax.axvline(0, color='k')
	ax.axhline(0, color='k')
	plt.colorbar(im, ax=ax)

fig, ax = plt.subplots(1, 1)
plot_decode(ax, scores, train_times, event_id_train)
mne.viz.tight_layout(fig=fig)
plt.show()


# 2. Generalization over time + generalization across conditions
train_slices = partial(create_slices, width=2)
results = time_generalization(epochs_list,
							  train_slices=train_slices,
							  epochs_list_generalize=epochs_list_generalize,
	                          generalization='cardinal', n_jobs=5)
train_times = 1e3 * results['train_times']
test_times =  1e3 * results['test_times']
scores = results['scores']
scores_gen = results['scores_generalize']
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax1, ax2, ax3, ax4 = ax.T.flatten()
plot_decode(ax1, scores, train_times, event_id_train)
plot_decode(ax2, scores_gen, train_times, event_id_train)
plot_time_gen(ax3, plt, scores, train_times, test_times, event_id_train)
plot_time_gen(ax4, plt, scores_gen, train_times, test_times, event_id_gen)
mne.viz.tight_layout(fig=fig)
plt.show()

# 3.Generalization over time: around diagonal or asymetrical generalization
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
ax1, ax2, ax3 = ax.T.flatten()

# Around diagonal
test_slices = create_slices(5, start=-5)
results = time_generalization(epochs_list, test_slices=test_slices,
	                          generalization='diagonal', n_jobs=5)
train_times = 1e3 * results['train_times']
scores = results['scores']
plot_time_gen(ax1, plt, scores, train_times, train_times, event_id_train)

# Between specific time points
train_slices = create_slices(15,start=10)
test_slices = [create_slices(30)] * len(train_slices)
results = time_generalization(epochs_list, train_slices=train_slices,
							  test_slices=test_slices,
	                          generalization='cardinal', n_jobs=5)
train_times = 1e3 * results['train_times']
test_times =  1e3 * results['test_times']
scores = results['scores']
plot_time_gen(ax2, plt, scores, train_times, test_times, event_id_train)

# At specific time points
train_slices = create_slices(31,start=0)
test_slices = [create_slices(15, start=10)] * len(train_slices)
results = time_generalization(epochs_list, 
							  test_slices=test_slices, 
							  train_slices=train_slices,
	                          generalization='cardinal', n_jobs=5)
train_times = 1e3 * results['train_times']
test_times =  1e3 * results['test_times']
scores = results['scores']
plot_time_gen(ax3, plt, scores, train_times, test_times, event_id_train)

mne.viz.tight_layout(fig=fig)
plt.show()