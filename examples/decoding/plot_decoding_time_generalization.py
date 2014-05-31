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
from mne.fixes import partial

data_path = sample.data_path()

#
# Load and filter data, set up epochs
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
events_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
# read file
raw = mne.io.Raw(raw_fname, preload=True)
# pick MEG channels only
picks = mne.pick_types(raw.info, meg=True, exclude='bads')

# band pass filtering signals: time generalization is here applied with an
# evoked signals.
raw.filter(1, 30, method='iir')

# get events
events = mne.read_events(events_fname)
event_id = {'AudL': 1, 'VisL': 3, 'AudR': 2, 'VisR': 4}
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
epochs_list_gen = [epochs[k] for k in event_id_gen]
mne.epochs.equalize_epoch_counts(epochs_list_gen)

# #############################################################################
# -----------------------------
# 1. Classic decoding over time
# -----------------------------
# In this scenario, each classifier is trained and tested at a single time
# point. This results in the classic decoding score across time
#
# 1.1. Setup sliding window parameters
train_slices = partial(create_slices, width=2)

# 1.2. Run decoding
# Compute Area Under the Curver (AUC) Receiver Operator Curve (ROC) score
# of time generalization. A perfect decoding would lead to AUCs of 1.
# Chance level is at 0.5.
# The default classifier is a linear SVM (C=1) after feature scaling.
results = time_generalization(epochs_list, train_slices=train_slices,
                              generalization='diagonal',
                              clf=None, cv=5, scoring="roc_auc",
                              shuffle=True, n_jobs=1)
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
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('ROC classification score')
    ax.set_title('Decoding (%s vs. %s)' % tuple(event_id))
    ax.legend(loc='best')

fig, ax = plt.subplots(1, 1)
plot_decode(ax, scores, train_times, event_id_train)
mne.viz.tight_layout(fig=fig)
plt.show()


# --------------------------------------------------------------
# 2. Generalization over time + generalization across conditions
# --------------------------------------------------------------
# In this scenario, each classifier is trained in a particular time point, and
# subsquently tested on its ability to generalize across other time points.
#
# Note that each classifier can also be tested on its ability to generalize to
# a second dataset. Specifically here, the classifiers are trained on
# epochs_list (left audio versus left visual), and generalized to
# epochs_list_gen (right audio versus right visual)

# The sliding window parameters can be modified with create_slice(). Here,
# the classifiers are non-overlapping and use 2 consecutive time samples.
train_slices = partial(create_slices, width=2, across_step=2)

# Run main script
results = time_generalization(epochs_list, train_slices=train_slices,
                              epochs_list_gen=epochs_list_gen,
                              generalization='cardinal', n_jobs=1)
train_times = 1e3 * results['train_times']
test_times = 1e3 * results['test_times']
scores = results['scores']
scores_gen = results['scores_gen']

# Vizualize


def plot_time_gen(ax, plt, scores, train_time, test_time, event_id):
    im = ax.imshow(scores, interpolation='nearest', origin='lower',
                   extent=[test_times[0], test_times[-1],
                   train_times[0], train_times[-1]],
                   vmin=0., vmax=1.)
    ax.set_xlabel('Times Test (ms)')
    ax.set_ylabel('Times Train (ms)')
    ax.set_title('Time generalization (%s vs. %s)' % tuple(event_id))
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    plt.colorbar(im, ax=ax)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax1, ax2, ax3, ax4 = ax.T.flatten()
# plot time generalization for (cross-validated) training set
plot_decode(ax1, scores, train_times, event_id_train)
plot_time_gen(ax3, plt, scores, train_times, test_times, event_id_train)
# plot time generalization for generalization dataset
plot_decode(ax2, scores_gen, train_times, event_id_train)
plot_time_gen(ax4, plt, scores_gen, train_times, test_times, event_id_gen)
mne.viz.tight_layout(fig=fig)
plt.show()


# -----------------------------------
# 3. Advanced temporal generalization
# -----------------------------------
# Other temporal generalization scenario can be implemented by modifying the
# training and testing slices.
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
ax1, ax2, ax3 = ax.T.flatten()

# Generalize around diagonal
test_slices = create_slices(5, start=-5)  # generalize +/- around train sample
results = time_generalization(epochs_list, test_slices=test_slices,
                              generalization='diagonal', n_jobs=1)
train_times = 1e3 * results['train_times']
scores = results['scores']
plot_time_gen(ax1, plt, scores, train_times, train_times, event_id_train)

# Generalize between specific time points
train_slices = create_slices(15, start=10)
test_slices = [create_slices(30) for _ in range(len(train_slices))]
results = time_generalization(epochs_list, train_slices=train_slices,
                              test_slices=test_slices,
                              generalization='cardinal', n_jobs=1)
train_times = 1e3 * results['train_times']
test_times = 1e3 * results['test_times']
scores = results['scores']
plot_time_gen(ax2, plt, scores, train_times, test_times, event_id_train)

# Generalize at specific time points
train_slices = create_slices(31, start=0)
test_slices = [create_slices(15, start=10) for _ in range(len(train_slices))]
results = time_generalization(epochs_list, test_slices=test_slices,
                              train_slices=train_slices,
                              generalization='cardinal', n_jobs=1)
train_times = 1e3 * results['train_times']
test_times = 1e3 * results['test_times']
scores = results['scores']
plot_time_gen(ax3, plt, scores, train_times, test_times, event_id_train)

mne.viz.tight_layout(fig=fig)
plt.show()
