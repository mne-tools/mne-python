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

More 'advanced' examples can be found at:
https://gist.github.com/kingjr/7b4aa44438781e138fcc
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
event_id_select = ['AudL', 'VisL']  # only classify 2 differents set of trials
tmin, tmax = -0.1, 0.5

# Read epochs
decim = 3  # decimate to make the example faster to run
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=None, preload=True,
                    reject=dict(mag=5e-12), decim=decim)

epochs_list = [epochs[k] for k in event_id_select]
mne.epochs.equalize_epoch_counts(epochs_list)

# --------------------------------------------------------------
# 2. Generalization over time + generalization across conditions
# --------------------------------------------------------------
# Each classifier is trained in a particular time point, and  subsquently
# tested on its ability to generalize across other time points.
#
# The sliding window parameters can be modified with create_slice(). Here,
# the classifiers are non-overlapping and use 2 consecutive time samples.
train_slices = partial(create_slices, step=2, length=2)

# Run main script
results = time_generalization(epochs_list, train_slices=train_slices,
                              n_jobs=1)
# Gather results
train_times = 1e3 * results['train_times']
test_times = 1e3 * results['test_times']
scores = results['scores']

# Vizualize
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax1, ax2 = ax.T.flatten()

# Plot diagonal of temporal generalization matrix (i.e. equivalent of decoding
# across time)
ax1.plot(train_times, np.diag(scores), label="Classif. score")
ax1.axhline(0.5, color='k', linestyle='--', label="Chance level")
ax1.axvline(0, color='r', label='Stim onset')
ax1.set_ylim(0, 1)
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('ROC classification score')
ax1.set_title('Decoding (%s vs. %s)' % tuple(event_id_select))
ax1.legend(loc='best')

# Plot time generalization for (cross-validated) training set
im = ax2.imshow(scores, interpolation='nearest', origin='lower',
                extent=[test_times[0], test_times[-1],
                        train_times[0], train_times[-1]],
                vmin=0., vmax=1.)
ax2.set_xlabel('Times Test (ms)')
ax2.set_ylabel('Times Train (ms)')
ax2.set_title('Time generalization (%s vs. %s)' % tuple(event_id_select))
ax2.axvline(0, color='k')
ax2.axhline(0, color='k')
plt.colorbar(im, ax=ax2)

mne.viz.tight_layout(fig=fig)
plt.show()
