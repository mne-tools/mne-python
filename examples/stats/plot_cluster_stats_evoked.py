"""
=======================================================
Permutation F-test on sensor data with 1D cluster level
=======================================================

One tests if the evoked response is significantly different
between conditions. Multiple comparison problem is adressed
with cluster level permutation test.

"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np

import mne
from mne import fiff
from mne.stats import permutation_cluster_test
from mne.datasets import sample

###############################################################################
# Set parameters
data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_id = 1
tmin = -0.2
tmax = 0.5

#   Setup for reading the raw data
raw = fiff.setup_read_raw(raw_fname)
events = mne.read_events(event_fname)

channel = 'MEG 1332'
include = [channel]

###############################################################################
# Read epochs for the channel of interest
picks = fiff.pick_types(raw['info'], meg=False, include=include)
event_id = 1
data1, times, channel_names = mne.read_epochs(raw, events, event_id,
                            tmin, tmax, picks=picks, baseline=(None, 0))
condition1 = np.squeeze(np.array([d['epoch'] for d in data1])) # as 3D matrix

event_id = 2
data2, times, channel_names = mne.read_epochs(raw, events, event_id,
                            tmin, tmax, picks=picks, baseline=(None, 0))
condition2 = np.squeeze(np.array([d['epoch'] for d in data2])) # as 3D matrix

###############################################################################
# Compute statistic
threshold = 6.0
T_obs, clusters, cluster_p_values, H0 = \
                permutation_cluster_test([condition1, condition2],
                            n_permutations=1000, threshold=threshold, tail=1)

###############################################################################
# Plot
import pylab as pl
pl.close('all')
pl.subplot(211)
pl.title('Channel : ' + channel)
pl.plot(times, condition1.mean(axis=0) - condition2.mean(axis=0),
        label="ERF Contrast (Event 1 - Event 2)")
pl.ylabel("MEG (T / m)")
pl.legend()
pl.subplot(212)
for i_c, (start, stop) in enumerate(clusters):
    if cluster_p_values[i_c] <= 0.05:
        h = pl.axvspan(times[start], times[stop-1], color='r', alpha=0.3)
    else:
        pl.axvspan(times[start], times[stop-1], color=(0.3, 0.3, 0.3),
                   alpha=0.3)
hf = pl.plot(times, T_obs, 'g')
pl.legend((h, ), ('cluster p-value < 0.05', ))
pl.xlabel("time (ms)")
pl.ylabel("f-values")
pl.show()
