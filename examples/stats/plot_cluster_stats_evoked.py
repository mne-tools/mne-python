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
raw = fiff.Raw(raw_fname)
events = mne.read_events(event_fname)

channel = 'MEG 1332'  # include only this channel in analysis
include = [channel]

###############################################################################
# Read epochs for the channel of interest
picks = fiff.pick_types(raw.info, meg=False, eog=True, include=include)
event_id = 1
reject = dict(grad=4000e-13, eog=150e-6)
epochs1 = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                     baseline=(None, 0), reject=reject)
condition1 = epochs1.get_data()  # as 3D matrix

event_id = 2
epochs2 = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                     baseline=(None, 0), reject=reject)
condition2 = epochs2.get_data()  # as 3D matrix

condition1 = condition1[:, 0, :]  # take only one channel to get a 2D array
condition2 = condition2[:, 0, :]  # take only one channel to get a 2D array

###############################################################################
# Compute statistic
threshold = 6.0
T_obs, clusters, cluster_p_values, H0 = \
                permutation_cluster_test([condition1, condition2],
                            n_permutations=1000, threshold=threshold, tail=1,
                            n_jobs=2)

###############################################################################
# Plot
times = epochs1.times
import pylab as pl
pl.close('all')
pl.subplot(211)
pl.title('Channel : ' + channel)
pl.plot(times, condition1.mean(axis=0) - condition2.mean(axis=0),
        label="ERF Contrast (Event 1 - Event 2)")
pl.ylabel("MEG (T / m)")
pl.legend()
pl.subplot(212)
for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        h = pl.axvspan(times[c.start], times[c.stop - 1], color='r', alpha=0.3)
    else:
        pl.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
                   alpha=0.3)
hf = pl.plot(times, T_obs, 'g')
pl.legend((h, ), ('cluster p-value < 0.05', ))
pl.xlabel("time (ms)")
pl.ylabel("f-values")
pl.show()
