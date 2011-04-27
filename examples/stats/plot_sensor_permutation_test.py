"""
=================================
Permutation T-test on sensor data
=================================

One tests if the signal significantly deviates from 0
during a fixed time window of interest. Here computation
is performed on MNE sample dataset between 40 and 60 ms.

"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np

import mne
from mne import fiff
from mne.stats import permutation_t_test
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

#   Set up pick list: MEG + STI 014 - bad channels (modify to your needs)
include = []  # or stim channel ['STI 014']
exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more

# pick MEG Magnetometers
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=True,
                                            include=include, exclude=exclude)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6))
data = epochs.get_data()
times = epochs.times

temporal_mask = np.logical_and(0.04 <= times, times <= 0.06)
data = np.squeeze(np.mean(data[:, :, temporal_mask], axis=2))

n_permutations = 50000
T0, p_values, H0 = permutation_t_test(data, n_permutations, n_jobs=2)

significant_sensors = picks[p_values <= 0.05]
significant_sensors_names = [raw.info['ch_names'][k]
                              for k in significant_sensors]

print "Number of significant sensors : %d" % len(significant_sensors)
print "Sensors names : %s" % significant_sensors_names

###############################################################################
# View location of significantly active sensors
import pylab as pl

# load sensor layout
from mne.layouts import Layout
layout = Layout('Vectorview-grad')

# Extract mask and indices of active sensors in layout
idx_of_sensors = [layout.names.index(name)
                    for name in significant_sensors_names
                    if name in layout.names]
mask_significant_sensors = np.zeros(len(layout.pos), dtype=np.bool)
mask_significant_sensors[idx_of_sensors] = True
mask_non_significant_sensors = mask_significant_sensors == False

# plot it
pl.figure(facecolor='k')
pl.axis('off')
pl.axis('tight')
pl.scatter(layout.pos[mask_significant_sensors, 0],
           layout.pos[mask_significant_sensors, 1], s=50, c='r')
pl.scatter(layout.pos[mask_non_significant_sensors, 0],
           layout.pos[mask_non_significant_sensors, 1], c='w')
title = 'MNE sample data (Left auditory between 40 and 60 ms)'
pl.figtext(0.03, 0.93, title, color='w', fontsize=18)
pl.show()
pl.show()
