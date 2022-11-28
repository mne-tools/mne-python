# -*- coding: utf-8 -*-
"""
.. _ex-perm-test:

=================================
Permutation T-test on sensor data
=================================

One tests if the signal significantly deviates from 0
during a fixed time window of interest. Here computation
is performed on MNE sample dataset between 40 and 60 ms.

"""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

# %%

import numpy as np

import mne
from mne import io
from mne.stats import permutation_t_test
from mne.datasets import sample

print(__doc__)

# %%
# Set parameters
data_path = sample.data_path()
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'
event_fname = meg_path / 'sample_audvis_filt-0-40_raw-eve.fif'
event_id = 1
tmin = -0.2
tmax = 0.5

#   Setup for reading the raw data
raw = io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

# pick MEG Gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=True,
                       exclude='bads')
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6))
data = epochs.get_data()
times = epochs.times

temporal_mask = np.logical_and(0.04 <= times, times <= 0.06)
data = np.mean(data[:, :, temporal_mask], axis=2)

n_permutations = 50000
T0, p_values, H0 = permutation_t_test(data, n_permutations, n_jobs=None)

significant_sensors = picks[p_values <= 0.05]
significant_sensors_names = [raw.ch_names[k] for k in significant_sensors]

print("Number of significant sensors : %d" % len(significant_sensors))
print("Sensors names : %s" % significant_sensors_names)

# %%
# View location of significantly active sensors

evoked = mne.EvokedArray(-np.log10(p_values)[:, np.newaxis],
                         epochs.info, tmin=0.)

# Extract mask and indices of active sensors in the layout
stats_picks = mne.pick_channels(evoked.ch_names, significant_sensors_names)
mask = p_values[:, np.newaxis] <= 0.05

evoked.plot_topomap(ch_type='grad', times=[0], scalings=1,
                    time_format=None, cmap='Reds', vlim=(0., np.max),
                    units='-log10(p)', cbar_fmt='-%0.1f', mask=mask,
                    size=3, show_names=lambda x: x[4:] + ' ' * 20,
                    time_unit='s')
