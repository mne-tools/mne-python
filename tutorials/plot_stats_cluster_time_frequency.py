"""
.. _tut_stats_cluster_sensor_2samp_tfr:

=========================================================================
Non-parametric between conditions cluster statistic on single trial power
=========================================================================

This script shows how to compare clusters in time-frequency
power estimates between conditions. It uses a non-parametric
statistical procedure based on permutations and cluster
level statistics.

The procedure consists in:

  - extracting epochs for 2 conditions
  - compute single trial power estimates
  - baseline line correct the power estimates (power ratios)
  - compute stats to see if the power estimates are significantly different
    between conditions.

"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.time_frequency import single_trial_power
from mne.stats import permutation_cluster_test
from mne.datasets import sample

print(__doc__)

###############################################################################
# Set parameters
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
event_id = 1
tmin = -0.2
tmax = 0.5

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

include = []
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                       stim=False, include=include, exclude='bads')

ch_name = raw.info['ch_names'][picks[0]]

# Load condition 1
reject = dict(grad=4000e-13, eog=150e-6)
event_id = 1
epochs_condition_1 = mne.Epochs(raw, events, event_id, tmin, tmax,
                                picks=picks, baseline=(None, 0),
                                reject=reject)
data_condition_1 = epochs_condition_1.get_data()  # as 3D matrix
data_condition_1 *= 1e13  # change unit to fT / cm

# Load condition 2
event_id = 2
epochs_condition_2 = mne.Epochs(raw, events, event_id, tmin, tmax,
                                picks=picks, baseline=(None, 0),
                                reject=reject)
data_condition_2 = epochs_condition_2.get_data()  # as 3D matrix
data_condition_2 *= 1e13  # change unit to fT / cm

# Take only one channel
data_condition_1 = data_condition_1[:, 97:98, :]
data_condition_2 = data_condition_2[:, 97:98, :]

# Time vector
times = 1e3 * epochs_condition_1.times  # change unit to ms

###############################################################################
# Factor to downsample the temporal dimension of the PSD computed by
# single_trial_power.  Decimation occurs after frequency decomposition and can
# be used to reduce memory usage (and possibly comptuational time of downstream
# operations such as nonparametric statistics) if you don't need high
# spectrotemporal resolution.
decim = 2
frequencies = np.arange(7, 30, 3)  # define frequencies of interest
sfreq = raw.info['sfreq']  # sampling in Hz
n_cycles = 1.5

epochs_power_1 = single_trial_power(data_condition_1, sfreq=sfreq,
                                    frequencies=frequencies,
                                    n_cycles=n_cycles, decim=decim)

epochs_power_2 = single_trial_power(data_condition_2, sfreq=sfreq,
                                    frequencies=frequencies,
                                    n_cycles=n_cycles, decim=decim)

epochs_power_1 = epochs_power_1[:, 0, :, :]  # only 1 channel to get 3D matrix
epochs_power_2 = epochs_power_2[:, 0, :, :]  # only 1 channel to get 3D matrix

###############################################################################
# Compute ratio with baseline power (be sure to correct time vector with
# decimation factor)
baseline_mask = times[::decim] < 0
epochs_baseline_1 = np.mean(epochs_power_1[:, :, baseline_mask], axis=2)
epochs_power_1 /= epochs_baseline_1[..., np.newaxis]
epochs_baseline_2 = np.mean(epochs_power_2[:, :, baseline_mask], axis=2)
epochs_power_2 /= epochs_baseline_2[..., np.newaxis]

###############################################################################
# Compute statistic
threshold = 6.0
T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_test([epochs_power_1, epochs_power_2],
                             n_permutations=100, threshold=threshold, tail=0)

###############################################################################
# View time-frequency plots
plt.clf()
plt.subplots_adjust(0.12, 0.08, 0.96, 0.94, 0.2, 0.43)
plt.subplot(2, 1, 1)
evoked_contrast = np.mean(data_condition_1, 0) - np.mean(data_condition_2, 0)
plt.plot(times, evoked_contrast.T)
plt.title('Contrast of evoked response (%s)' % ch_name)
plt.xlabel('time (ms)')
plt.ylabel('Magnetic Field (fT/cm)')
plt.xlim(times[0], times[-1])
plt.ylim(-100, 200)

plt.subplot(2, 1, 2)

# Create new stats image with only significant clusters
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(clusters, cluster_p_values):
    if p_val <= 0.05:
        T_obs_plot[c] = T_obs[c]

plt.imshow(T_obs,
           extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
           aspect='auto', origin='lower', cmap='RdBu_r')
plt.imshow(T_obs_plot,
           extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
           aspect='auto', origin='lower', cmap='RdBu_r')

plt.xlabel('time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title('Induced power (%s)' % ch_name)
plt.show()
