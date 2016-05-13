"""
.. _tut_stats_cluster_sensor_1samp_tfr:

===============================================================
Non-parametric 1 sample cluster statistic on single trial power
===============================================================

This script shows how to estimate significant clusters
in time-frequency power estimates. It uses a non-parametric
statistical procedure based on permutations and cluster
level statistics.

The procedure consists in:

  - extracting epochs
  - compute single trial power estimates
  - baseline line correct the power estimates (power ratios)
  - compute stats to see if ratio deviates from 1.

"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.time_frequency import single_trial_power
from mne.stats import permutation_cluster_1samp_test
from mne.datasets import sample

print(__doc__)

###############################################################################
# Set parameters
# --------------
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_id = 1
tmin = -0.3
tmax = 0.6

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname)
events = mne.find_events(raw, stim_channel='STI 014')

include = []
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                       stim=False, include=include, exclude='bads')

# Load condition 1
event_id = 1
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6))
data = epochs.get_data()  # as 3D matrix
data *= 1e13  # change unit to fT / cm
# Time vector
times = 1e3 * epochs.times  # change unit to ms

# Take only one channel
ch_name = raw.info['ch_names'][97]
data = data[:, 97:98, :]

evoked_data = np.mean(data, 0)

# data -= evoked_data[None,:,:] # remove evoked component
# evoked_data = np.mean(data, 0)

# Factor to down-sample the temporal dimension of the PSD computed by
# single_trial_power.  Decimation occurs after frequency decomposition and can
# be used to reduce memory usage (and possibly computational time of downstream
# operations such as nonparametric statistics) if you don't need high
# spectrotemporal resolution.
decim = 5
frequencies = np.arange(8, 40, 2)  # define frequencies of interest
sfreq = raw.info['sfreq']  # sampling in Hz
epochs_power = single_trial_power(data, sfreq=sfreq, frequencies=frequencies,
                                  n_cycles=4, n_jobs=1,
                                  baseline=(-100, 0), times=times,
                                  baseline_mode='ratio', decim=decim)

# Crop in time to keep only what is between 0 and 400 ms
time_mask = (times > 0) & (times < 400)
evoked_data = evoked_data[:, time_mask]
times = times[time_mask]

# The time vector reflects the original time points, not the decimated time
# points returned by single trial power. Be sure to decimate the time mask
# appropriately.
epochs_power = epochs_power[..., time_mask[::decim]]

epochs_power = epochs_power[:, 0, :, :]
epochs_power = np.log10(epochs_power)  # take log of ratio
# under the null hypothesis epochs_power should be now be 0

###############################################################################
# Compute statistic
# -----------------
threshold = 2.5
T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_1samp_test(epochs_power, n_permutations=100,
                                   threshold=threshold, tail=0)

###############################################################################
# View time-frequency plots
# -------------------------
plt.clf()
plt.subplots_adjust(0.12, 0.08, 0.96, 0.94, 0.2, 0.43)
plt.subplot(2, 1, 1)
plt.plot(times, evoked_data.T)
plt.title('Evoked response (%s)' % ch_name)
plt.xlabel('time (ms)')
plt.ylabel('Magnetic Field (fT/cm)')
plt.xlim(times[0], times[-1])
plt.ylim(-100, 250)

plt.subplot(2, 1, 2)

# Create new stats image with only significant clusters
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(clusters, cluster_p_values):
    if p_val <= 0.05:
        T_obs_plot[c] = T_obs[c]

vmax = np.max(np.abs(T_obs))
vmin = -vmax
plt.imshow(T_obs, cmap=plt.cm.gray,
           extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
           aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
           extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
           aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.xlabel('time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title('Induced power (%s)' % ch_name)
plt.show()
