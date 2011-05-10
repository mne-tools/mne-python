"""
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
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np

import mne
from mne import fiff
from mne.time_frequency import single_trial_power
from mne.stats import permutation_cluster_1samp_test
from mne.datasets import sample

###############################################################################
# Set parameters
data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_id = 1
tmin = -0.2
tmax = 0.5

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.find_events(raw)

include = []
exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more

# picks MEG gradiometers
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                                stim=False, include=include, exclude=exclude)

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

frequencies = np.arange(8, 40, 2)  # define frequencies of interest
Fs = raw.info['sfreq']  # sampling in Hz
epochs_power = single_trial_power(data, Fs=Fs, frequencies=frequencies,
                                  n_cycles=4, use_fft=False, n_jobs=1,
                                  baseline=(-100, 0), times=times,
                                  baseline_mode='ratio')

# Crop in time to keep only what is between 0 and 400 ms
time_mask = (times > 0) & (times < 400)
epochs_power = epochs_power[:, :, :, time_mask]
evoked_data = evoked_data[:, time_mask]
times = times[time_mask]

epochs_power = epochs_power[:, 0, :, :]
epochs_power = np.log10(epochs_power)  # take log of ratio
# under the null hypothesis epochs_power should be now be 0

###############################################################################
# Compute statistic
threshold = 2.5
T_obs, clusters, cluster_p_values, H0 = \
                   permutation_cluster_1samp_test(epochs_power,
                               n_permutations=100, threshold=threshold, tail=0)

###############################################################################
# View time-frequency plots
import pylab as pl
pl.clf()
pl.subplots_adjust(0.12, 0.08, 0.96, 0.94, 0.2, 0.43)
pl.subplot(2, 1, 1)
pl.plot(times, evoked_data.T)
pl.title('Evoked response (%s)' % ch_name)
pl.xlabel('time (ms)')
pl.ylabel('Magnetic Field (fT/cm)')
pl.xlim(times[0], times[-1])
pl.ylim(-100, 250)

pl.subplot(2, 1, 2)

# Create new stats image with only significant clusters
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(clusters, cluster_p_values):
    if p_val <= 0.05:
        T_obs_plot[c] = T_obs[c]

vmax = np.max(np.abs(T_obs))
vmin = -vmax
pl.imshow(T_obs, cmap=pl.cm.gray, extent=[times[0], times[-1],
                                          frequencies[0], frequencies[-1]],
                                  aspect='auto', origin='lower',
                                  vmin=vmin, vmax=vmax)
pl.imshow(T_obs_plot, cmap=pl.cm.jet, extent=[times[0], times[-1],
                                          frequencies[0], frequencies[-1]],
                                  aspect='auto', origin='lower',
                                  vmin=vmin, vmax=vmax)
pl.colorbar()
pl.xlabel('time (ms)')
pl.ylabel('Frequency (Hz)')
pl.title('Induced power (%s)' % ch_name)
pl.show()
