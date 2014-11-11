"""
==============================================================
Time-frequency analysis using multitaper method for real and
simulated data.
==============================================================

Plot for real data rendered in z-score relative to baseline in log-scale.
"""
print(__doc__)

# Authors: Hari Bharadwaj <hari@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
import mne
from mne import io, create_info, EpochsArray
from mne.time_frequency import tfr_multitaper
from mne.datasets import somato

###############################################################################
# Load real somatosensory sample data.
data_path = somato.data_path()
raw_fname = data_path + '/MEG/somato/sef_raw_sss.fif'
event_id, tmin, tmax = 1, -1., 3.

# Setup for reading the raw data
raw = io.Raw(raw_fname)
baseline = (None, 0)
events = mne.find_events(raw, stim_channel='STI 014')

# Pick a good channel for somatosensory responses.
picks = [raw.info['ch_names'].index('MEG 1142'), ]

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline, reject=dict(grad=4000e-13))

###############################################################################
# Calculate power and intertrial coherence

freqs = np.arange(5, 50, 2)  # define frequencies of interest
n_cycles = freqs / 2.  # 0.5 second time windows for all frequencies

# Choose time x (full) bandwidth product
time_bandwidth = 4.0  # With 0.5 s time windows, this gives 8 Hz smoothing

power = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                       time_bandwidth=time_bandwidth, return_itc=False,
                       n_jobs=1)

# Baseline correction can be applied to power or done in plots
# To illustrate the baseline correction in plots the next line is commented
# power.apply_baseline(baseline=(-0.5, 0), mode='zlogratio')

# Plot power. BAseline correct using z-score in log-scale.
power.plot([0], baseline=(-0.5, 0), mode='zlogratio', vmin=-10, vmax=50)


###############################################################################
# Simulated example
###############################################################################

sfreq = 1000.0
ch_names = ['SIM0001', 'SIM0002']
ch_types = ['grad', 'grad']
info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

n_times = int(sfreq)  # Second long epochs
n_epochs = 40
noise = np.random.randn(n_epochs, len(ch_names), n_times)

# Add a 50 Hz sinusoidal burst to the noise and ramp it.
t = np.arange(n_times) / sfreq
signal = np.sin(np.pi * 2 * 50 * t)  # 50 Hz sinusoid signal
signal[np.logical_or(t < 0.45, t > 0.55)] = 0  # Hard windowing
on_time = np.logical_and(t >= 0.48, t <= 0.52)
signal[on_time] *= np.hanning(on_time.sum())  # Ramping
dat = noise + signal

reject = dict(grad=4000)
events = np.empty((n_epochs, 3))
first_event_sample = 100
event_id = dict(Sin50Hz=1)
for k in range(n_epochs):
    events[k, :] = first_event_sample + k * n_times, 0, event_id['Sin50Hz']

epochs = EpochsArray(data=dat, info=info, events=events, event_id=event_id,
                     reject=reject)

freqs = np.arange(5, 100, 3)

#############################################
# Consider different parameter possibilities
#############################################

# You can trade time resolution or frequency resolution or both
# in order to get a reduction in variance

# (1) Lease smoothing (most variance/background fluctuations)
n_cycles = freqs/2.
time_bandwidth = 2.0  # Least possible frequency-smoothing (1 taper)
power = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                       time_bandwidth=time_bandwidth, return_itc=False)
# Plot results. Baseline correct based on first 100 ms.
power.plot([0], baseline=(0., 0.1), mode='mean', vmin=0., vmax=5.,
           title='Sim: Least smoothing, most variance')


# (2) Less frequency smoothing, more time smoothing
n_cycles = freqs
time_bandwidth = 4.0  # Least possible frequency-smoothing (3 tapers)
power = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                       time_bandwidth=time_bandwidth, return_itc=False)
# Plot results. Baseline correct based on first 100 ms.
power.plot([0], baseline=(0., 0.1), mode='mean', vmin=0., vmax=5.,
           title='Sim: Less frequency smoothing, more time smoothing')


# (3) Less-time smoothing, more frequency smoothing
n_cycles = freqs/2.
time_bandwidth = 8.0  # Least possible frequency-smoothing (6 taper)
power = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                       time_bandwidth=time_bandwidth, return_itc=False)
# Plot results. Baseline correct based on first 100 ms.
power.plot([0], baseline=(0., 0.1), mode='mean', vmin=0., vmax=5.,
           title='Sim: Less time smoothing, more frequency smoothing')
