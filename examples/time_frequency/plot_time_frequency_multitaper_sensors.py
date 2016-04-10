"""
===============================================
Time-frequency analysis using multitaper method
===============================================

This examples computes induced power and intertrial
coherence (ITC) using a multitaper method on a somato sensory MEG data.
The power plot is rendered so that baseline is mean zero.
"""
# Authors: Hari Bharadwaj <hari@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne import io
from mne.time_frequency import tfr_multitaper
from mne.datasets import somato

print(__doc__)

###############################################################################
# Load real somatosensory sample data.
data_path = somato.data_path()
raw_fname = data_path + '/MEG/somato/sef_raw_sss.fif'
event_id, tmin, tmax = 1, -1., 3.

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname)
baseline = (None, 0)
events = mne.find_events(raw, stim_channel='STI 014')

# Pick a good channel for somatosensory responses.
picks = [raw.info['ch_names'].index('MEG 1142'), ]

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline, reject=dict(grad=4000e-13))

###############################################################################
# Calculate power

freqs = np.arange(5., 50., 2.)  # define frequencies of interest
n_cycles = freqs / 2.  # 0.5 second time windows for all frequencies

# Choose time x (full) bandwidth product
time_bandwidth = 4.0  # With 0.5 s time windows, this gives 8 Hz smoothing

power, itc = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                            use_fft=True, time_bandwidth=time_bandwidth,
                            return_itc=True, n_jobs=1)

# Plot results (with baseline correction only for power)
power.plot([0], baseline=(-0.5, 0), mode='mean', title='MEG 1142 - Power')
itc.plot([0], title='MEG 1142 - Intertrial Coherence')
