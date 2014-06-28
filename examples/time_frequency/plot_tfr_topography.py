"""
==============================================================
Time-frequency representations on topographies for MEG sensors
==============================================================

Both average power and intertrial coherence are displayed.
"""
print(__doc__)

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import mne
from mne import io
from mne.time_frequency import tfr_morlet
from mne.datasets import sample

data_path = sample.data_path()

raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
event_id, tmin, tmax = 1, -0.2, 0.5

# Setup for reading the raw data
raw = io.Raw(raw_fname)
events = mne.read_events(event_fname)

include = []
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                       stim=False, include=include, exclude='bads')

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6))

###############################################################################
# Calculate power and intertrial coherence

freqs = np.arange(7, 30, 3)  # define frequencies of interest
n_cycles = freqs / 7.  # different number of cycle per frequency
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=False,
                        return_itc=True, decim=3, n_jobs=1)

power.plot_topo(baseline=(None, 0), mode='ratio', title='Average power',
                vmin=0., vmax=14.)
power.plot([92], baseline=(None, 0), mode='ratio')

itc.plot_topo(title='Inter-Trial coherence', vmin=0., vmax=1.)
