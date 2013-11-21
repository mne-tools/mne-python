"""
===================================================================
Plot time-frequency representations on topographies for MEG sensors
===================================================================

Both induced power and phase locking values are displayed.
"""
print __doc__

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np
import pylab as pl
import mne
from mne import fiff
from mne.time_frequency import induced_power
from mne.viz import plot_topo_power, plot_topo_phase_lock
from mne.datasets import sample

data_path = sample.data_path()

raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
event_id, tmin, tmax = 1, -0.2, 0.5

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.read_events(event_fname)

include = []
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

# picks MEG gradiometers
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                        stim=False, include=include, exclude='bads')

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6))
data = epochs.get_data()  # as 3D matrix

layout = mne.layouts.read_layout('Vectorview-all')

###############################################################################
# Calculate power and phase locking value

frequencies = np.arange(7, 30, 3)  # define frequencies of interest
n_cycles = frequencies / float(7)  # different number of cycle per frequency
Fs = raw.info['sfreq']  # sampling in Hz
decim = 3
power, phase_lock = induced_power(data, Fs=Fs, frequencies=frequencies,
                                  n_cycles=n_cycles, n_jobs=1, use_fft=False,
                                  decim=decim, zero_mean=True)

###############################################################################
# Prepare topography plots, set baseline correction parameters

baseline = (None, 0)  # set the baseline for induced power
mode = 'ratio'  # set mode for baseline rescaling

###############################################################################
# Show topography of power.

title = 'Induced power - MNE sample data'
plot_topo_power(epochs, power, frequencies, layout, baseline=baseline,
                mode=mode, decim=decim, vmin=0., vmax=14, title=title)
pl.show()

###############################################################################
# Show topography of phase locking value (PLV)

mode = None  # no baseline rescaling for PLV

title = 'Phase locking value - MNE sample data'
plot_topo_phase_lock(epochs, phase_lock, frequencies, layout,
                     baseline=baseline, mode=mode, decim=decim, title=title)

pl.show()
