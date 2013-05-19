"""
======================================
Investigate Single Trial Power Spectra
======================================

In some situations it is desirable conduct a simple FFT spectral
analyis, either to inform or to validate subsequent / previous
multitaper or time-frequency analyses.
In this exammple we will look at single trial spectra and then
compute average cross-trial spectra to identify channels and
frequencies of interest for subsequent TFR analyses.
"""

# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

print __doc__

import numpy as np
import scipy

import mne
from mne import fiff
from mne.datasets import sample

###############################################################################
# Set parameters
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'

# Setup for reading the raw data
raw = fiff.Raw(raw_fname, preload=True)
# raw.filter(l_freq=5, n_jobs=2)
events = mne.read_events(event_fname)

event_id = 1
# choses large time window to improve spectral estimation
tmin, tmax = -1, 1
include = []
raw.info['bads'] += ['MEG 2443']  # bads

# picks MEG gradiometers
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                        stim=False, include=include, exclude='bads')

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6))

data = epochs.get_data()[:, :-1, :]  # drop EOG channel

# compute frequencies returned by FFT
sfreq, n_times = raw.info['sfreq'], data.shape[2]
freqs = np.arange(n_times) * sfreq / n_times

# select frequencies of interests
foi = (freqs > 7) & (freqs < 100)  # cut below 7Hz to improve plot

# compute magnitudes from Fast Fourier transform for selected frequencies
magnitudes = np.abs(scipy.fft(data))[:, :, foi]

some_trial = 12
powers = [magnitudes[some_trial], np.mean(magnitudes, axis=0)]
titles = ['Single trial FFT magnitudes', 'cross-trial FFT magnitudes']
yticks = (np.arange(0, 200, 20), freqs[foi][::20].round(1))

import pylab as pl
for power, title in zip(powers, titles):
    pl.figure()
    pl.title(title)
    pl.imshow(power.T, aspect='auto', origin='lower')
    pl.ylabel('Frequency (Hz)')
    pl.xlabel('MEG channels (Gradiomemters)')
    pl.yticks(*yticks)
    pl.show()

max_power_ch = tuple(epochs.ch_names[k] for k in np.argmax(powers[1], 0)[:2])
print 'Maximum power in channels: %s, %s' % max_power_ch

# In the second image we can observe certain channel groups exposing
# stronger power than others. Second, in comparison to the single
# trial image we can see the frequency extent growing for these channels.
# This indicates oscillatory responses might have been captured here.
# Finally, we clearly see power line artifacts across channels.
# Against the background of these insights we might want to supply
# the channels around indices 100, 180 to a time-frequency
# analysis for frequencies 7-30Hz. (=> continued in plot_time_frequency)
