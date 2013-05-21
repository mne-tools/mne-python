"""
==========================================
Investigate Single Trial Magnitude Spectra
==========================================

In some situations it is desirable conduct a simple FFT spectral
analysis, either to inform or to validate subsequent / previous
time-frequency analyses.
In this example we will look at single trial spectra and then
compute average spectra to identify channels and
frequencies of interest for subsequent TFR analyses.
We will conclude with exposing different windowing techniques that
help addressing the problem of spectral leakage.
"""

# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

print __doc__

import numpy as np
import scipy
from scipy import fftpack

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
events = mne.read_events(event_fname)

event_id = dict(vis_l=3, vis_r=4)  # use more trial
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
freq_mask = (freqs > 7) & (freqs < 100)  # cut below 7Hz to improve plot

# compute magnitudes from Fast Fourier transform for selected frequencies
magnitudes = np.abs(fftpack.fft(data))[:, :, freq_mask]

import pylab as pl
some_trial = 12
spectra = (magnitudes[some_trial], np.mean(magnitudes, axis=0))
titles = ('single trial', 'averaged across trials')
xlabel = 'MEG channels (Gradiomemters)'
ylabel = 'Frequency (Hz)'
yticks = freqs[freq_mask].round(1)

fig, axes = pl.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
for ii, (spec, title, ax) in enumerate(
        zip(spectra, titles, axes.flatten())):
    ax.set_title(title, size='small')
    ax.imshow(spec.T, aspect='auto', origin='lower')
    ax.set_yticks(np.arange(0, len(yticks), 20))
    if ii < 1:
        ax.set_xlabel(xlabel)
        ax.set_yticklabels(yticks[::20])
        ax.set_ylabel(ylabel)

fig.suptitle('FFT spectra (rectangular window)')
mne.viz.tight_layout(2)
fig.show()

max_ch = tuple(epochs.ch_names[k] for k in np.argmax(spectra[1], 0)[:2])
print 'Maximum magnitude in channels: %s, %s' % max_ch

# In the second image we can observe certain channel groups exposing
# stronger power than others. Second, in comparison to the single
# trial image we can see the frequency extent growing for these channels.
# This indicates oscillatory responses might have been captured here.
# Finally, we clearly see power line artifacts across channels.
# Against the background of these insights we might want to supply
# the channels around indices 100, 180 to a time-frequency
# analysis for frequencies 7-30Hz. (=> continued in plot_time_frequency)

# However, as we did not use an explicit windowing function the waveforms
# are truncated at the beginning and the end of our time window which
# is not beneficial for the quality of our spectral estimation.
# In the following, we will hence apply a Hanning window and multi-taper
# windows to mitigate the spectral leakage and double-check our results.

# let's assemble a list of of alternative single trial spectral estimates.

# first the Hanning
hw = scipy.hanning(n_times)
magnitudes = [np.abs(fftpack.fft(data[some_trial] * hw))[:, freq_mask]]

# ... now the multi taper
from mne.time_frequency.multitaper import multitaper_psd
mt_x,  mt_freqs = multitaper_psd(data[some_trial], raw.info['sfreq'])
# as this is returns the power estimates, we will take the square root
magnitudes.append(np.sqrt(mt_x[:, (mt_freqs > 7) & (mt_freqs < 100)]))

for m, title in zip(magnitudes, ['Hanning', 'multitaper']):
    pl.figure()
    pl.title('single trial spectra (%s)' % title)
    pl.imshow(m.T, aspect='auto', origin='lower')
    pl.ylabel(ylabel)
    pl.yticks(np.arange(0, len(yticks), 20), yticks[::20])
    pl.xlabel(xlabel)
    pl.show()

# The quality of the plots improves with the tapers applied,
# especially with the multi taper. Still, for our purpose the initial
# variant did produce acceptable results, hence, we don't have to update
# our conclusions. But it is advisable to at least use the
# Hanning window.
