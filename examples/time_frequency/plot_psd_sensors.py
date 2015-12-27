"""
==============================================================
PSD estimation for MEG sensors
==============================================================

PSD calculation with both multitaper and welch's method are displayed
"""
# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

import mne
from mne import io
from mne.time_frequency import psd_welch, psd_multitaper
from mne.datasets import somato

print(__doc__)

###############################################################################
# Set parameters
data_path = somato.data_path()
raw_fname = data_path + '/MEG/somato/sef_raw_sss.fif'
event_id, tmin, tmax = 1, -1., 3.
fmin, fmax = 2, 40
n_fft = 256

# Setup for reading the raw data
raw = io.Raw(raw_fname)
baseline = (None, 0)
events = mne.find_events(raw, stim_channel='STI 014')

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True, stim=False)

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline, reject=dict(grad=4000e-13, eog=350e-6))
picks_psd = picks[:5]

###############################################################################
# Calculate power spectral density
psds_we, freqs_we = psd_welch(epochs, tmin=tmin, tmax=tmax, fmin=fmin,
                              fmax=fmax, n_fft=n_fft, proj=False,
                              picks=picks_psd)
psds_mt, freqs_mt = psd_multitaper(epochs, tmin=tmin, tmax=tmax, fmin=fmin,
                                   fmax=fmax, low_bias=True, proj=False,
                                   picks=picks_psd)

f, axs = plt.subplots(1, 2)
for psd, freqs, ax in zip([psds_we, psds_mt], [freqs_we, freqs_mt], axs):
    ax.plot(freqs, psd.mean(0).T)
axs[0].set(title='Welch PSD')
axs[1].set(title='Multitaper PSD')
plt.setp(axs, xlabel='Frequency', ylabel='Power Spectral Density (PSD)')
mne.viz.tight_layout()
