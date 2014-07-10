"""
======================================
Investigate Single Trial Power Spectra
======================================

In this example we will look at single trial spectra and then
compute average spectra to identify channels and
frequencies of interest for subsequent TFR analyses.
"""

# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import sample
from mne.time_frequency import compute_epochs_psd
###############################################################################
# Set parameters
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'

# Setup for reading the raw data
raw = io.Raw(raw_fname)
events = mne.read_events(event_fname)

tmin, tmax, event_id = -1., 1., 1
include = []
raw.info['bads'] += ['MEG 2443']  # bads

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                       stim=False, include=include, exclude='bads')

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks, proj=True,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6))


n_fft = 256  # the FFT size. Ideally a power of 2
psds, freqs = compute_epochs_psd(epochs, fmin=2, fmax=200, n_fft=n_fft,
                                 n_jobs=2)

# average psds and save psds from first trial separately
average_psds = psds.mean(0)
average_psds = 10 * np.log10(average_psds)  # transform into dB
some_psds = 10 * np.log10(psds[12])


fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))

fig.suptitle('Single trial power', fontsize=12)

freq_mask = freqs < 150
freqs = freqs[freq_mask]

ax1.set_title('single trial', fontsize=10)
ax1.imshow(some_psds[:, freq_mask].T, aspect='auto', origin='lower')
ax1.set_yticks(np.arange(0, len(freqs), 10))
ax1.set_yticklabels(freqs[::10].round(1))
ax1.set_ylabel('Frequency (Hz)')

ax2.set_title('averaged over trials', fontsize=10)
ax2.imshow(average_psds[:, freq_mask].T, aspect='auto', origin='lower')
ax2.set_xticks(np.arange(0, len(picks), 30))
ax2.set_xticklabels(picks[::30])
ax2.set_xlabel('MEG channel index (Gradiometers)')

mne.viz.tight_layout()
plt.show()

# In the second image we clearly observe certain channel groups exposing
# stronger power than others. Second, in comparison to the single
# trial image we can see the frequency extent slightly growing for these
# channels which might indicate oscillatory responses.
# The ``plot_time_frequency.py`` example investigates one of the channels
# around index 140.
# Finally, also note the power line artifacts across all channels.

# Now let's take a look at the spatial distributions of the lower frequencies
# Note. We're 'abusing' the Evoked.plot_topomap method here to display
# our average powermap

evoked = epochs.average()  # create evoked
evoked.data = average_psds[:, freq_mask]  # insert our psd data
evoked.times = freqs  # replace times with frequencies.
evoked.plot_topomap(ch_type='grad', times=range(5, 12, 2),
                    scale=1, scale_time=1, time_format='%0.1f Hz',
                    cmap='Reds', vmin=np.min, vmax=np.max,
                    unit='dB', format='-%0.1f')
