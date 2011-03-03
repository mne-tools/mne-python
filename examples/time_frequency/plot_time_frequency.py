"""
=========================================================
Time frequency : Induced power and inter-trial phase-lock
=========================================================

This script shows how to compute induced power and inter-trial
phase-lock for a list of epochs read in a raw file given
a list of events.

"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np

import mne
from mne import fiff
from mne import time_frequency
from mne.datasets import sample

###############################################################################
# Set parameters
data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
event_id = 1
tmin = -0.2
tmax = 0.5

# Setup for reading the raw data
raw = fiff.setup_read_raw(raw_fname)
events = mne.read_events(event_fname)

include = []
exclude = raw['info']['bads'] + ['MEG 2443', 'EEG 053'] # bads + 2 more

# picks MEG gradiometers
picks = fiff.pick_types(raw['info'], meg='grad', eeg=False,
                                stim=False, include=include, exclude=exclude)

picks = [picks[97]]
epochs = mne.Epochs(raw, events, event_id,
                    tmin, tmax, picks=picks, baseline=(None, 0))
data = epochs.get_data() # as 3D matrix
evoked = epochs.average() # compute evoked fields

times = 1e3 * epochs.times # change unit to ms
evoked *= 1e13 # change unit to fT / cm

frequencies = np.arange(7, 30, 3) # define frequencies of interest
Fs = raw['info']['sfreq'] # sampling in Hz
power, phase_lock = time_frequency(data, Fs=Fs, frequencies=frequencies,
                                   n_cycles=2, n_jobs=1, use_fft=False)

###############################################################################
# View time-frequency plots
import pylab as pl
pl.clf()
pl.subplots_adjust(0.1, 0.08, 0.96, 0.94, 0.2, 0.63)
pl.subplot(3, 1, 1)
pl.plot(times, evoked.T)
pl.title('Evoked response (%s)' % raw['info']['ch_names'][picks[0]])
pl.xlabel('time (ms)')
pl.ylabel('Magnetic Field (fT/cm)')
pl.xlim(times[0], times[-1])
pl.ylim(-150, 300)

pl.subplot(3, 1, 2)
pl.imshow(20*np.log10(power[0]), extent=[times[0], times[-1],
                                      frequencies[0], frequencies[-1]],
          aspect='auto', origin='lower')
pl.xlabel('Time (s)')
pl.ylabel('Frequency (Hz)')
pl.title('Induced power (%s)' % raw['info']['ch_names'][picks[0]])
pl.colorbar()

pl.subplot(3, 1, 3)
pl.imshow(phase_lock[0], extent=[times[0], times[-1],
                              frequencies[0], frequencies[-1]],
          aspect='auto', origin='lower')
pl.xlabel('Time (s)')
pl.ylabel('PLF')
pl.title('Phase-lock (%s)' % raw['info']['ch_names'][picks[0]])
pl.colorbar()
pl.show()
