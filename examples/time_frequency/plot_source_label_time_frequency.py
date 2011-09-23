"""
=========================================================
Compute power and phase lock in label of the source space
=========================================================

Returns time-frequency maps of induced power and phase lock
in the source space. The inverse method is linear based on dSPM inverse
operator.

"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np

import mne
from mne import fiff
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, source_induced_power

###############################################################################
# Set parameters
data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
label_name = 'Aud-lh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name

tmin, tmax, event_id = -0.2, 0.5, 1

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.find_events(raw)
inverse_operator = read_inverse_operator(fname_inv)

include = []
exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more

# picks MEG channels
picks = fiff.pick_types(raw.info, meg=True, eeg=False, eog=True,
                                stim=False, include=include, exclude=exclude)

# Load condition 1
event_id = 1
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6),
                    preload=True)

# Compute a source estimate per frequency band
frequencies = np.arange(7, 30, 2)  # define frequencies of interest
label = mne.read_label(fname_label)
power, phase_lock = source_induced_power(epochs, inverse_operator, frequencies,
                            label, baseline=(-0.1, 0), baseline_mode='percent',
                            n_cycles=2, n_jobs=1)

power = np.mean(power, axis=0)  # average over sources
phase_lock = np.mean(phase_lock, axis=0)  # average over sources
times = epochs.times

###############################################################################
# View time-frequency plots
import pylab as pl
pl.clf()
pl.subplots_adjust(0.1, 0.08, 0.96, 0.94, 0.2, 0.43)
pl.subplot(2, 1, 1)
pl.imshow(20 * power, extent=[times[0], times[-1],
                                      frequencies[0], frequencies[-1]],
          aspect='auto', origin='lower')
pl.xlabel('Time (s)')
pl.ylabel('Frequency (Hz)')
pl.title('Induced power in %s' % label_name)
pl.colorbar()

pl.subplot(2, 1, 2)
pl.imshow(phase_lock, extent=[times[0], times[-1],
                              frequencies[0], frequencies[-1]],
          aspect='auto', origin='lower')
pl.xlabel('Time (s)')
pl.ylabel('Frequency (Hz)')
pl.title('Phase-lock in %s' % label_name)
pl.colorbar()
pl.show()
