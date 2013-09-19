"""
=========================================================
Compute power and phase lock in label of the source space
=========================================================

Compute time-frequency maps of power and phase lock in the source space.
The inverse method is linear based on dSPM inverse operator.

The example also shows the difference in the time-frequency maps
when they are computed with and without subtracting the evoked response
from each epoch. The former results in induced activity only while the
latter also includes evoked (stimulus-locked) activity.
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
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
label_name = 'Aud-rh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name

tmin, tmax, event_id = -0.2, 0.5, 2

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.find_events(raw, stim_channel='STI 014')
inverse_operator = read_inverse_operator(fname_inv)

include = []
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

# Picks MEG channels
picks = fiff.pick_types(raw.info, meg=True, eeg=False, eog=True,
                                stim=False, include=include, exclude='bads')

# Load epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6),
                    preload=True)

# Compute a source estimate per frequency band inlcuding and excluding the
# evoked response
frequencies = np.arange(7, 30, 2)  # define frequencies of interest
label = mne.read_label(fname_label)
n_cycles = frequencies / float(7)  # different number of cycle per frequency

import pylab as pl
pl.clf()

for ii, title in enumerate(['evoked + induced', 'induced only']):
    if ii == 1:
        # subtract the evoked response in order to exclude evoked activity
        epochs.subtract_evoked()

    # compute the source space power and phase lock
    power, phase_lock = source_induced_power(epochs, inverse_operator,
        frequencies, label, baseline=(-0.1, 0), baseline_mode='percent',
        n_cycles=n_cycles, n_jobs=1)

    power = np.mean(power, axis=0)  # average over sources
    phase_lock = np.mean(phase_lock, axis=0)  # average over sources
    times = epochs.times

    ##########################################################################
    # View time-frequency plots
    pl.subplots_adjust(0.1, 0.08, 0.96, 0.94, 0.2, 0.43)
    pl.subplot(2, 2, 2 * ii + 1)
    pl.imshow(20 * power, extent=[times[0], times[-1],
                                  frequencies[0], frequencies[-1]],
              aspect='auto', origin='lower')
    pl.xlabel('Time (s)')
    pl.ylabel('Frequency (Hz)')
    pl.title('Power (%s) in %s' % (title, label_name))
    pl.colorbar()

    pl.subplot(2, 2, 2 * ii + 2)
    pl.imshow(phase_lock, extent=[times[0], times[-1],
                                  frequencies[0], frequencies[-1]],
              aspect='auto', origin='lower')
    pl.xlabel('Time (s)')
    pl.ylabel('Frequency (Hz)')
    pl.title('Phase-lock (%s) in %s' % (title, label_name))
    pl.colorbar()
    pl.show()
