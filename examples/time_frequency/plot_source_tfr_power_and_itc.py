"""
========================================================================
Compute induced power and itc in source space using dSPM and multitapers
========================================================================

Compute induced power and inter-trial-coherence in source space,
using a multitaper time-frequency transform on a list of source
estimate objects.

"""
# Authors: Dirk Gütlin <dirk.guetlin@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np

import mne
from mne import io
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs
from mne.time_frequency import tfr_multitaper

print(__doc__)

###############################################################################
# Set parameters
data_path = sample.data_path()
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
fname_inv = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-meg-oct-6-meg-fixed-inv.fif')
fname_inv_vol = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis-meg-vol-7-meg-inv.fif')
subjects_dir = op.join(data_path, 'subjects')
tmin, tmax, event_id = -0.2, 0.5, 1

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname)
events = mne.find_events(raw, stim_channel='STI 014')

include = []
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True,
                       stim=False, include=include, exclude='bads')

# Load condition 1
event_id = 1
events = events[:10]  # take 10 events to keep the computation time low
# Use linear detrend to reduce any edge artifacts
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6),
                    preload=True, detrend=1)

###############################################################################
# compute SourceTFR for a list of SourceEstimates

inverse_operator = read_inverse_operator(fname_inv)

# return_generator=True will reduce memory load
# delayed=True will reduce time needed for computation
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2=1. / 9.,
                            method="dSPM", prepared=False,
                            return_generator=True, delayed=True)

# calculate a time-frequency transform in source space
freqs = np.arange(8, 18, 2)
pow, itc = tfr_multitaper(stcs, freqs=freqs, n_cycles=2, use_fft=True,
                          average=True, return_itc=True)

###############################################################################
# plot mean power, itc between fmin and fmax

fmin, fmax = 8, 12
initial_time = 0.1

pow.plot(fmin=fmin, fmax=fmax, subjects_dir=subjects_dir,
         initial_time=initial_time)
itc.plot(fmin=fmin, fmax=fmax, subjects_dir=subjects_dir,
         initial_time=initial_time)

###############################################################################
# compute SourceTFR for a list of VolVectorSourceEstimates

inverse_operator = read_inverse_operator(fname_inv_vol)

# return_generator=True will reduce memory load
# delayed=True will reduce time needed for computation
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2=1. / 9.,
                            method="dSPM", pick_ori="vector", prepared=False,
                            return_generator=True, delayed=True)

# calculate a time-frequency transform in source space
freqs = np.arange(8, 18, 2)
pow, itc = tfr_multitaper(stcs, freqs=freqs, n_cycles=2, use_fft=True,
                          average=True, return_itc=True)

###############################################################################
# plot mean power and itc between fmin and fmax for volume estimates

fmin, fmax = 8, 12
initial_time = 0.1

src = inverse_operator["src"]
pow.plot(fmin=fmin, fmax=fmax, src=src, subject="sample",
         subjects_dir=subjects_dir, mode='glass_brain')

itc.plot(fmin=fmin, fmax=fmax, src=src, subject="sample",
         subjects_dir=subjects_dir, mode='glass_brain')
