"""
==============================================================
Compute seed-based time-frequency connectivity in sensor space
==============================================================

Computes the connectivity between a seed-gradiometer close to the visual cortex
and all other gradiometers. The connectivity is computed in the time-frequency
domain using Morlet wavelets and the debiased squared weighted phase lag index
:footcite:`VinckEtAl2011` is used as connectivity metric.
"""
# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne import io
from mne.connectivity import spectral_connectivity, seed_target_indices
from mne.datasets import sample
from mne.time_frequency import AverageTFR

print(__doc__)

###############################################################################
# Set parameters
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

# Add a bad channel
raw.info['bads'] += ['MEG 2443']

# Pick MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=True,
                       exclude='bads')

# Create epochs for left-visual condition
event_id, tmin, tmax = 3, -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6),
                    preload=True)

# Use 'MEG 2343' as seed
seed_ch = 'MEG 2343'
picks_ch_names = [raw.ch_names[i] for i in picks]

# Create seed-target indices for connectivity computation
seed = picks_ch_names.index(seed_ch)
targets = np.arange(len(picks))
indices = seed_target_indices(seed, targets)

# Define wavelet frequencies and number of cycles
cwt_freqs = np.arange(7, 30, 2)
cwt_n_cycles = cwt_freqs / 7.

# Run the connectivity analysis using 2 parallel jobs
sfreq = raw.info['sfreq']  # the sampling frequency
con, freqs, times, _, _ = spectral_connectivity(
    epochs, indices=indices,
    method='wpli2_debiased', mode='cwt_morlet', sfreq=sfreq,
    cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles, n_jobs=1)

# Mark the seed channel with a value of 1.0, so we can see it in the plot
con[np.where(indices[1] == seed)] = 1.0

# Show topography of connectivity from seed
title = 'WPLI2 - Visual - Seed %s' % seed_ch

layout = mne.find_layout(epochs.info, 'meg')  # use full layout

tfr = AverageTFR(epochs.info, con, times, freqs, len(epochs))
tfr.plot_topo(fig_facecolor='w', font_color='k', border='k')


###############################################################################
# References
# ----------
# .. footbibliography::
