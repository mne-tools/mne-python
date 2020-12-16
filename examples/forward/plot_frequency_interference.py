# -*- coding: utf-8 -*-
"""
.. _tut-dipole-frequencies-source-modeling:

Frequency Interference
======================

This example shows how the spatial offset of phase-locked source time
courses with lower frequency oscillations have much less destructive
interference than those with higher frequencies leading to a 1/f power
spectral density (psd) distribution. This phenomena is commonly
misattributed to the skull acting as a lowpass filter.

.. contents:: This tutorial covers:
   :local:
   :depth: 2

"""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, apply_inverse
from mne.time_frequency import psd_array_multitaper
from mne.simulation import simulate_evoked


data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
sample_dir = op.join(data_path, 'MEG', 'sample')

# Read evoked data
fname_evoked = op.join(sample_dir, 'sample_audvis-ave.fif')
evoked = mne.read_evokeds(fname_evoked, condition=0, baseline=(None, 0))

# Read inverse solution
fname_inv = op.join(sample_dir, 'sample_audvis-meg-oct-6-meg-inv.fif')
inv = read_inverse_operator(fname_inv)

# Apply inverse solution, set pick_ori='vector' to obtain a
# :class:`mne.VectorSourceEstimate` object
snr = 3.0
lambda2 = 1.0 / snr ** 2
stc = apply_inverse(evoked, inv, lambda2, 'dSPM')

fwd = mne.read_forward_solution(
    op.join(sample_dir, 'sample_audvis-meg-eeg-oct-6-fwd.fif'))
mne.convert_forward_solution(fwd, surf_ori=True, copy=False)


##############################################################################
# Visualize 1/f power spectral density in m/eeg data
# --------------------------------------------------
#
# The power decreases with increasing frequency, this is usually an
# exponential decay.

psds, freqs = psd_array_multitaper(stc.data, stc.sfreq, fmax=50)
fig, ax = plt.subplots()
ax.plot(freqs, psds[:100].T)
ax.set_title('1/f Power Decay in Source Time Course')
ax.set_ylabel('Power')
ax.set_xlabel('Frequency (Hz)')
fig.show()

##############################################################################
# Make frequency time courses with equal power
# --------------------------------------------
#
# In this example, we use three different frequencies in the normal
# range to study physiologically to show the trend of decreasing power as
# frequency increases.

freqs = np.linspace(10, 50, 5)
np.random.seed(11)

for i in range(stc.data.shape[0]):
    stc.data[i] = np.sum([np.sin(2. * np.pi * freq * stc.times)
                          for freq in freqs], axis=0) * 1e-10


psds, freqs_psd = psd_array_multitaper(stc.data, stc.sfreq, fmin=2, fmax=55)
fig, ax = plt.subplots()
ax.plot(freqs_psd, psds[:100].T)
ax.set_title('Equal Power in Source Time Course')
ax.set_ylabel('Power')
ax.set_xlabel('Frequency (Hz)')
fig.show()

##############################################################################
# Compute evoked with forward model
# ---------------------------------
#
# Now we can simulate evoked with our pure sine wave source time course
# with known frequency peaks. We see that the lower the frequency, the greater
# the power. This is because differences in spatial location result in
# phase offset due to the difference in their distances to the sensors. Higher
# frequencies descructively interfere with each other more because the same
# distance is a larger phase offset causing less in-phase and more random
# interference.

evoked_sim = simulate_evoked(fwd, stc, evoked.info, cov=None, nave=np.inf)
psds, freqs_psd = psd_array_multitaper(
    evoked_sim.data, stc.sfreq, fmin=10, fmax=50)
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].plot(freqs_psd, psds[mne.pick_types(evoked_sim.info, meg='mag')].T)
axes[0].set_title('mag')
axes[1].plot(freqs_psd, psds[mne.pick_types(evoked_sim.info, meg='grad')].T)
axes[1].set_title('grad')
axes[2].plot(freqs_psd, psds[mne.pick_types(evoked_sim.info, eeg=True)].T)
axes[2].set_title('eeg')
fig.suptitle('1/f Power Decay in Evoked')
axes[0].set_ylabel('Power')
axes[1].set_xlabel('Frequency (Hz)')
fig.show()
