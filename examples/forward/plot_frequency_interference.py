# -*- coding: utf-8 -*-
"""
.. _ex-frequency-interference:

Frequency Interference
======================

This example shows how the spatial offset of nearly phase-locked source time
courses with lower frequency oscillations have much less destructive
interference than those with higher frequencies leading to a 1/f power
spectral density (psd) distribution. This phenomena is commonly
misattributed to the skull acting as a lowpass filter.

.. contents:: This tutorial covers:
   :local:
   :depth: 2

"""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#          John Mosher   <John.C.Mosher@uth.tmc.edu>
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


##############################################################################
# Visualize 1/f power spectral density in m/eeg data
# --------------------------------------------------
#
# The power decreases with increasing frequency, this is usually an
# exponential decay.

psds, freqs_psd = psd_array_multitaper(
    evoked.data, evoked.info['sfreq'], fmin=2, fmax=50)
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].plot(freqs_psd, psds[mne.pick_types(evoked.info, meg='mag')].T)
axes[0].set_title('mag')
axes[1].plot(freqs_psd, psds[mne.pick_types(evoked.info, meg='grad')].T)
axes[1].set_title('grad')
axes[2].plot(freqs_psd, psds[mne.pick_types(evoked.info, eeg=True)].T)
axes[2].set_title('eeg')
fig.suptitle('1/f Power Decay in Real Evoked')
axes[0].set_ylabel('Power')
axes[1].set_xlabel('Frequency (Hz)')
fig.show()

##############################################################################
# Make pure sine wave time courses with equal power for different frequencies
# ---------------------------------------------------------------------------
#
# In this example, we use different frequencies in the normal physiological
# range (alpha/beta) to show the trend of decreasing power as frequency
# increases for nearly phase-locked sources.

# Read inverse solution
fname_inv = op.join(sample_dir, 'sample_audvis-meg-oct-6-meg-inv.fif')
inv = read_inverse_operator(fname_inv)

# Apply inverse solution, set to obtain an :class:`mne.SourceEstimate` object
snr = 3.0
lambda2 = 1.0 / snr ** 2
stc = apply_inverse(evoked, inv, lambda2, 'dSPM')

fwd = mne.read_forward_solution(
    op.join(sample_dir, 'sample_audvis-meg-eeg-oct-6-fwd.fif'))
mne.convert_forward_solution(fwd, surf_ori=True, copy=False)

freqs = np.round(np.linspace(15, 40, 5), 2)

lh, rh = fwd['src']
seed_vert = lh['rr'][stc.lh_vertno[99]]  # reference, zero phase-offset
phase_shift = 1 / stc.sfreq  # 1 ms shifted per mm distant
max_dist = 30  # distance in mm of nearly in-phase sources


def euclidean_dist(vert0, vert1):
    return np.sum((vert1 - vert0) ** 2) ** 0.5


fig, axes = plt.subplots(1, freqs.size, figsize=(10, 5))
fig.suptitle('Source Time Courses')
stcs = dict()  # make source time courses for each frequency sine wave
for ax, freq in zip(axes, freqs):
    for i in range(stc.data.shape[0]):
        vert = lh['rr'][stc.lh_vertno[i]] if i < stc.lh_vertno.size else \
            rh['rr'][stc.rh_vertno[i - stc.lh_vertno.size]]
        dist = euclidean_dist(seed_vert, vert) * 1e3  # m -> mm
        stc.data[i] = np.sin(
            2. * np.pi * freq * (stc.times + phase_shift * dist)) * 1e-10 \
            if dist < max_dist else 0  # if too large distance, no oscillation

    stcs[freq] = stc.copy()
    psds, freqs_psd = psd_array_multitaper(stc.data, stc.sfreq,
                                           fmin=2, fmax=55)
    ax.plot(freqs_psd, psds[0])
    ax.set_title(f'{freq} Hz')


axes[0].set_ylabel('Power')
axes[freqs.size // 2].set_xlabel('Frequency (Hz)')

fig.tight_layout()
fig.show()

##############################################################################
# Compute evoked with forward model
# ---------------------------------
#
# Now we can simulate evoked with our pure sine wave source time course
# with known frequency peaks. We see that the lower the frequency, the greater
# the power. This is because the same millisecond differences in phase
# offset results in greater descructive interference for higher frequencies
# because, for higher frequencies, the period is shorter so the same amount of
# time is a greater proportion of the period and thus the signals are more
# offset in phase. This causes greater descructive interference.

fig, axes_all = plt.subplots(3, freqs.size, figsize=(12, 8))
fig.suptitle('1/f Power Decay in Simulated Evoked')

for axes, freq in zip(axes_all.T, freqs):
    evoked_sim = simulate_evoked(fwd, stcs[freq], evoked.info,
                                 cov=None, nave=np.inf)
    psds, freqs_psd = psd_array_multitaper(
        evoked_sim.data, stc.sfreq, fmin=2, fmax=60)

    axes[0].plot(freqs_psd, psds[mne.pick_types(evoked_sim.info, meg='mag')].T)
    axes[0].set_title(f'mag {freq} Hz')
    axes[0].set_ylim([0, 1e-26])
    axes[1].plot(
        freqs_psd, psds[mne.pick_types(evoked_sim.info, meg='grad')].T)
    axes[1].set_title(f'grad {freq} Hz')
    axes[1].set_ylim([0, 1e-23])
    axes[2].plot(freqs_psd, psds[mne.pick_types(evoked_sim.info, eeg=True)].T)
    axes[2].set_title(f'eeg {freq} Hz')
    axes[2].set_ylim([0, 1e-11])


for ax in axes_all.flatten():
    ax.set_xlim([5, 50])


axes_all[1, 0].set_ylabel('Power')
axes_all[0, freqs.size // 2].set_xlabel('Frequency (Hz)')

fig.tight_layout()
fig.show()
