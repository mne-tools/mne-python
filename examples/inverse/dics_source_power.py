# -*- coding: utf-8 -*-
"""
.. _ex-inverse-source-power:

==========================================
Compute source power using DICS beamformer
==========================================

Compute a Dynamic Imaging of Coherent Sources (DICS) :footcite:`GrossEtAl2001`
filter from single-trial activity to estimate source power across a frequency
band. This example demonstrates how to source localize the event-related
synchronization (ERS) of beta band activity in the
:ref:`somato dataset <somato-dataset>`.
"""
# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
#         Roman Goj <roman.goj@gmail.com>
#         Denis Engemann <denis.engemann@gmail.com>
#         Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause

# %%

import numpy as np
import mne
from mne.datasets import somato
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd

print(__doc__)

# %%
# Reading the raw data and creating epochs:
data_path = somato.data_path()
subject = '01'
task = 'somato'
raw_fname = (data_path / f'sub-{subject}' / 'meg' /
             f'sub-{subject}_task-{task}_meg.fif')

# Use a shorter segment of raw just for speed here
raw = mne.io.read_raw_fif(raw_fname)
raw.crop(0, 120)  # one minute for speed (looks similar to using all ~800 sec)

# Read epochs
events = mne.find_events(raw)

epochs = mne.Epochs(raw, events, event_id=1, tmin=-1.5, tmax=2, preload=True)
del raw

# Paths to forward operator and FreeSurfer subject directory
fname_fwd = (data_path / 'derivatives' / f'sub-{subject}' /
             f'sub-{subject}_task-{task}-fwd.fif')

subjects_dir = data_path / 'derivatives' / 'freesurfer' / 'subjects'

# %%
# We are interested in the beta band. Define a range of frequencies, using a
# log scale, from 12 to 30 Hz.
freqs = np.logspace(np.log10(12), np.log10(30), 9)

# %%
# Computing the cross-spectral density matrix for the beta frequency band, for
# different time intervals. We use a decim value of 20 to speed up the
# computation in this example at the loss of accuracy.
csd = csd_morlet(epochs, freqs, tmin=-1, tmax=1.5, decim=20)
csd_baseline = csd_morlet(epochs, freqs, tmin=-1, tmax=0, decim=20)
# ERS activity starts at 0.5 seconds after stimulus onset
csd_ers = csd_morlet(epochs, freqs, tmin=0.5, tmax=1.5, decim=20)
info = epochs.info
del epochs

# %%
# To compute the source power for a frequency band, rather than each frequency
# separately, we average the CSD objects across frequencies.
csd = csd.mean()
csd_baseline = csd_baseline.mean()
csd_ers = csd_ers.mean()

# %%
# Computing DICS spatial filters using the CSD that was computed on the entire
# timecourse.
fwd = mne.read_forward_solution(fname_fwd)
filters = make_dics(info, fwd, csd, noise_csd=csd_baseline,
                    pick_ori='max-power', reduce_rank=True, real_filter=True)
del fwd

# %%
# Applying DICS spatial filters separately to the CSD computed using the
# baseline and the CSD computed during the ERS activity.
baseline_source_power, freqs = apply_dics_csd(csd_baseline, filters)
beta_source_power, freqs = apply_dics_csd(csd_ers, filters)

# %%
# Visualizing source power during ERS activity relative to the baseline power.
stc = beta_source_power / baseline_source_power
message = 'DICS source power in the 12-30 Hz frequency band'
brain = stc.plot(hemi='both', views='axial', subjects_dir=subjects_dir,
                 subject=subject, time_label=message)

# %%
# References
# ----------
# .. footbibliography::
