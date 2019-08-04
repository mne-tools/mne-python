"""
.. _ex-inverse-source-power:

=========================================
Compute source power using DICS beamfomer
=========================================

Compute a Dynamic Imaging of Coherent Sources (DICS) [1]_ filter from
single-trial activity to estimate source power across a frequency band. This
example demonstrates how to source localize the event-related synchronization
(ERS) of beta band activity in this dataset: :ref:`somato-dataset`

References
----------
.. [1] Gross et al. Dynamic imaging of coherent sources: Studying neural
       interactions in the human brain. PNAS (2001) vol. 98 (2) pp. 694-699
"""
# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
#         Roman Goj <roman.goj@gmail.com>
#         Denis Engemann <denis.engemann@gmail.com>
#         Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op

import numpy as np
import mne
from mne.datasets import somato
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd

print(__doc__)

###############################################################################
# Reading the raw data and creating epochs:
data_path = somato.data_path()
subject = '01'
task = 'somato'
raw_fname = op.join(data_path, 'sub-{}'.format(subject), 'meg',
                    'sub-{}_task-{}_meg.fif'.format(subject, task))

raw = mne.io.read_raw_fif(raw_fname)

# Set picks, use a single sensor type
picks = mne.pick_types(raw.info, meg='grad', exclude='bads')

# Read epochs
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id=1, tmin=-1.5, tmax=2, picks=picks,
                    preload=True)

# Read forward operator and point to freesurfer subject directory
fname_fwd = op.join(data_path, 'derivatives', 'sub-{}'.format(subject),
                    'sub-{}_task-{}-fwd.fif'.format(subject, task))
subjects_dir = op.join(data_path, 'derivatives', 'freesurfer', 'subjects')

fwd = mne.read_forward_solution(fname_fwd)

###############################################################################
# We are interested in the beta band. Define a range of frequencies, using a
# log scale, from 12 to 30 Hz.
freqs = np.logspace(np.log10(12), np.log10(30), 9)

###############################################################################
# Computing the cross-spectral density matrix for the beta frequency band, for
# different time intervals. We use a decim value of 20 to speed up the
# computation in this example at the loss of accuracy.
csd = csd_morlet(epochs, freqs, tmin=-1, tmax=1.5, decim=20)
csd_baseline = csd_morlet(epochs, freqs, tmin=-1, tmax=0, decim=20)
# ERS activity starts at 0.5 seconds after stimulus onset
csd_ers = csd_morlet(epochs, freqs, tmin=0.5, tmax=1.5, decim=20)

###############################################################################
# Computing DICS spatial filters using the CSD that was computed on the entire
# timecourse.
filters = make_dics(epochs.info, fwd, csd.mean(), pick_ori='max-power')

###############################################################################
# Applying DICS spatial filters separately to the CSD computed using the
# baseline and the CSD computed during the ERS activity.
baseline_source_power, freqs = apply_dics_csd(csd_baseline.mean(), filters)
beta_source_power, freqs = apply_dics_csd(csd_ers.mean(), filters)

###############################################################################
# Visualizing source power during ERS activity relative to the baseline power.
stc = beta_source_power / baseline_source_power
message = 'DICS source power in the 12-30 Hz frequency band'
brain = stc.plot(hemi='both', views='par', subjects_dir=subjects_dir,
                 subject=subject, time_label=message)
