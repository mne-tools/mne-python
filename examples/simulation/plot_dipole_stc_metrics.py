"""
===========================================================
Compare simulated and estimated dipole-like source activity
===========================================================

This example illustrates how to compare the simulated and estimated
stcs by computing different metrics. Simulated source is a cortical region.
It is meant to be a brief introduction and only highlights the simplest
use case.

"""
# Author: Kostiantyn Maksymenko <kostiantyn.maksymenko@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.simulation import (stc_peak_position_error,
                            stc_spacial_deviation)

print(__doc__)

np.random.seed(42)

# Import sample data
data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'sample'
evoked_fname = op.join(data_path, 'MEG', subject, 'sample_audvis-ave.fif')
info = mne.io.read_info(evoked_fname)
tstep = 1. / info['sfreq']

# Import forward operator and source space
fwd_fname = op.join(data_path, 'MEG', subject,
                    'sample_audvis-meg-eeg-oct-6-fwd.fif')
fwd = mne.read_forward_solution(fwd_fname)
src = fwd['src']

# Select a dipole on the caudal middle frontal cortex
selected_label = mne.read_labels_from_annot(
    subject, regexp='caudalmiddlefrontal-lh', subjects_dir=subjects_dir)[0]

location = 1915
extent = 0.  # One dipole source
label = mne.label.select_sources(
    subject, selected_label, location=location, extent=extent,
    subjects_dir=subjects_dir)

# Define the time course of the activity
source_time_series = np.sin(2. * np.pi * 18. * np.arange(100) * tstep) * 10e-9

# Define when the activity occurs using events.
n_events = 50
events = np.zeros((n_events, 3))
events[:, 0] = 200 * np.arange(n_events)  # Events sample.
events[:, 2] = 1  # All events have the sample id.

# Create simulated source activity.
source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)
source_simulator.add_data(label, source_time_series, events)

# Simulate raw data with sensor noise
raw = mne.simulation.simulate_raw(info, source_simulator, forward=fwd)
raw = raw.pick_types(meg=False, eeg=True, stim=True)
cov = mne.make_ad_hoc_cov(raw.info)
mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.04])

# Compute evoked from raw
events = mne.find_events(raw)
tmax = (len(source_time_series) - 1) * tstep
epochs = mne.Epochs(raw, events, 1, tmin=0, tmax=tmax)
evoked = epochs.average()

# Create backgraund truth stc
stc_true = source_simulator.get_stc(start_sample=0,
                                    stop_sample=len(source_time_series))

# Compute inverse solution
snr = 3.0
inv_method = 'sLORETA'
lambda2 = 1.0 / snr ** 2

inverse_operator = make_inverse_operator(evoked.info, fwd, cov,
                                         loose='auto', depth=0.8,
                                         fixed=True)

stc_est = apply_inverse(evoked, inverse_operator, lambda2, inv_method,
                        pick_ori=None)

thresholds = ['0%', '10%', '20%', '30%', '50%', '70%', '80%', '90%',
              '95%', '99%']

y_mean = np.zeros(len(thresholds))
y_std = np.zeros(len(thresholds))
x = range(len(y_mean))
for i, thr in enumerate(thresholds):
    y_mean[i] = stc_peak_position_error(stc_true, stc_est, src, threshold=thr,
                                        per_sample=False)
    y_std[i] = stc_spacial_deviation(stc_true, stc_est, src, threshold=thr,
                                     per_sample=False)
# Visualization

# PPE
f1, ax1 = plt.subplots()
ax1.plot(x, y_mean, '.-')
ax1.set_title('PPE')
ax1.set_ylabel('Error')
ax1.set_xticks(x)
ax1.set_xticklabels(thresholds)

# SD
f2, ax2 = plt.subplots()
ax2.plot(x, y_std, '.-')
ax2.set_title('SD')
ax2.set_ylabel('Error')
ax2.set_xticks(x)
ax2.set_xticklabels(thresholds)
