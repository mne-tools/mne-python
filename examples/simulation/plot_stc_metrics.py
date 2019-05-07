"""
===============================================
Compare simulated and estimated source activity
===============================================

This example illustrates how to use :mod:`mne.simulation.metrics`
module to compare the simulated and estimated stcs by computing different
metrics. It is meant to be a brief introduction and only highlights
the simplest use case.

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
from mne.simulation import (stc_dipole_localization_error,
                            stc_f1_score, stc_precision_score,
                            stc_recall_score, stc_cosine_score)

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

# To select a region to activate, we use the caudal middle frontal to grow
# a region of interest.
selected_label = mne.read_labels_from_annot(
    subject, regexp='caudalmiddlefrontal-lh', subjects_dir=subjects_dir)[0]
location = 'center'  # Use the center of the region as a seed.
extent = 20.  # Extent in mm of the region.
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
snr = 30.0
inv_method = 'sLORETA'
lambda2 = 1.0 / snr ** 2

inverse_operator = make_inverse_operator(evoked.info, fwd, cov,
                                         loose='auto', depth=0.8,
                                         fixed=True)

stc_est = apply_inverse(evoked, inverse_operator, lambda2, inv_method,
                        pick_ori=None)

# Compute performance scores for different source amplitude thresholds
thresholds = ['30%', '50%', '70%', '80%', '90%', '95%']
dles = []
f1s = []
precs = []
recs = []
for thr in thresholds:
    dles.append(stc_dipole_localization_error(stc_true, stc_est, src,
                                              threshold=thr,
                                              per_sample=False))
    f1s.append(stc_f1_score(stc_true, stc_est, threshold=thr,
                            per_sample=False))
    precs.append(stc_precision_score(stc_true, stc_est, threshold=thr,
                                     per_sample=False))
    recs.append(stc_recall_score(stc_true, stc_est, threshold=thr,
                                 per_sample=False))

# Visualization scores
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col')
f.suptitle('Performance scores per threshold')

ax1.plot(dles, '.-')
ax1.set_title('DLE')
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 1))
ax1.set_ylabel('Score')

ax2.plot(f1s, '.-')
ax2.set_title('F1 score')
ax2.set_ylim([0, 1])

ax3.plot(precs, '.-')
ax3.set_xlabel('Threshold')
ax3.set_title('Precision')
ax3.set_xticks(range(6))
ax3.set_xticklabels(thresholds)
ax3.set_ylim([0, 1])
ax3.set_ylabel('Score')

ax4.plot(recs, '.-')
ax4.set_xlabel('Threshold')
ax4.set_title('Recall')
ax4.set_xticks(range(6))
ax4.set_xticklabels(thresholds)
ax4.set_ylim([0, 1])

# Cosine score
plt.figure()
plt.plot(stc_true.times, stc_cosine_score(stc_true, stc_est))
plt.xlabel('Time')
plt.ylabel('Score')
plt.title('Cosine score')
plt.show()
