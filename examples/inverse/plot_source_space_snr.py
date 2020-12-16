# -*- coding: utf-8 -*-
"""
===============================
Computing source space SNR
===============================

This example shows how to compute and plot source space SNR as in
:footcite:`GoldenholzEtAl2009`.
"""
# Author: Padma Sundaram <tottochan@gmail.com>
#         Kaisu Lankinen <klankinen@mgh.harvard.edu>
#
# License: BSD (3-clause)

# sphinx_gallery_thumbnail_number = 2

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse
import numpy as np
import matplotlib.pyplot as plt

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'

# Read data
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
evoked = mne.read_evokeds(fname_evoked, condition='Left Auditory',
                          baseline=(None, 0))
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'
fwd = mne.read_forward_solution(fname_fwd)
cov = mne.read_cov(fname_cov)

# Read inverse operator:
inv_op = make_inverse_operator(evoked.info, fwd, cov, fixed=True, verbose=True)

# Calculate MNE:
snr = 3.0
lambda2 = 1.0 / snr ** 2
stc = apply_inverse(evoked, inv_op, lambda2, 'MNE', verbose=True)

# Calculate SNR in source space:
snr_stc = stc.estimate_snr(evoked.info, fwd, cov)

# Plot an average SNR across source points over time:
ave = np.mean(snr_stc.data, axis=0)

fig, ax = plt.subplots()
ax.plot(evoked.times, ave)
ax.set(xlabel='Time (sec)', ylabel='SNR MEG-EEG')
fig.tight_layout()

# Find time point of maximum SNR
maxidx = np.argmax(ave)

# Plot SNR on source space at the time point of maximum SNR:
kwargs = dict(initial_time=evoked.times[maxidx], hemi='split',
              views=['lat', 'med'], subjects_dir=subjects_dir, size=(600, 600),
              clim=dict(kind='value', lims=(-100, -70, -40)),
              transparent=True, colormap='viridis')
brain = snr_stc.plot(**kwargs)

###############################################################################
# EEG
# ---
# Next we do the same for EEG and plot the result on the cortex:

evoked_eeg = evoked.copy().pick_types(eeg=True, meg=False)
inv_op_eeg = make_inverse_operator(evoked_eeg.info, fwd, cov, fixed=True,
                                   verbose=True)
stc_eeg = apply_inverse(evoked_eeg, inv_op_eeg, lambda2, 'MNE', verbose=True)
snr_stc_eeg = stc_eeg.estimate_snr(evoked_eeg.info, fwd, cov)
brain = snr_stc_eeg.plot(**kwargs)

###############################################################################
# The same can be done for MEG, which looks more similar to the MEG-EEG case
# than the EEG case does.
#
# References
# ----------
# .. footbibliography::
