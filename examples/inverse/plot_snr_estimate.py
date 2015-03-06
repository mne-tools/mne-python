# -*- coding: utf-8 -*-
"""
============================
Plot an estimate of data SNR
============================

This estimates the SNR as a function of time for a set of data.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from os import path as op
import matplotlib.pyplot as plt

from mne.datasets.testing import data_path
from mne.minimum_norm import estimate_snr, read_inverse_operator
from mne import read_evokeds

print(__doc__)

data_dir = op.join(data_path(), 'MEG', 'sample')
fname_inv = op.join(data_dir, 'sample_audvis_trunc-meg-eeg-oct-4-meg-inv.fif')
fname_evoked = op.join(data_dir, 'sample_audvis-ave.fif')

inv = read_inverse_operator(fname_inv)
evoked = read_evokeds(fname_evoked)[0]
snr_est = estimate_snr(evoked, inv, verbose=True)

plt.plot(evoked.times, snr_est, color=[0, 0, 1])
plt.show()
