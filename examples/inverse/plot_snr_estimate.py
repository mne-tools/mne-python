# -*- coding: utf-8 -*-
"""
============================
Plot an estimate of data SNR
============================

This estimates the SNR as a function of time for a set of data
using two different methods.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from os import path as op

from mne.datasets.sample import data_path
from mne.minimum_norm import read_inverse_operator, apply_inverse
from mne import read_evokeds, read_forward_solution, read_cov
from mne.viz import plot_snr_estimate

print(__doc__)

# Estimate overall SNR of the data
# --------------------------------
data_dir = op.join(data_path(), 'MEG', 'sample')
fname_inv = op.join(data_dir, 'sample_audvis-meg-oct-6-meg-inv.fif')
fname_evoked = op.join(data_dir, 'sample_audvis-ave.fif')

inv = read_inverse_operator(fname_inv)
evoked = read_evokeds(fname_evoked, baseline=(None, 0))[0]

plot_snr_estimate(evoked, inv)

# Estimate the time-varying SNR in source space
# ---------------------------------------------
fname_fwd = op.join(data_dir, 'sample_audvis-meg-oct-6-fwd.fif')
fname_cov = op.join(data_dir, 'sample_audvis-cov.fif')

evoked.crop(0.1, 0.11)
stc = apply_inverse(evoked, inv, 1. / 9., 'MNE')
fwd = read_forward_solution(fname_fwd)
cov = read_cov(fname_cov)
snr_stc = stc.estimate_snr(evoked.info, fwd, cov)
snr_stc.plot()
