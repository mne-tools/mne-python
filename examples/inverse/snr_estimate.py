# -*- coding: utf-8 -*-
"""
.. _ex-snr-estimate:

==================================
Estimate data SNR using an inverse
==================================

This estimates the SNR as a function of time for a set of data
using a minimum-norm inverse operator.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

# %%

from mne.datasets.sample import data_path
from mne.minimum_norm import read_inverse_operator
from mne import read_evokeds
from mne.viz import plot_snr_estimate

print(__doc__)

data_dir = data_path() / 'MEG' / 'sample'
fname_inv = data_dir / 'sample_audvis-meg-oct-6-meg-inv.fif'
fname_evoked = data_dir / 'sample_audvis-ave.fif'

inv = read_inverse_operator(fname_inv)
evoked = read_evokeds(fname_evoked, baseline=(None, 0))[0]

plot_snr_estimate(evoked, inv)
