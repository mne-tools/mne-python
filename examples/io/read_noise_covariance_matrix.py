# -*- coding: utf-8 -*-
"""
.. _ex-read-noise-cov:

=========================================
Reading/Writing a noise covariance matrix
=========================================

How to plot a noise covariance matrix.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

# %%

import mne
from mne.datasets import sample

data_path = sample.data_path()
fname_cov = data_path / 'MEG' / 'sample' / 'sample_audvis-cov.fif'
fname_evo = data_path / 'MEG' / 'sample' / 'sample_audvis-ave.fif'

cov = mne.read_cov(fname_cov)
print(cov)
ev_info = mne.io.read_info(fname_evo)

# %%
# Plot covariance

cov.plot(ev_info, exclude='bads', show_svd=False)
