"""
=========================================
Reading/Writing a noise covariance matrix
=========================================

Plot a noise covariance matrix.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

from os import path as op
import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_evo = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')

cov = mne.read_cov(fname_cov)
print(cov)
evoked = mne.read_evokeds(fname_evo)[0]

###############################################################################
# Show covariance

cov.plot(evoked.info, exclude='bads', show_svd=False)
