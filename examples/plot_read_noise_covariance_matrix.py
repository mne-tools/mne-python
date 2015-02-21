"""
=========================================
Reading/Writing a noise covariance matrix
=========================================

Plot a noise covariance matrix.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample
import matplotlib.pyplot as plt

print(__doc__)

data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

cov = mne.Covariance(fname)
print(cov)

###############################################################################
# Show covariance

# Note: if you have the measurement info you can use mne.viz.plot_cov

plt.matshow(cov.data, cmap='RdBu_r')
plt.title('Noise covariance matrix (%d channels)' % cov.data.shape[0])
plt.show()
