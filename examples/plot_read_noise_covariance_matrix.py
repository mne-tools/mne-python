"""
=========================================
Reading/Writing a noise covariance matrix
=========================================
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne.datasets import sample

data_path = sample.data_path('.')
fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

cov = mne.Covariance(fname)
print cov

###############################################################################
# Show covariance

# Note: if you have the measurement info you can use mne.viz.plot_cov

import pylab as pl
pl.matshow(cov.data)
pl.title('Noise covariance matrix (%d channels)' % cov.data.shape[0])
pl.show()
