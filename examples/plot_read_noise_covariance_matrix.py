"""
=========================================
Reading/Writing a noise covariance matrix
=========================================
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import os
import mne

fname = os.environ['MNE_SAMPLE_DATASET_PATH']
fname += '/MEG/sample/sample_audvis-cov.fif'

cov = mne.Covariance(kind='full')
cov.load(fname)

print cov

###############################################################################
# Show covariance
import pylab as pl
pl.matshow(cov.data)
pl.show()
