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
from mne import fiff

fname = os.environ['MNE_SAMPLE_DATASET_PATH']
fname += '/MEG/sample/sample_audvis-cov.fif'

# Reading
fid, tree, _ = fiff.fiff_open(fname)
cov_type = 1
cov = mne.read_cov(fid, tree, cov_type)
fid.close()

# Writing
mne.write_cov_file('cov.fif', cov)

# print "covariance matrix size: %s x %s" % cov['data'].shape
# 
# ###############################################################################
# # Show covariance
# import pylab as pl
# pl.matshow(cov['data'])
# pl.show()
