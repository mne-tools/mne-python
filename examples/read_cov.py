"""Reading a noise covariance matrix
"""
print __doc__

import fiff

fname = 'MNE-sample-data/MEG/sample/sample_audvis-cov.fif'

fid, tree, _ = fiff.fiff_open(fname)

cov_type = 1
cov = fiff.read_cov(fid, tree, cov_type)

print "covariance matrix size: %s x %s" % cov['data'].shape

###############################################################################
# Show covariance
import pylab as pl
pl.matshow(cov['data'])
pl.show()
