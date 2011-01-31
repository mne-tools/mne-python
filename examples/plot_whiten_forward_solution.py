"""
========================================================
Whiten a forward operator with a noise covariance matrix
========================================================
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import os
import mne
from mne import fiff

fwd_fname = os.environ['MNE_SAMPLE_DATASET_PATH']
fwd_fname += '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = os.environ['MNE_SAMPLE_DATASET_PATH']
ave_fname += '/MEG/sample/sample_audvis-ave.fif'
cov_fname = os.environ['MNE_SAMPLE_DATASET_PATH']
cov_fname += '/MEG/sample/sample_audvis-cov.fif'

# Reading
ave = fiff.read_evoked(ave_fname, setno=0, baseline=(None, 0))
fwd = mne.read_forward_solution(fwd_fname)

cov = mne.Covariance()
cov.load(cov_fname)

ave_whiten, fwd_whiten, W = cov.whiten_evoked_and_forward(ave, fwd, eps=0.2)

leadfield = fwd_whiten['sol']['data']

print "Leadfield size : %d x %d" % leadfield.shape

###############################################################################
# Show result
import pylab as pl
pl.matshow(leadfield[:306,:500])
pl.xlabel('sources')
pl.ylabel('sensors')
pl.title('Lead field matrix')
pl.show()
