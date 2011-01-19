"""
==================================
Reading and writing an evoked file
==================================

"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

print __doc__

import os
from mne import fiff

fname = os.environ['MNE_SAMPLE_DATASET_PATH']
fname += '/MEG/sample/sample_audvis-ave.fif'

# Reading
data = fiff.read_evoked(fname)

# Writing
fiff.write_evoked('evoked.fif', data)

###############################################################################
# Show result

import pylab as pl
pl.plot(data['evoked']['times'], data['evoked']['epochs'][:306,:].T)
pl.xlabel('time (s)')
pl.ylabel('MEG data (T)')
pl.show()
