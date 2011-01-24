"""
==================================
Reading and writing an evoked file
==================================

"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import os
from mne import fiff

fname = os.environ['MNE_SAMPLE_DATASET_PATH']
fname += '/MEG/sample/sample_audvis-ave.fif'

# Reading
data = fiff.read_evoked(fname, setno=0, baseline=(None, 0))

###############################################################################
# Show result
import pylab as pl
pl.clf()
pl.subplot(3, 1, 1)
pl.plot(data['evoked']['times'], data['evoked']['epochs'][0:306:3,:].T)
pl.title('Planar Gradiometers')
pl.xlabel('time (s)')
pl.ylabel('MEG data (T / m)')
pl.subplot(3, 1, 2)
pl.plot(data['evoked']['times'], data['evoked']['epochs'][1:306:3,:].T)
pl.title('Axial Gradiometers')
pl.xlabel('time (s)')
pl.ylabel('MEG data (T / m)')
pl.subplot(3, 1, 3)
pl.plot(data['evoked']['times'], data['evoked']['epochs'][2:306:3,:].T)
pl.title('Magnetometers')
pl.xlabel('time (s)')
pl.ylabel('MEG data (T)')
pl.subplots_adjust(0.175, 0.08, 0.94, 0.94, 0.2, 0.63)
pl.show()
