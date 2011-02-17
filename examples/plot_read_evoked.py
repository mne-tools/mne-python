"""
==================================
Reading and writing an evoked file
==================================

"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

from mne import fiff
from mne.datasets import sample
data_path = sample.data_path('.')

fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

# Reading
data = fiff.read_evoked(fname, setno=0, baseline=(None, 0))

###############################################################################
# Show result
import pylab as pl
pl.clf()
pl.subplot(3, 1, 1)
pl.plot(1000*data['evoked']['times'],
        1e13*data['evoked']['epochs'][0:306:3,:].T)
pl.ylim([-200, 200])
pl.title('Planar Gradiometers 1')
pl.xlabel('time (ms)')
pl.ylabel('MEG data (fT/cm)')
pl.subplot(3, 1, 2)
pl.plot(1000*data['evoked']['times'],
        1e13*data['evoked']['epochs'][1:306:3,:].T)
pl.ylim([-200, 200])
pl.title('Planar Gradiometers 2')
pl.xlabel('time (ms)')
pl.ylabel('MEG data (fT/cm)')
pl.subplot(3, 1, 3)
pl.plot(1000*data['evoked']['times'],
        1e15*data['evoked']['epochs'][2:306:3,:].T)
pl.ylim([-600, 600])
pl.title('Magnetometers')
pl.xlabel('time (ms)')
pl.ylabel('MEG data (fT)')
pl.subplots_adjust(0.175, 0.08, 0.94, 0.94, 0.2, 0.63)
pl.show()
