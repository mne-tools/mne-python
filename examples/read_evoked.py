"""Reading an evoked file
"""
print __doc__

import fiff

fname = 'MNE-sample-data/MEG/sample/sample_audvis-ave.fif'

data = fiff.read_evoked(fname)

fiff.write_evoked('evoked.fif', data)
data2 = fiff.read_evoked('evoked.fif')

from scipy import linalg
print linalg.norm(data['evoked']['epochs'] - data2['evoked']['epochs'])

###############################################################################
# Show result

import pylab as pl
pl.plot(data['evoked']['times'], data['evoked']['epochs'][:306,:].T)
pl.xlabel('time (ms)')
pl.ylabel('MEG data (T)')
pl.show()
