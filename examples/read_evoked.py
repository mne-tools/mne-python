"""Reading an evoked file
"""
print __doc__

import fiff

fname = 'sm02a1-ave.fif'
# fname = 'MNE-sample-data/MEG/sample/sample_audvis-ave.fif'

data = fiff.read_evoked(fname)

###############################################################################
# Show result

import pylab as pl
pl.plot(data['evoked']['times'], data['evoked']['epochs'][:306,:].T)
pl.xlabel('time (ms)')
pl.ylabel('MEG data (T)')
pl.show()
