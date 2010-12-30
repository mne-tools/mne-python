"""Reading and writing an evoked file
"""
print __doc__

import fiff

fname = 'MNE-sample-data/MEG/sample/sample_audvis-ave.fif'

# Reading
data = fiff.read_evoked(fname)

# Writing
fiff.write_evoked('evoked.fif', data)

###############################################################################
# Show result

import pylab as pl
pl.plot(data['evoked']['times'], data['evoked']['epochs'][:306,:].T)
pl.xlabel('time (ms)')
pl.ylabel('MEG data (T)')
pl.show()
