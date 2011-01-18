"""
============================
Compute MNE inverse solution
============================
"""
print __doc__

from mne import fiff

fname_inv = 'MNE-sample-data/MEG/sample/sample_audvis-ave-7-meg-inv.fif'
fname_data = 'MNE-sample-data/MEG/sample/sample_audvis-ave.fif'

# inv = fiff.read_inverse_operator(fname)
setno = 0
lambda2 = 10
dSPM = True

res = fiff.compute_inverse(fname_data, setno, fname_inv, lambda2, dSPM)

import pylab as pl
pl.plot(res['sol'][::100,:].T)
pl.xlabel('time (s)')
pl.ylabel('Source amplitude')
pl.show()
