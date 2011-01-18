"""
============================
Compute MNE inverse solution
============================
"""
print __doc__

import mne

fname_inv = 'MNE-sample-data/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_data = 'MNE-sample-data/MEG/sample/sample_audvis-ave.fif'

setno = 0
lambda2 = 10
dSPM = True

res = mne.compute_inverse(fname_data, setno, fname_inv, lambda2, dSPM)

import pylab as pl
pl.plot(res['sol'][::100,:].T)
pl.xlabel('time (ms)')
pl.ylabel('Source amplitude')
pl.show()
