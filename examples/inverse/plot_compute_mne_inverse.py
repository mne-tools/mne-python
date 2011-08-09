"""
================================================
Compute MNE-dSPM inverse solution on evoked data
================================================

Compute dSPM inverse solution on MNE evoked dataset
and stores the solution in stc files for visualisation.

"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import pylab as pl
from mne.datasets import sample
from mne.fiff import Evoked
from mne.minimum_norm import apply_inverse, read_inverse_operator


data_path = sample.data_path('..')
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'

setno = 0
snr = 3.0
lambda2 = 1.0 / snr ** 2
dSPM = True

# Load data
evoked = Evoked(fname_evoked, setno=setno, baseline=(None, 0))
inverse_operator = read_inverse_operator(fname_inv)

# Compute inverse solution
stc = apply_inverse(evoked, inverse_operator, lambda2, dSPM, pick_normal=False)

# Save result in stc files
stc.save('mne_dSPM_inverse')

###############################################################################
# View activation time-series
pl.plot(1e3 * stc.times, stc.data[::100, :].T)
pl.xlabel('time (ms)')
pl.ylabel('dSPM value')
pl.show()
