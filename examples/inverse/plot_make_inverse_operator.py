"""
===============================================================
Assemble inverse operator and compute MNE-dSPM inverse solution
===============================================================

Assemble an EEG inverse operator and compute dSPM inverse solution
on MNE evoked dataset and stores the solution in stc files for
visualisation.

"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import pylab as pl
import mne
from mne.datasets import sample
from mne.fiff import Evoked
from mne.minimum_norm import make_inverse_operator, apply_inverse

data_path = sample.data_path('..')
fname_fwd = data_path + '/MEG/sample/sample_audvis-eeg-oct-6-fwd.fif'
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'

setno = 0
snr = 3.0
lambda2 = 1.0 / snr ** 2
dSPM = True

# Load data
evoked = Evoked(fname_evoked, setno=setno, baseline=(None, 0))
forward = mne.read_forward_solution(fname_fwd, surf_ori=True)
noise_cov = mne.Covariance(fname_cov)
inverse_operator = make_inverse_operator(evoked.info, forward, noise_cov,
                                         loose=0.2, depth=0.8)

# Compute inverse solution
stc = apply_inverse(evoked, inverse_operator, lambda2, dSPM, pick_normal=False)

# Save result in stc files
stc.save('mne_dSPM_inverse')

###############################################################################
# View activation time-series
pl.close('all')
pl.plot(1e3 * stc.times, stc.data[::150, :].T)
pl.xlabel('time (ms)')
pl.ylabel('dSPM value')
pl.show()
