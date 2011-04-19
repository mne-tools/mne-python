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
import mne
from mne.datasets import sample
from mne.fiff import Evoked
from mne.minimum_norm import minimum_norm

data_path = sample.data_path('.')
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'

setno = 0
snr = 3.0
lambda2 = 1.0 / snr ** 2
dSPM = True

# Load data
evoked = Evoked(fname_evoked, setno=setno, baseline=(None, 0))
forward = mne.read_forward_solution(fname_fwd)
noise_cov = mne.Covariance(fname_cov)

# Compute whitener from noise covariance matrix
whitener = noise_cov.get_whitener(evoked.info, mag_reg=0.1,
                                  grad_reg=0.1, eeg_reg=0.1, pca=True)
# Compute inverse solution
stc = minimum_norm(evoked, forward, whitener, orientation='loose',
                             method='dspm', snr=3, loose=0.2)

# Save result in stc files
stc.save('mne_dSPM_inverse')

###############################################################################
# View activation time-series
pl.close('all')
pl.plot(1e3 * stc.times, stc.data[::100, :].T)
pl.xlabel('time (ms)')
pl.ylabel('dSPM value')
pl.show()
