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

import numpy as np
import pylab as pl
import mne
from mne.datasets import sample
from mne.fiff import Evoked

data_path = sample.data_path('.')
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'

setno = 0
snr = 3.0
lambda2 = 1.0 / snr**2
dSPM = True

# Load data
evoked = Evoked(fname_evoked, setno=setno, baseline=(None, 0))
forward = mne.read_forward_solution(fname_fwd)
noise_cov = mne.Covariance()
noise_cov.load(fname_cov)

# Compute inverse solution
stc, K, W = mne.minimum_norm(evoked, forward, noise_cov, orientation='free',
                             method='dspm', snr=3, loose=0.2, pca=True)

# Save result in stc files
lh_vertices = stc['inv']['src'][0]['vertno']
rh_vertices = stc['inv']['src'][1]['vertno']
lh_data = stc['sol'][:len(lh_vertices)]
rh_data = stc['sol'][-len(rh_vertices):]

mne.write_stc('mne_dSPM_inverse-lh.stc', tmin=stc['tmin'], tstep=stc['tstep'],
            vertices=lh_vertices, data=lh_data)
mne.write_stc('mne_dSPM_inverse-rh.stc', tmin=stc['tmin'], tstep=stc['tstep'],
            vertices=rh_vertices, data=rh_data)

###############################################################################
# View activation time-series
times = stc['tmin'] + stc['tstep'] * np.arange(stc['sol'].shape[1])
pl.close('all')
pl.plot(1e3*times, stc['sol'][::100,:].T)
pl.xlabel('time (ms)')
pl.ylabel('dSPM value')
pl.show()
