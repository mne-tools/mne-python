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
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'

setno = 0
snr = 3.0
lambda2 = 1.0 / snr ** 2
dSPM = True

# Load data
evoked = Evoked(fname_evoked, setno=setno, baseline=(None, 0))
inverse_operator = mne.read_inverse_operator(fname_inv)

# Compute inverse solution
res = mne.apply_inverse(evoked, inverse_operator, lambda2, dSPM)

# Save result in stc files
lh_vertices = res['inv']['src'][0]['vertno']
rh_vertices = res['inv']['src'][1]['vertno']
lh_data = res['sol'][:len(lh_vertices)]
rh_data = res['sol'][-len(rh_vertices):]

mne.write_stc('mne_dSPM_inverse-lh.stc', tmin=res['tmin'], tstep=res['tstep'],
               vertices=lh_vertices, data=lh_data)
mne.write_stc('mne_dSPM_inverse-rh.stc', tmin=res['tmin'], tstep=res['tstep'],
               vertices=rh_vertices, data=rh_data)

###############################################################################
# View activation time-series
times = res['tmin'] + res['tstep'] * np.arange(lh_data.shape[1])
pl.plot(1e3 * times, res['sol'][::100, :].T)
pl.xlabel('time (ms)')
pl.ylabel('dSPM value')
pl.show()
