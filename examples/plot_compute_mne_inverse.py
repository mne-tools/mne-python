"""
=================================
Compute MNE-dSPM inverse solution
=================================

Compute dSPM inverse solution on MNE sample data set
and stores the solution in stc files for visualisation.

"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import os
import numpy as np
import pylab as pl
import mne

fname_inv = os.environ['MNE_SAMPLE_DATASET_PATH']
fname_inv += '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_data = os.environ['MNE_SAMPLE_DATASET_PATH']
fname_data += '/MEG/sample/sample_audvis-ave.fif'

setno = 0
snr = 3.0
lambda2 = 1.0 / snr**2
dSPM = True

res = mne.compute_inverse(fname_data, setno, fname_inv, lambda2, dSPM,
                          baseline=(None, 0))

lh_vertices = res['inv']['src'][0]['vertno']
rh_vertices = res['inv']['src'][1]['vertno']
lh_data = res['sol'][:len(lh_vertices)]
rh_data = res['sol'][len(rh_vertices):]

# Save result in stc files
mne.write_stc('mne_dSPM_inverse-lh.stc', tmin=res['tmin'], tstep=res['tstep'],
               vertices=lh_vertices, data=lh_data)
mne.write_stc('mne_dSPM_inverse-rh.stc', tmin=res['tmin'], tstep=res['tstep'],
               vertices=rh_vertices, data=rh_data)

###############################################################################
# View activation time-series
times = res['tmin'] + res['tstep'] * np.arange(lh_data.shape[1])
pl.plot(times, res['sol'][::100,:].T)
pl.xlabel('time (ms)')
pl.ylabel('Source amplitude')
pl.show()
