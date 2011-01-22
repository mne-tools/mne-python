"""
============================
Compute MNE inverse solution
============================
"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

print __doc__

import os
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

# XXX : kind of ugly
import numpy as np
res['vertices'] = np.r_[res['inv']['src'][0]['vertno']]
# res['vertices'] = np.r_[res['inv']['src'][0]['vertno'],
#                         res['inv']['src'][1]['vertno']]
# res['data'] = res['sol']
res['data'] = res['sol'][:len(res['vertices'])]

# Save result in stc file
mne.write_stc('mne_dSPM_inverse-lh.stc', res)

import pylab as pl
pl.plot(res['sol'][::100,:].T)
pl.xlabel('time (ms)')
pl.ylabel('Source amplitude')
pl.show()
