"""
===========================
Get inverse operator matrix
===========================

Get inverse matrix from an inverse operator for specific parameter settings

"""
# Author: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)

print(__doc__)

import numpy as np
import mne
from mne.datasets import sample
from mne.minimum_norm import cross_talk_function, read_inverse_operator


## Example how to compute CTFs

data_path = sample.data_path()
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'

fname_label = [data_path + '/MEG/sample/labels/Vis-rh.label',
               data_path + '/MEG/sample/labels/Vis-lh.label',
               data_path + '/MEG/sample/labels/Aud-rh.label',
               data_path + '/MEG/sample/labels/Aud-lh.label']

# In order to get leadfield with fixed source orientation, read forward solution again
forward = mne.read_forward_solution(fname_fwd, force_fixed=True, surf_ori=True)

# read label(s)
labels = [mne.read_label(ss) for ss in fname_label]
nr_labels = len(labels)

inverse_operator = read_inverse_operator(fname_inv)

fname_stem = 'ctf_3'
for method in ('MNE', 'dSPM', 'sLORETA'):
    stc_ctf, label_singvals = cross_talk_function(inverse_operator, forward, labels, method=method,
                                  lambda2=1 / 9., mode='svd', svd_comp=3)

    fname_out = fname_stem + '_' + method
    print "Writing CTFs to files %s" % fname_out
    # signed
    stc_ctf.save(fname_out)

    # unsigned
    stc_ctf._data = np.abs( stc_ctf.data )
    stc_ctf.save(fname_out+'_abs')
