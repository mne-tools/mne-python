"""
===========================
Compute cross-talk functions (CTFs) for labels
for different linear inverse operators
===========================

CTFs are computed for four labels in the MNE sample data set
for three linear inverse operators (MNE, dSPM, sLORETA).
CTFs are saved as STC files for visualization in mne_analyze.
CTFs describe the sensitivity of a linear estimator (e.g. for
one label) to sources across the cortical surface. Sensitivity
to sources outside the label is undesirable, and referred to as
"leakage" or "cross-talk".
"""

# Author: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)

print(__doc__)

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

# In order to get leadfield with fixed source orientation,
# read forward solution with fixed orientations
forward = mne.read_forward_solution(fname_fwd, force_fixed=True, surf_ori=True)

# read label(s)
labels = [mne.read_label(ss) for ss in fname_label]

inverse_operator = read_inverse_operator(fname_inv)

fname_stem = 'ctf'
for method in ('MNE', 'dSPM', 'sLORETA'):
    stc_ctf, label_singvals = cross_talk_function(inverse_operator, forward,
                                                  labels, method=method,
                                                  lambda2=1 / 9., signed=True,
                                                  mode='svd', n_svd_comp=3)

    fname_out = fname_stem + '_' + method
    print("Writing CTFs to files %s" % fname_out)
    stc_ctf.save(fname_out)
