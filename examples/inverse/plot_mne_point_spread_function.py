"""
==========================================================
Compute point-spread functions (PSFs) for MNE/dSPM/sLORETA
==========================================================

PSFs are computed for four labels in the MNE sample data set
for linear inverse operators (MNE, dSPM, sLORETA).
PSFs describe the spread of activation from one label
across the cortical surface.
"""

# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

print(__doc__)

import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, point_spread_function

data_path = sample.data_path()
subjects_dir = data_path + '/subjects/'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_label = [data_path + '/MEG/sample/labels/Aud-rh.label',
               data_path + '/MEG/sample/labels/Aud-lh.label',
               data_path + '/MEG/sample/labels/Vis-rh.label',
               data_path + '/MEG/sample/labels/Vis-lh.label']


# read forward solution (sources in surface-based coordinates)
forward = mne.read_forward_solution(fname_fwd, force_fixed=False,
                                    surf_ori=True)

# read inverse operator
inverse_operator = read_inverse_operator(fname_inv)
# read label(s)
labels = [mne.read_label(ss) for ss in fname_label]

# regularisation parameter
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = 'MNE'  # can be 'MNE' or 'sLORETA'
mode = 'svd'
n_svd_comp = 1

stc_psf, psf_evoked, singvals = point_spread_function(inverse_operator,
                                                      forward,
                                                      method=method,
                                                      labels=labels,
                                                      lambda2=lambda2,
                                                      pick_ori='normal',
                                                      mode=mode,
                                                      n_svd_comp=n_svd_comp)

fmax = stc_psf.data[:, 0].max()
fmid = fmax / 2.
fmin = 0.

time_label = "Label %d"

brain = stc_psf.plot(surface='inflated', hemi='rh', subjects_dir=subjects_dir,
                     time_label=time_label, fmin=fmin, fmid=fmid, fmax=fmax)

# Save PSFs for visualization in mne_analyze.
#fname_out = 'psf_' + method
#print("Writing STC to file: %s" % fname_out)
#stc_psf.save(fname_out)
