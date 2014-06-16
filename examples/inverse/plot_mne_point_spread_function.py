# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

# TODO
# add bad channels?
# other ~linear estimators, beamformers etc.?

print(__doc__)

import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, point_spread_function

## Compute PSFs for labels in MNE sample data set

data_path = sample.data_path()
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_label = [data_path + '/MEG/sample/labels/Vis-rh.label',
               data_path + '/MEG/sample/labels/Vis-lh.label',
               data_path + '/MEG/sample/labels/Aud-rh.label',
               data_path + '/MEG/sample/labels/Aud-lh.label']


# read forward solution (sources in surface-based coordinates)
forward = mne.read_forward_solution(fname_fwd, surf_ori=True)
# read inverse operator
inverse_operator = read_inverse_operator(fname_inv)
# read label(s)
labels = [mne.read_label(ss) for ss in fname_label]

# regularisation parameter
snr = 3.0
lambda2 = 1.0 / snr ** 2

for method in ('MNE', 'dSPM', 'sLORETA'):
    stc_psf, evoked_fwd, label_singval = point_spread_function(inverse_operator,
                                                      forward, method=method,
                                                      labels=labels,
                                                      lambda2=lambda2,
                                                      mode='svd', svd_comp=2)
    fname_out = 'psf_' + method
    print "Writing STC to file: %s" % fname_out
    stc_psf.save(fname_out)

# from mne.datasets import sample
# subjects_dir = sample.data_path() + '/subjects/'
# stc_psf.plot(surface='inflated', hemi='rh', subjects_dir=subjects_dir, time_viewer=True)


