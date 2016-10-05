"""
================================================================
Compute sparse inverse solution with mixed norm: MxNE and irMxNE
================================================================

Runs (ir)MxNE (L1/L2 or L0.5/L2 mixed norm) inverse solver.
L0.5/L2 is done with irMxNE which allows for sparser
source estimates with less amplitude bias due to the non-convexity
of the L0.5/L2 mixed norm penalty.

See
Gramfort A., Kowalski M. and Hamalainen, M.:
Mixed-norm estimates for the M/EEG inverse problem using accelerated
gradient methods, Physics in Medicine and Biology, 2012
http://dx.doi.org/10.1088/0031-9155/57/7/1937

Strohmeier D., Haueisen J., and Gramfort A.:
The iterative reweighted Mixed-Norm Estimate for
spatio-temporal MEG/EEG source reconstruction,
IEEE Transactions on Medical Imaging, 2016
DOI: 10.1109/TMI.2016.2553445
"""
# Author: Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD (3-clause)

import numpy as np

import matplotlib.pyplot as plt

from nilearn.plotting import plot_stat_map, plot_glass_brain
from nilearn.image import index_img

import mne
from mne.datasets import sample
from mne.inverse_sparse import mixed_norm
from mne.minimum_norm import make_inverse_operator, apply_inverse

print(__doc__)

data_path = sample.data_path()
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-vol-7-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'

# Read noise covariance matrix
cov = mne.read_cov(cov_fname)

# Handling average file
condition = 'Left Auditory'
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
evoked.pick_types(meg=True, eeg=False)
evoked.crop(tmin=0, tmax=0.2)

# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)

ylim = dict(eeg=[-10, 10], grad=[-400, 400], mag=[-600, 600])
evoked.plot(ylim=ylim, proj=True)

###############################################################################
# Run solver
alpha = 50.  # regularization parameter between 0 and 100 (100 is high)
loose, depth = 1.0, 0.3  # loose orientation & depth weighting
n_mxne_iter = 50  # if > 1 use L0.5/L2 reweighted mixed norm solver
# if n_mxne_iter > 1 dSPM weighting can be avoided.

# Compute dSPM solution to be used as weights in MxNE
inverse_operator = make_inverse_operator(evoked.info, forward, cov,
                                         loose=loose, depth=depth)
stc_dspm = apply_inverse(evoked, inverse_operator, lambda2=1. / 9.,
                         method='dSPM')

# Compute (ir)MxNE inverse solution
stc, residual = mixed_norm(evoked, forward, cov, alpha, loose=loose,
                           depth=depth, maxit=3000, tol=1e-6,
                           active_set_size=10, debias=True, weights=stc_dspm,
                           weights_min=4., n_mxne_iter=n_mxne_iter,
                           return_residual=True)

# Show residual
residual.plot(ylim=ylim, proj=True)

###############################################################################
# View source estimate as an overlay on the structural image
stc.save('mxne_aud-vol')

img = mne.save_stc_as_volume('mxne_inverse_aud.nii.gz', stc,
                             forward['src'], mri_resolution=False)

t1_fname = data_path + '/subjects/sample/mri/T1.mgz'

tidx = np.argmin(np.abs(stc.times - 0.080))
_img = index_img(img, tidx)
plot_glass_brain(_img, colorbar=True, threshold=1e-10,
                 title='MxNE (t=%.3f s.)' % stc.times[tidx])
plot_stat_map(_img, t1_fname, threshold=1e-10,
              title='MxNE (t=%.3f s.)' % stc.times[tidx])

tidx = np.argmin(np.abs(stc.times - 0.102))
_img = index_img(img, tidx)
plot_glass_brain(_img, colorbar=True, threshold=1e-10,
                 title='MxNE (t=%.3f s.)' % stc.times[tidx])
plot_stat_map(_img, t1_fname, threshold=1e-10,
              title='MxNE (t=%.3f s.)' % stc.times[tidx])

# plot source time courses
plt.figure()
plt.plot(stc.times, stc.data.T * 1e9)
plt.xlabel('Time (ms)')
plt.ylabel('Ampltidue (nAm)')
plt.show()
