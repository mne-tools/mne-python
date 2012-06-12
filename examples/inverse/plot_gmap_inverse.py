"""
=============================================================
Compute sparse solver based on hierarchical Bayes (gamma MAP)
=============================================================

See
Wipf et al. Analysis of Empirical Bayesian Methods for 
Neuroelectromagnetic Source Localization. 
Advances in Neural Information Processing Systems (2007)
"""
# Author: Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#         Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
# License: BSD (3-clause)

print __doc__

import mne
from mne import fiff
from mne.datasets import sample
from mne.minimum_norma import gamma_map
from mne.viz import plot_sparse_source_estimates

import numpy as np

data_path = sample.data_path('/Users/alex/work/data/MNE-sample-data')
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

# Read noise covariance matrix
cov = mne.read_cov(cov_fname)
# Handling average file
evoked = fiff.read_evoked(ave_fname, setno=0, baseline=(None, 0))
evoked.crop(tmin=0, tmax=0.2)
# Handling forward solution
forward = mne.read_forward_solution(fwd_fname, force_fixed=False,
                                    surf_ori=True)

###############################################################################
# Run solver
norient = 3
stc = gamma_map(evoked, forward, cov, maxit=500, tol=1e-20, maxit_nactive=200,
                gammas=np.ones(forward.shape[1] / norient, 1))

###############################################################################
# View in 2D and 3D ("glass" brain like 3D plot)
plot_sparse_source_estimates(forward['src'], stc, bgcolor=(1, 1, 1),
                             opacity=0.1, fig_name='g-map')
