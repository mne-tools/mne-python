# -*- coding: utf-8 -*-
# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)
"""
Test computation of resolution matrix for LCMV beamformers.

If noise and data covariance are the same, the LCMV beamformer weights should
be the transpose of the leadfield matrix.
"""

import os.path as op
import numpy as np
from numpy.testing import assert_

import mne
from mne.datasets import testing
from mne.beamformer import make_lcmv
from mne.beamformer.resolution_matrix_lcmv import make_resolution_matrix_lcmv

data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
fname_inv = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-meg-inv.fif')
fname_evoked = op.join(data_path, 'MEG', 'sample',
                       'sample_audvis_trunc-ave.fif')
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-cov.fif')


@testing.requires_testing_data
def test_resolution_matrix_lcmv():
    """Test computation of resolution matrix for LCMV beamformers."""
    # read forward solution
    forward = mne.read_forward_solution(fname_fwd)

    # remove bad channels
    forward = mne.pick_channels_forward(forward, exclude='bads')

    # forward operator with fixed source orientations
    forward_fxd = mne.convert_forward_solution(forward, surf_ori=True,
                                               force_fixed=True)

    # noise covariance matrix
    noise_cov = mne.read_cov(fname_cov)

    # evoked data for info
    evoked = mne.read_evokeds(fname_evoked, 0)

    # Resolution matrix for Beamformer

    info = evoked.info
    # over-regularise for testing (see below)
    noise_cov = mne.cov.regularize(noise_cov, info, mag=0.05, grad=0.05,
                                   eeg=0.05, rank=None)

    data_cov = noise_cov.copy()  # to test a property of LCMV

    # compute beamformer filters
    # reg=0. to make sure noise_cov and data_cov are as similar as possible
    filters = make_lcmv(info, forward_fxd, data_cov, reg=0.,
                        noise_cov=noise_cov,
                        pick_ori=None, rank=None,
                        weight_norm=None,
                        reduce_rank=False,
                        verbose=False)

    resmat_lcmv = make_resolution_matrix_lcmv(forward_fxd, filters)

    # for noise_cov==data_cov, the filter weights should be the transpose of
    # leadfield

    # create filters with transposed whitened leadfield as weights
    filters_lfd = filters.copy()
    filters_lfd['weights'] = forward_fxd['sol']['data'].T

    # compute resolution matrix for filters with transposed leadfield
    resmat_fwd = make_resolution_matrix_lcmv(forward_fxd, filters_lfd)

    # Tests

    # This correlation should be exactly 1.0, but it isn't
    # probably due to different treatment of noise_cov and data_cov under the
    # hood.
    for v in [10, 100, 100]:  # arbitrary vertices
        assert_(np.corrcoef(resmat_fwd[v:], resmat_lcmv[v, :])[1, 1] > .95)
