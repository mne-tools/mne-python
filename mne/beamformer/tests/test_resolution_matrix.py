# -*- coding: utf-8 -*-
# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)
"""
Test computation of resolution matrix for LCMV beamformers.

If noise and data covariance are the same, the LCMV beamformer weights should
be the transpose of the leadfield matrix.
"""

from copy import deepcopy
import os.path as op
import numpy as np
from numpy.testing import assert_allclose

import mne
from mne.datasets import testing
from mne.beamformer import make_lcmv, make_lcmv_resolution_matrix

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

    # evoked info
    info = mne.io.read_info(fname_evoked)
    mne.pick_info(info, mne.pick_types(info), copy=False)  # good MEG channels
    info['projs'] = []

    # noise covariance matrix
    noise_cov = mne.make_ad_hoc_cov(info)

    # Resolution matrix for Beamformer
    data_cov = noise_cov.copy()  # to test a property of LCMV

    # compute beamformer filters
    # reg=0. to make sure noise_cov and data_cov are as similar as possible
    filters = make_lcmv(info, forward_fxd, data_cov, reg=0.,
                        noise_cov=noise_cov,
                        pick_ori=None, rank=None,
                        weight_norm=None,
                        reduce_rank=False,
                        verbose=False)

    # Compute resolution matrix for beamformer
    resmat_lcmv = make_lcmv_resolution_matrix(filters, forward_fxd, info)

    # for noise_cov==data_cov, the filter weights should be the transpose of
    # leadfield

    # create filters with transposed whitened leadfield as weights
    forward_fxd = mne.pick_channels_forward(forward_fxd, info['ch_names'])
    filters_lfd = deepcopy(filters)
    filters_lfd['weights'][:] = forward_fxd['sol']['data'].T

    # compute resolution matrix for filters with transposed leadfield
    resmat_fwd = make_lcmv_resolution_matrix(filters_lfd, forward_fxd, info)

    # This correlation should be exactly 1.0, but it isn't
    # probably due to different treatment of noise_cov and data_cov under the
    # hood.
    for mat in (resmat_fwd, resmat_lcmv):
        mat -= np.mean(resmat_fwd, axis=1, keepdims=True)
        mat /= np.std(mat, axis=1, keepdims=True)
    corrs = np.mean(resmat_fwd * resmat_lcmv, axis=1)
    assert_allclose(corrs, 1., atol=0.12)
