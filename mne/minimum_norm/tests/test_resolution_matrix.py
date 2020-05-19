# -*- coding: utf-8 -*-
# Author: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
from numpy.testing import (assert_equal, assert_array_almost_equal,
                           assert_array_equal)

import mne
from mne.datasets import testing
from mne.minimum_norm.resolution_matrix import (make_inverse_resolution_matrix,
                                                get_cross_talk,
                                                get_point_spread)

data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
fname_inv = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-meg-inv.fif')
fname_evoked = op.join(data_path, 'MEG', 'sample',
                       'sample_audvis_trunc-ave.fif')
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
fname_t1 = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
fname_src = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
fname_src_fs = op.join(data_path, 'subjects', 'fsaverage', 'bem',
                       'fsaverage-ico-5-src.fif')
fname_src_3 = op.join(data_path, 'subjects', 'sample', 'bem',
                      'sample-oct-4-src.fif')
fname_stc = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-meg')
fname_vol = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-grad-vol-7-fwd-sensmap-vol.w')
fname_vsrc = op.join(data_path, 'MEG', 'sample',
                     'sample_audvis_trunc-meg-vol-7-fwd.fif')
fname_inv_vol = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc-meg-vol-7-meg-inv.fif')
rng = np.random.RandomState(0)

fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-cov.fif')


@testing.requires_testing_data
def test_resolution_matrix():
    """Test make_inverse_resolution_matrix() function."""
    # read forward solution
    forward = mne.read_forward_solution(fname_fwd)

    # forward operator with fixed source orientations
    forward_fxd = mne.convert_forward_solution(forward, surf_ori=True,
                                               force_fixed=True)

    # noise covariance matrix
    noise_cov = mne.read_cov(fname_cov)

    # evoked data for info
    evoked = mne.read_evokeds(fname_evoked, 0)

    # make inverse operator from forward solution
    # free source orientation
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        info=evoked.info, forward=forward, noise_cov=noise_cov, loose=1.,
        depth=None)
    # fixed source orientation
    inverse_operator_fxd = mne.minimum_norm.make_inverse_operator(
        info=evoked.info, forward=forward, noise_cov=noise_cov, loose=0.,
        depth=None, fixed=True)

    # regularisation parameter based on SNR
    snr = 3.0
    lambda2 = 1.0 / snr ** 2

    # resolution matrices for free source orientation

    # compute resolution matrix for MNE with free source orientations
    rm_mne_free = make_inverse_resolution_matrix(forward, inverse_operator,
                                                 method='MNE', lambda2=lambda2)

    # compute resolution matrix for MNE, fwd fixed and inv free
    rm_mne_fxdfree = make_inverse_resolution_matrix(forward_fxd,
                                                    inverse_operator,
                                                    method='MNE',
                                                    lambda2=lambda2)

    # resolution matrices for fixed source orientation

    # compute resolution matrix for MNE
    rm_mne = make_inverse_resolution_matrix(forward_fxd, inverse_operator_fxd,
                                            method='MNE', lambda2=lambda2)

    # compute resolution matrix for sLORETA
    rm_lor = make_inverse_resolution_matrix(forward_fxd, inverse_operator_fxd,
                                            method='sLORETA', lambda2=lambda2)

    # rectify resolution matrix for sLORETA before determining maxima
    rm_lor_abs = np.abs(rm_lor)

    # get maxima per column
    maxidxs = rm_lor_abs.argmax(axis=0)

    # create array with the expected stepwise increase in maximum indices
    goodidxs = np.arange(0, len(maxidxs), 1)

    # Tests

    # Does sLORETA have zero dipole localization error for columns/PSFs?
    assert_array_equal(maxidxs, goodidxs)

    # MNE resolution matrices symmetric?
    assert_array_almost_equal(rm_mne, rm_mne.T)
    assert_array_almost_equal(rm_mne_free, rm_mne_free.T)

    # Test conversion to STC
    idx = [1, 100, 400]
    stc_psf = get_point_spread(rm_mne, forward_fxd['src'], idx, norm=True)
    stc_ctf = get_cross_talk(rm_mne, forward_fxd['src'], idx, norm=True)

    assert_array_almost_equal(stc_psf.data, stc_ctf.data)

    # Test application of free inv to fixed fwd
    assert_equal(rm_mne_fxdfree.shape, (3 * rm_mne.shape[0],
                 rm_mne.shape[0]))
