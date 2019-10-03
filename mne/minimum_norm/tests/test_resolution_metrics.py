# -*- coding: utf-8 -*-
# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)
"""
Test the following properties for resolution metrics.

Peak localisation error of MNE is the same for PSFs and CTFs.
Peak localisation error of sLORETA for PSFs is zero.
Currently only for fixed source orientations.
"""

import os.path as op
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_equal,
                           assert_array_equal)

import mne
from mne.datasets import testing
from mne.minimum_norm.resolution_matrix import make_resolution_matrix
from mne.minimum_norm.resolution_metrics import (localisation_error,
                                                 spatial_width,
                                                 _rectify_resolution_matrix)

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
def test_resolution_metrics():
    """Test resolution metrics."""
    # read forward solution
    fwd = mne.read_forward_solution(fname_fwd)

    # forward operator with fixed source orientations
    fwd = mne.convert_forward_solution(fwd, surf_ori=True,
                                       force_fixed=True)

    # noise covariance matrix
    noise_cov = mne.read_cov(fname_cov)

    # evoked data for info
    evoked = mne.evoked.read_evokeds(fname_evoked, 0)

    # make inverse operator from forward solution
    # free source orientation
    # inverse_operator = mne.minimum_norm.make_inverse_operator(
    #     info=evoked.info, forward=forward, noise_cov=noise_cov, loose=1.,
    #     depth=None)

    # fixed source orientation
    inv = mne.minimum_norm.make_inverse_operator(
        info=evoked.info, forward=fwd, noise_cov=noise_cov, loose=0.,
        depth=None, fixed=True)

    # regularisation parameter based on SNR
    snr = 3.0
    lambda2 = 1.0 / snr ** 2

    print('###\nComputing resolution matrices.\n###')

    # resolution matrices for fixed source orientation

    # compute resolution matrix for MNE
    rm_mne = make_resolution_matrix(fwd, inv,
                                    method='MNE', lambda2=lambda2)

    # compute resolution matrix for sLORETA
    rm_lor = make_resolution_matrix(fwd, inv,
                                    method='sLORETA', lambda2=lambda2)

    # Compute localisation error
    le_mne_psf = localisation_error(rm_mne, fwd['src'], type='psf',
                                    metric='peak')
    le_mne_ctf = localisation_error(rm_mne, fwd['src'], type='ctf',
                                    metric='peak')
    le_lor_psf = localisation_error(rm_lor, fwd['src'], type='psf',
                                    metric='peak')

    # Compute spatial spread
    sd_mne_ctf = spatial_width(rm_mne, fwd['src'], type='ctf', metric='sd')
    sd_lor_ctf = spatial_width(rm_lor, fwd['src'], type='ctf', metric='sd')

    # # Tests

    # For MNE: PLE for PSF and CTF equal?
    assert_array_almost_equal(le_mne_psf, le_mne_ctf)
    # Zero PLE for sLORETA?
    assert_equal((le_lor_psf == 0.).all(), True)
    # Spatial deviation of CTFs for MNE and sLORETA equal?
    assert_array_almost_equal(sd_mne_ctf, sd_lor_ctf)

    # test "rectification" of resolution matrix
    r1 = np.ones([8, 4])
    r2 = _rectify_resolution_matrix(r1)

    assert_array_equal(r2, np.sqrt(2) * np.ones([4, 4]))
