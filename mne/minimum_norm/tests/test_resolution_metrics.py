# -*- coding: utf-8 -*-
# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#          Daniel McCloy <dan.mccloy@gmail.com>
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
import pytest
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_)

import mne
from mne.datasets import testing
from mne.minimum_norm.resolution_matrix import make_inverse_resolution_matrix
from mne.minimum_norm.spatial_resolution import (resolution_metrics,
                                                 _rectify_resolution_matrix)
from mne.utils import run_tests_if_main

data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
fname_inv = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-meg-inv.fif')
fname_evoked = op.join(data_path, 'MEG', 'sample',
                       'sample_audvis_trunc-ave.fif')
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
    evoked = mne.read_evokeds(fname_evoked, 0)

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
    rm_mne = make_inverse_resolution_matrix(fwd, inv,
                                            method='MNE', lambda2=lambda2)

    # compute very smooth MNE
    rm_mne_smooth = make_inverse_resolution_matrix(fwd, inv,
                                                   method='MNE', lambda2=100.)

    # compute resolution matrix for sLORETA
    rm_lor = make_inverse_resolution_matrix(fwd, inv,
                                            method='sLORETA', lambda2=lambda2)

    # Compute localisation error (STCs)
    # Peak
    le_mne_psf = resolution_metrics(rm_mne, fwd['src'], function='psf',
                                    metric='peak_err')
    le_mne_ctf = resolution_metrics(rm_mne, fwd['src'], function='ctf',
                                    metric='peak_err')
    le_lor_psf = resolution_metrics(rm_lor, fwd['src'], function='psf',
                                    metric='peak_err')
    # Centre-of-gravity
    cog_mne_psf = resolution_metrics(rm_mne, fwd['src'], function='psf',
                                     metric='cog_err')
    cog_mne_ctf = resolution_metrics(rm_mne, fwd['src'], function='ctf',
                                     metric='cog_err')

    # Compute spatial spread (STCs)
    # Spatial deviation
    sd_mne_psf = resolution_metrics(rm_mne, fwd['src'], function='psf',
                                    metric='sd_ext')
    sd_mne_psf_smooth = resolution_metrics(rm_mne_smooth, fwd['src'],
                                           function='psf',
                                           metric='sd_ext')
    sd_mne_ctf = resolution_metrics(rm_mne, fwd['src'], function='ctf',
                                    metric='sd_ext')
    sd_lor_ctf = resolution_metrics(rm_lor, fwd['src'], function='ctf',
                                    metric='sd_ext')
    # Maximum radius
    mr_mne_psf = resolution_metrics(rm_mne, fwd['src'], function='psf',
                                    metric='maxrad_ext', threshold=0.6)
    mr_mne_psf_smooth = resolution_metrics(rm_mne_smooth, fwd['src'],
                                           function='psf', metric='maxrad_ext',
                                           threshold=0.6)
    mr_mne_ctf = resolution_metrics(rm_mne, fwd['src'], function='ctf',
                                    metric='maxrad_ext', threshold=0.6)
    mr_lor_ctf = resolution_metrics(rm_lor, fwd['src'], function='ctf',
                                    metric='maxrad_ext', threshold=0.6)
    # lower threshold -> larger spatial extent
    mr_mne_psf_0 = resolution_metrics(rm_mne, fwd['src'], function='psf',
                                      metric='maxrad_ext', threshold=0.)
    mr_mne_psf_9 = resolution_metrics(rm_mne, fwd['src'], function='psf',
                                      metric='maxrad_ext', threshold=0.9)

    # Compute relative amplitude (STCs)
    ra_mne_psf = resolution_metrics(rm_mne, fwd['src'], function='psf',
                                    metric='peak_amp')
    ra_mne_ctf = resolution_metrics(rm_mne, fwd['src'], function='ctf',
                                    metric='peak_amp')

    # Tests

    with pytest.raises(ValueError, match='is not a recognized metric'):
        resolution_metrics(rm_mne, fwd['src'], function='psf', metric='foo')
    with pytest.raises(ValueError, match='a recognised resolution function'):
        resolution_metrics(rm_mne, fwd['src'], function='foo',
                           metric='peak_err')

    # For MNE: PLE for PSF and CTF equal?
    assert_array_almost_equal(le_mne_psf.data, le_mne_ctf.data)
    assert_array_almost_equal(cog_mne_psf.data, cog_mne_ctf.data)
    # For MNE: SD and maxrad for PSF and CTF equal?
    assert_array_almost_equal(sd_mne_psf.data, sd_mne_ctf.data)
    assert_array_almost_equal(mr_mne_psf.data, mr_mne_ctf.data)
    assert_((mr_mne_psf_0.data > mr_mne_psf_9.data).all())
    # For MNE: RA for PSF and CTF equal?
    assert_array_almost_equal(ra_mne_psf.data, ra_mne_ctf.data)
    # Zero PLE for sLORETA?
    assert_((le_lor_psf.data == 0.).all())
    # Spatial deviation and maxrad of CTFs for MNE and sLORETA equal?
    assert_array_almost_equal(sd_mne_ctf.data, sd_lor_ctf.data)
    assert_array_almost_equal(mr_mne_ctf.data, mr_lor_ctf.data)
    # Smooth MNE has larger spatial extent?
    assert_(np.sum(sd_mne_psf_smooth.data) > np.sum(sd_mne_psf.data))
    assert_(np.sum(mr_mne_psf_smooth.data) > np.sum(mr_mne_psf.data))

    # test "rectification" of resolution matrix
    r1 = np.ones([8, 4])
    r2 = _rectify_resolution_matrix(r1)

    assert_array_equal(r2, np.sqrt(2) * np.ones((4, 4)))


run_tests_if_main()
