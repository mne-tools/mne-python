"""
Test the following properties for resolution metrics.

Peak localisation error of MNE is the same for PSFs and CTFs.
Peak localisation error of sLORETA for PSFs is zero.
Currently only for fixed source orientations.
"""

import os.path as op
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_)

import mne
from mne.datasets import testing
from mne.minimum_norm.resolution_matrix import make_resolution_matrix
from mne.minimum_norm.resolution_metrics import (resolution_metrics,
                                                 _rectify_resolution_matrix)

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
    rm_mne = make_resolution_matrix(fwd, inv,
                                    method='MNE', lambda2=lambda2)

    # compute resolution matrix for sLORETA
    rm_lor = make_resolution_matrix(fwd, inv,
                                    method='sLORETA', lambda2=lambda2)

    # Compute localisation error (STCs)
    le_mne_psf = resolution_metrics(rm_mne, fwd['src'], function='psf',
                                    kind='localization_error', metric='peak')
    le_mne_ctf = resolution_metrics(rm_mne, fwd['src'], function='ctf',
                                    kind='localization_error', metric='peak')
    le_lor_psf = resolution_metrics(rm_lor, fwd['src'], function='psf',
                                    kind='localization_error', metric='peak')

    # Compute spatial spread (STCs)
    sd_mne_psf = resolution_metrics(rm_mne, fwd['src'], function='psf',
                                    kind='spatial_extent', metric='sd')
    sd_mne_ctf = resolution_metrics(rm_mne, fwd['src'], function='ctf',
                                    kind='spatial_extent', metric='sd')
    sd_lor_ctf = resolution_metrics(rm_lor, fwd['src'], function='ctf',
                                    kind='spatial_extent', metric='sd')

    # Compute relative amplitude (STCs)
    ra_mne_psf = resolution_metrics(rm_mne, fwd['src'], function='psf',
                                    kind='relative_amplitude', metric='peak')
    ra_mne_ctf = resolution_metrics(rm_mne, fwd['src'], function='ctf',
                                    kind='relative_amplitude', metric='peak')

    # # Tests

    # For MNE: PLE for PSF and CTF equal?
    assert_array_almost_equal(le_mne_psf.data, le_mne_ctf.data)
    # For MNE: SD for PSF and CTF equal?
    assert_array_almost_equal(sd_mne_psf.data, sd_mne_ctf.data)
    # For MNE: RA for PSF and CTF equal?
    assert_array_almost_equal(ra_mne_psf.data, ra_mne_ctf.data)
    # Zero PLE for sLORETA?
    assert_((le_lor_psf.data == 0.).all())
    # Spatial deviation of CTFs for MNE and sLORETA equal?
    assert_array_almost_equal(sd_mne_ctf.data, sd_lor_ctf.data)

    # test "rectification" of resolution matrix
    r1 = np.ones([8, 4])
    r2 = _rectify_resolution_matrix(r1)

    assert_array_equal(r2, np.sqrt(2) * np.ones((4, 4)))
