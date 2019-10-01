"""
Test the following properties of the resolution matrix.

Resolution matrix is symmetrical for MNE.
Resolution matrix has zero dipole localisation error for
columns (PSFs).
"""

import os.path as op
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_equal)

import mne
from mne.datasets import testing
from mne.minimum_norm import make_resolution_matrix

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
                    'sample_audvis-meg-oct-6-fwd.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-shrunk-cov.fif')

# get functions for resolution matrix, metrics etc.


@testing.requires_testing_data
def test_resolution_matrix():
    """Test whether MNE's resolution matrix is symmetric."""
    # read forward solution
    forward = mne.read_forward_solution(fname_fwd)

    # only use normal components in forward solution
    forward = mne.convert_forward_solution(forward, surf_ori=True,
                                           force_fixed=True)

    # noise covariance matrix
    noise_cov = mne.read_cov(fname_cov)

    # evoked data for info
    evoked = mne.evoked.read_evokeds(fname_evoked, 0)

    # make inverse operator from forward solution
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        info=evoked.info, forward=forward, noise_cov=noise_cov, loose=0.,
        depth=None, fixed=True)

    # regularisation parameter based on SNR
    snr = 3.0
    lambda2 = 1.0 / snr ** 2

    # compute resolution matrix for MNE
    rm_mne = make_resolution_matrix(forward, inverse_operator, method='MNE',
                                    lambda2=lambda2)

    # compute resolution matrix for sLORETA
    rm_lor = make_resolution_matrix(forward, inverse_operator,
                                    method='sLORETA', lambda2=lambda2)

    # rectify resolution matrix for sLORETA before determining maxima
    rm_lor_abs = np.abs(rm_lor)

    # get maxima per column
    maxidxs = rm_lor_abs.argmax(axis=0)

    # create array with the expected stepwise increase in maximum indices
    goodidxs = np.arange(0, len(maxidxs), 1)

    # Tests
    assert_array_almost_equal(rm_mne, rm_mne.T)

    # Does sLORETA have zero DLE for columns?
    assert_equal(maxidxs, goodidxs)
