"""
===================================================================
R must be symmetric for L2-MNE
===================================================================
"""

# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)

import os.path as op
from copy import deepcopy
import warnings

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_equal,
                           assert_array_equal, assert_allclose)
from nose.tools import assert_true, assert_raises, assert_not_equal
import mne
from mne.tests.common import assert_naming
from mne.utils import (_TempDir, requires_pandas, slow_test, requires_version,
                       run_tests_if_main)

from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator
from mne.minimum_norm.psf_ctf import cross_talk_function, \
    point_spread_function, _get_matrix_from_inverse_operator, _label_svd, \
    _prepare_info


warnings.simplefilter('always')

data_path = sample.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')

fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-meg-eeg-oct-6-fwd.fif')

# covariance matrix for inverse operator
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')

# evoked data
fname_evoked = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')

fname_label = [op.join(data_path, 'MEG', 'sample', 'labels', 'Aud-rh.label')]

pick_meg = True
pick_eeg = True


def test_invmat():
    """Test inverse operator matrix
    """

    # tests some sub-functions of PSF/CTF, e.g. if resolution matrix symmetric
    # and SVD works

    # read forward solution
    forward = mne.read_forward_solution(
        fname_fwd, force_fixed=True, surf_ori=True)

    forward = mne.pick_channels_forward(forward, exclude='bads')

    # read covariance matrix
    noise_cov = mne.read_cov(fname_cov)

    # read evoked data
    evoked = mne.read_evokeds(fname_evoked, condition=0)

    evoked.pick_types(meg=pick_meg, eeg=pick_eeg)
    info = evoked.info

    # regularisation parameter
    snr = 3.
    lambda2 = 1.0 / snr ** 2

    inverse_operator = mne.minimum_norm.make_inverse_operator(info, forward,
                                                              noise_cov=noise_cov, fixed=True, depth=None, loose=None)

    leadfield = mne.minimum_norm.psf_ctf._pick_leadfield(
        forward['sol']['data'], forward, evoked.ch_names)

    # test SVD of sub-leadfield
    # arbitrary vertex index
    vert_idx = 1000
    # dummy sub_leadfield for SVD
    n_rep = 5
    sub_leadfield = np.tile(leadfield[:, vert_idx], (n_rep, 1)).T

    n_svd_comp = 1
    this_label_lfd_summary, s_svd = _label_svd(sub_leadfield, n_svd_comp, info)

    # arrays should be the same before and after SVD
    assert_array_almost_equal(sub_leadfield[:, 0], this_label_lfd_summary[
                              :, 0] / np.sqrt(n_rep), decimal=5)

    # correlation should be exactly 1
    corr_lfd = np.corrcoef(sub_leadfield[:, 0], this_label_lfd_summary[:, 0])

    assert_true(corr_lfd[0, 1] > 0.9999)

    method = 'MNE'
    invmat, singvals = _get_matrix_from_inverse_operator(inverse_operator, forward, labels=None, method=method, lambda2=lambda2, mode='mean',
                                                pick_ori=None, n_svd_comp=None)

    info_inv = _prepare_info(inverse_operator)
    sub_invmat = np.tile(invmat[vert_idx, :], (n_rep, 1)).T

    this_label_inv_summary, s_svd = _label_svd(sub_invmat, n_svd_comp,
                                               info_inv)

    assert_array_almost_equal(sub_invmat[:, 0],
                              this_label_inv_summary[:, 0] / np.sqrt(n_rep), decimal=5)

    R = invmat.dot(leadfield)

    # resolution matrix should be symmetric
    assert_array_almost_equal(R, R.T)


def test_psfctf():
    """ Test cross-talk and point-spread functions
    """

    # read forward solution
    forward = mne.read_forward_solution(
        fname_fwd, force_fixed=True, surf_ori=True)

    # read covariance matrix
    noise_cov = mne.read_cov(fname_cov)

    # read evoked data
    evoked = mne.read_evokeds(fname_evoked, condition=0)

    evoked.pick_types(meg=pick_meg, eeg=pick_eeg)

    # get info for inverse operator
    info = evoked.info

    # read label(s)
    labels = [mne.read_label(ss) for ss in fname_label]

    # get label with one vertex
    # label = [labels[0].split(parts=16, subjects_dir=subjects_dir, subject='sample')[0]]
    label = [labels[0]]

    # make inverse operator based on specified forward solution
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        info, forward, noise_cov=noise_cov, fixed=True, depth=None)

    # regularisation parameter
    snr = 3.
    lambda2 = 1.0 / snr ** 2

    # how to reduce leadfield and inverse operator in labels
    mode = 'svd'
    n_svd_comp = 1

    # PSF/CTF for MNE
    method = 'MNE'  # can be 'MNE', 'dSPM', or 'sLORETA'

    stc_ctf_mne, ctf_s_mne = cross_talk_function(
        inverse_operator, forward, label, method=method, lambda2=lambda2,
        signed=True, mode=mode, n_svd_comp=n_svd_comp)

    stc_psf_mne, psf_s_mne = point_spread_function(
        inverse_operator, forward, method=method, labels=label,
        lambda2=lambda2, pick_ori=None, mode=mode, n_svd_comp=n_svd_comp)

    # PSF/CTF for sLORETA
    method = 'sLORETA'  # can be 'MNE', 'dSPM', or 'sLORETA'

    stc_ctf_lor, ctf_s_lor = cross_talk_function(
        inverse_operator, forward, label, method=method, lambda2=lambda2,
        signed=True, mode=mode, n_svd_comp=n_svd_comp)

    stc_psf_lor, psf_s_lor = point_spread_function(
        inverse_operator, forward, method=method, labels=label,
        lambda2=lambda2, pick_ori=None, mode=mode, n_svd_comp=n_svd_comp)

    # check if matrix dimensions correct
    assert_true(stc_ctf_mne.data.shape == stc_ctf_lor.data.shape)
    assert_true(stc_psf_mne.data.shape == stc_psf_lor.data.shape)
    assert_true(stc_psf_mne.data.shape == stc_ctf_lor.data.shape)

    # check of correlations between some PSFs/CTFs reasonable
    stc_corr = np.diag(np.corrcoef(
        stc_ctf_mne.data[:, 0:4].T, stc_psf_mne.data[:, 0:4].T))
    # liberal test, because SVD within labels may cause differences between STCs
    assert_true(all(stc_corr > 0.6))

    stc_corr = np.diag(np.corrcoef(
        stc_ctf_mne.data[:, 0:4].T, stc_ctf_lor.data[:, 0:4].T))
    # liberal test, because SVD within labels may cause differences between STCs
    assert_true(all(stc_corr > 0.6))

    # check if singular values for leadfield SVD the same
    assert_array_almost_equal(psf_s_mne[0]['eeg'], psf_s_lor[0]['eeg'])
    assert_array_almost_equal(psf_s_mne[0]['mag'], psf_s_lor[0]['mag'])
    assert_array_almost_equal(psf_s_mne[0]['grad'], psf_s_lor[0]['grad'])
