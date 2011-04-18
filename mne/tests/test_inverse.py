import os.path as op

import numpy as np
# from numpy.testing import assert_array_almost_equal, assert_equal

import mne
from mne.datasets import sample

examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname_inv = op.join(data_path, 'MEG', 'sample',
                                        'sample_audvis-meg-oct-6-meg-inv.fif')
fname_data = op.join(data_path, 'MEG', 'sample',
                                        'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample',
                                        'sample_audvis-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                                        'sample_audvis-meg-eeg-oct-6-fwd.fif')


def test_apply_mne_inverse_operator():
    """Test MNE inverse computation
    """

    setno = 0
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    dSPM = True

    evoked = mne.fiff.Evoked(fname_data, setno=setno, baseline=(None, 0))
    inverse_operator = mne.read_inverse_operator(fname_inv)

    res = mne.apply_inverse(evoked, inverse_operator, lambda2, dSPM)

    assert np.all(res['sol'] > 0)
    assert np.all(res['sol'] < 35)


def test_compute_minimum_norm():
    """Test MNE inverse computation
    """

    setno = 0
    noise_cov = mne.Covariance(fname_cov)
    forward = mne.read_forward_solution(fname_fwd)
    evoked = mne.fiff.Evoked(fname_data, setno=setno, baseline=(None, 0))
    whitener = noise_cov.get_whitener(evoked.info, mag_reg=0.1,
                                      grad_reg=0.1, eeg_reg=0.1, pca=True)
    stc, K, W = mne.minimum_norm(evoked, forward, whitener,
                        orientation='loose', method='dspm', snr=3, loose=0.2)

    # XXX : test something
