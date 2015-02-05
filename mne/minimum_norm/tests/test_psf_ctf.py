
import os.path as op
import mne
from mne.datasets import testing
from mne import read_forward_solution
from mne.minimum_norm import (read_inverse_operator,
                              point_spread_function, cross_talk_function)
from mne.utils import slow_test, run_tests_if_main

from nose.tools import assert_true

data_path = op.join(testing.data_path(download=False), 'MEG', 'sample')

fname_inv_meg = op.join(data_path,
                        'sample_audvis_trunc-meg-eeg-oct-4-meg-inv.fif')
fname_inv_meeg = op.join(data_path, 'sample_audvis_trunc-meg-eeg-oct-4-'
                         'meg-eeg-diagnoise-inv.fif')
fname_fwd = op.join(data_path, 'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')

fname_label = [op.join(data_path, 'labels', 'Aud-rh.label'),
               op.join(data_path, 'labels', 'Aud-lh.label')]

snr = 3.0
lambda2 = 1.0 / snr ** 2


@slow_test
@testing.requires_testing_data
def test_psf_ctf():
    """Test computation of PSFs and CTFs for linear estimators
    """
    forward = read_forward_solution(fname_fwd)
    labels = [mne.read_label(ss) for ss in fname_label]

    method = 'MNE'
    n_svd_comp = 2

    # make sure it works for both types of inverses
    for fname_inv in (fname_inv_meg, fname_inv_meeg):
        inverse_operator = read_inverse_operator(fname_inv)
        # Test PSFs (then CTFs)
        for mode in ('sum', 'svd'):
            stc_psf, psf_ev = point_spread_function(
                inverse_operator, forward, method=method, labels=labels,
                lambda2=lambda2, pick_ori='normal', mode=mode,
                n_svd_comp=n_svd_comp)

            n_vert, n_samples = stc_psf.shape
            should_n_vert = (inverse_operator['src'][1]['vertno'].shape[0] +
                             inverse_operator['src'][0]['vertno'].shape[0])
            if mode == 'svd':
                should_n_samples = len(labels) * n_svd_comp + 1
            else:
                should_n_samples = len(labels) + 1

            assert_true(n_vert == should_n_vert)
            assert_true(n_samples == should_n_samples)

            n_chan, n_samples = psf_ev.data.shape
            assert_true(n_chan == forward['nchan'])

        # Test CTFs
        for mode in ('sum', 'svd'):
            stc_ctf = cross_talk_function(
                inverse_operator, forward, labels, method=method,
                lambda2=lambda2, signed=False, mode=mode,
                n_svd_comp=n_svd_comp)

            n_vert, n_samples = stc_ctf.shape
            should_n_vert = (inverse_operator['src'][1]['vertno'].shape[0] +
                             inverse_operator['src'][0]['vertno'].shape[0])
            if mode == 'svd':
                should_n_samples = len(labels) * n_svd_comp + 1
            else:
                should_n_samples = len(labels) + 1

            assert_true(n_vert == should_n_vert)
            assert_true(n_samples == should_n_samples)


run_tests_if_main()
