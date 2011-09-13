import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal

from ...datasets import sample
from ... import fiff, Covariance, read_forward_solution
from ..inverse import apply_inverse, read_inverse_operator, \
                      make_inverse_operator


examples_folder = op.join(op.dirname(__file__), '..', '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname_inv = op.join(data_path, 'MEG', 'sample',
                            # 'sample_audvis-meg-eeg-oct-6-meg-eeg-inv.fif')
                            'sample_audvis-meg-oct-6-meg-inv.fif')
fname_data = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis-meg-oct-6-fwd.fif')
                            # 'sample_audvis-meg-eeg-oct-6-fwd.fif')


def test_inverse_operator():
    """Test MNE inverse computation with precomputed inverse operator."""
    setno = 0
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    dSPM = True

    evoked = fiff.Evoked(fname_data, setno=setno, baseline=(None, 0))
    inverse_operator = read_inverse_operator(fname_inv)

    stc = apply_inverse(evoked, inverse_operator, lambda2, dSPM)

    assert np.all(stc.data > 0)
    assert np.all(stc.data < 35)

    # Test MNE inverse computation starting from forward operator
    noise_cov = Covariance(fname_cov)
    evoked = fiff.Evoked(fname_data, setno=0, baseline=(None, 0))
    fwd_op = read_forward_solution(fname_fwd, surf_ori=True)
    my_inv_op = make_inverse_operator(evoked.info, fwd_op, noise_cov,
                                      loose=0.2, depth=0.8)

    my_stc = apply_inverse(evoked, my_inv_op, lambda2, dSPM)

    assert_equal(stc.times, my_stc.times)
    assert_array_almost_equal(stc.data, my_stc.data, 2)
