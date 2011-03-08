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

def test_io_inverse():
    """Test IO for inverse operator
    """
    fwd = mne.read_inverse_operator(fname_inv)

def test_compute_mne_inverse():
    """Test MNE inverse computation
    """

    setno = 0
    snr = 3.0
    lambda2 = 1.0 / snr**2
    dSPM = True

    res = mne.compute_inverse(fname_data, setno, fname_inv, lambda2, dSPM,
                              baseline=(None, 0))

    assert np.all(res['sol'] > 0)
    assert np.all(res['sol'] < 35)

