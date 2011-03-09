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


def test_compute_mne_inverse():
    """Test MNE inverse computation
    """

    setno = 0
    snr = 3.0
    lambda2 = 1.0 / snr**2
    dSPM = True

    evoked = mne.fiff.Evoked(fname_data, setno=setno, baseline=(None, 0))
    inverse_operator = mne.read_inverse_operator(fname_inv)

    res = mne.compute_inverse(evoked, inverse_operator, lambda2, dSPM)

    assert np.all(res['sol'] > 0)
    assert np.all(res['sol'] < 35)
