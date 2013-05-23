# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import os.path as op
import numpy as np
from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal

import mne
from mne.datasets import sample
from mne import fiff, read_cov, read_forward_solution
from mne.inverse_sparse import gamma_map

data_path = sample.data_path()
fname_evoked = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-eeg-oct-6-fwd.fif')


forward = read_forward_solution(fname_fwd, force_fixed=False, surf_ori=True)
evoked = fiff.Evoked(fname_evoked, setno=0, baseline=(None, 0))
evoked.crop(tmin=0, tmax=0.3)

cov = read_cov(fname_cov)
cov = mne.cov.regularize(cov, evoked.info)


def test_gamma_map():
    """Test Gamma MAP inverse"""

    alpha = 0.2
    stc = gamma_map(evoked, forward, cov, alpha, tol=1e-5,
                    xyz_same_gamma=True, update_mode=1)
    idx = np.argmax(np.sum(stc.data ** 2, axis=1))
    assert_true(np.concatenate(stc.vertno)[idx] == 96397)

    stc = gamma_map(evoked, forward, cov, alpha, tol=1e-5,
                    xyz_same_gamma=False, update_mode=1)
    idx = np.argmax(np.sum(stc.data ** 2, axis=1))
    assert_true(np.concatenate(stc.vertno)[idx] == 82010)

    # force fixed orientation
    stc, res = gamma_map(evoked, forward, cov, alpha, tol=1e-5,
                         xyz_same_gamma=False, update_mode=2,
                         loose=None, return_residual=True)
    idx = np.argmax(np.sum(stc.data ** 2, axis=1))
    assert_true(np.concatenate(stc.vertno)[idx] == 83398)

    assert_array_almost_equal(evoked.times, res.times)
