# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import os.path as op

from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal

from mne.datasets import testing
from mne import read_cov, read_forward_solution, read_evokeds
from mne.cov import regularize
from mne.inverse_sparse import gamma_map
from mne import pick_types_forward
from mne.utils import run_tests_if_main, slow_test

data_path = testing.data_path(download=False)
fname_evoked = op.join(data_path, 'MEG', 'sample',
                       'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
subjects_dir = op.join(data_path, 'subjects')


def _check_stc(stc, evoked, idx, ratio=50.):
    """Helper to check correctness"""
    assert_array_almost_equal(stc.times, evoked.times, 5)
    amps = np.sum(stc.data ** 2, axis=1)
    order = np.argsort(amps)[::-1]
    amps = amps[order]
    verts = np.concatenate(stc.vertices)[order]
    assert_equal(idx, verts[0], err_msg=str(list(verts)))
    assert_true(amps[0] > ratio * amps[1], msg=str(amps[0] / amps[1]))


@slow_test
@testing.requires_testing_data
def test_gamma_map():
    """Test Gamma MAP inverse"""
    forward = read_forward_solution(fname_fwd, force_fixed=False,
                                    surf_ori=True)
    forward = pick_types_forward(forward, meg=False, eeg=True)
    evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0),
                          proj=False)
    evoked.resample(50, npad=100)
    evoked.crop(tmin=0.1, tmax=0.16)  # crop to nice window near samp border

    cov = read_cov(fname_cov)
    cov = regularize(cov, evoked.info)

    alpha = 0.5
    stc = gamma_map(evoked, forward, cov, alpha, tol=1e-4,
                    xyz_same_gamma=True, update_mode=1)
    _check_stc(stc, evoked, 68477)

    stc = gamma_map(evoked, forward, cov, alpha, tol=1e-4,
                    xyz_same_gamma=False, update_mode=1)
    _check_stc(stc, evoked, 82010)

    # force fixed orientation
    stc = gamma_map(evoked, forward, cov, alpha, tol=1e-4,
                    xyz_same_gamma=False, update_mode=2,
                    loose=None, return_residual=False)
    _check_stc(stc, evoked, 85739, 20)


run_tests_if_main()
