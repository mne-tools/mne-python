# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import os.path as op
import numpy as np
from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal

from mne.datasets import testing
from mne import (read_cov, read_forward_solution, read_evokeds,
                 pick_types_forward, read_labels_from_annot)
from mne.cov import regularize
from mne.inverse_sparse import gamma_map
from mne.fixes import in1d
from mne.utils import run_tests_if_main, slow_test

data_path = testing.data_path(download=False)
fname_evoked = op.join(data_path, 'MEG', 'sample',
                       'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
subjects_dir = op.join(data_path, 'subjects')


def _check_stc(stc, evoked):
    """Helper to check STC correctness"""
    # This is where the result should land
    label = read_labels_from_annot('sample', 'aparc.a2009s', 'rh',
                                   subjects_dir=subjects_dir)
    label = [l for l in label
             if l.name in ['S_oc-temp_med_and_Lingual-rh',
                           'Unknown-rh', 'G_oc-temp_med-Parahip-rh']]
    label[0] += label[1]
    label[0] += label[2]
    label = label[0]

    assert_array_almost_equal(stc.times, evoked.times, 5)
    biggest = np.argsort(np.sum(stc.data ** 2, axis=1))[-5:]
    print(len(stc.vertices[0]), len(stc.vertices[1]))
    print(biggest)
    biggest = np.concatenate(stc.vertices)[
        biggest[biggest > len(stc.vertices[0])]]
    print(biggest)
    in_ = in1d(biggest, label.vertices)
    assert_true(in_.sum() > 0, str(biggest))


@slow_test
@testing.requires_testing_data
def test_gamma_map():
    """Test Gamma MAP inverse"""
    forward = read_forward_solution(fname_fwd, force_fixed=False,
                                    surf_ori=True)
    forward = pick_types_forward(forward, meg=False, eeg=True)
    evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
    evoked.resample(50)
    evoked.crop(tmin=0, tmax=0.3)

    cov = read_cov(fname_cov)
    cov = regularize(cov, evoked.info)

    alpha = 0.2
    # Same gamma
    stc = gamma_map(evoked, forward, cov, alpha, tol=1e-5,
                    xyz_same_gamma=True, update_mode=1, verbose=False)
    _check_stc(stc, evoked)
    # Not same gamma
    stc = gamma_map(evoked, forward, cov, alpha, tol=1e-5,
                    xyz_same_gamma=False, update_mode=1, verbose=False)
    _check_stc(stc, evoked)
    # Force fixed orientation
    stc, res = gamma_map(evoked, forward, cov, alpha, tol=1e-5,
                         xyz_same_gamma=False, update_mode=2,
                         loose=None, return_residual=True, verbose=False)
    _check_stc(stc, evoked)


run_tests_if_main()
