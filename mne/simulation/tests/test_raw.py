# Author: Mark Wronkiewicz <wronk@uw.edu>
#         Yousra Bekhti <>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true, assert_raises
import warnings

from mne.datasets import testing
from mne import read_label, read_forward_solution
from mne.time_frequency import morlet
from mne.simulation import generate_sparse_stc, generate_evoked
from mne import read_cov
from mne.io import Raw
from mne import pick_types_forward, read_evokeds

warnings.simplefilter('always')

data_path = testing.data_path(download=False)
fwd_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
raw_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                    'data', 'test_raw.fif')
cov_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                    'data', 'test-cov.fif')


@testing.requires_testing_data
def test_simulate_raw():
    """ Test simulation of raw data """

    raw = Raw(raw_fname)
    fwd = read_forward_solution(fwd_fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])
    cov = read_cov(cov_fname)
