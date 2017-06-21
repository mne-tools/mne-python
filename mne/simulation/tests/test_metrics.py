# Author: Yousra Bekhti <yousra.bekhti@gmail.com>
#         Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD (3-clause)


import os.path as op

import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import assert_true, assert_raises
import warnings

from mne import read_source_spaces
from mne.datasets import testing
from mne.simulation import simulate_sparse_stc, source_estimate_quantification
from mne.utils import run_tests_if_main

warnings.simplefilter('always')

data_path = testing.data_path(download=False)
src_fname = op.join(data_path, 'subjects', 'sample', 'bem',
                    'sample-oct-6-src.fif')


@testing.requires_testing_data
def test_metrics():
    """Test simulation metrics"""
    src = read_source_spaces(src_fname)
    times = np.arange(600) / 1000.
    rng = np.random.RandomState(42)
    stc1 = simulate_sparse_stc(src, n_dipoles=2, times=times, random_state=rng)
    stc2 = simulate_sparse_stc(src, n_dipoles=2, times=times, random_state=rng)
    E1_rms = source_estimate_quantification(stc1, stc1, metric='rms')
    E2_rms = source_estimate_quantification(stc2, stc2, metric='rms')
    E1_cos = source_estimate_quantification(stc1, stc1, metric='cosine')
    E2_cos = source_estimate_quantification(stc2, stc2, metric='cosine')

    # ### Tests to add
    assert_true(E1_rms == 0.)
    assert_true(E2_rms == 0.)
    assert_almost_equal(E1_cos, 0.)
    assert_almost_equal(E2_cos, 0.)
    stc_bad = stc2.copy().crop(0, 0.5)
    assert_raises(ValueError, source_estimate_quantification, stc1, stc_bad)
    stc_bad = stc2.copy()
    stc_bad.times -= 0.1
    assert_raises(ValueError, source_estimate_quantification, stc1, stc_bad)
    assert_raises(ValueError, source_estimate_quantification, stc1, stc2,
                  metric='foo')

run_tests_if_main()
