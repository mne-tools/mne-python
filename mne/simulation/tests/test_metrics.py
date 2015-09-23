# Author: Yousra Bekhti <yousra.bekhti@gmail.com>
#         Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD (3-clause)


import os.path as op

import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import assert_true, assert_raises
import warnings

from mne import (read_forward_solution, read_cov, pick_types_forward,
                 read_evokeds)
from mne.datasets import testing
from mne.simulation import (simulate_sparse_stc, simulate_evoked,
                            source_estimate_quantification)
from mne.io import Raw
from mne.minimum_norm.inverse import (apply_inverse, read_inverse_operator)
from mne.utils import run_tests_if_main

warnings.simplefilter('always')

data_path = testing.data_path(download=False)
fwd_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
inv_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-meg-inv.fif')
raw_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                    'data', 'test_raw.fif')
ave_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                    'data', 'test-ave.fif')
cov_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                    'data', 'test-cov.fif')

snr = 3.0
lambda2 = 1.0 / snr ** 2


@testing.requires_testing_data
def _simulate_evoked():
    """Simulate evoked data """

    raw = Raw(raw_fname)
    fwd = read_forward_solution(fwd_fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True, eeg=False,
                             exclude=raw.info['bads'])
    cov = read_cov(cov_fname)

    evoked_template = read_evokeds(ave_fname, condition=0, baseline=None)
    evoked_template.pick_types(meg=True, eeg=False, exclude=raw.info['bads'])

    snr = 6  # dB
    tmin = -0.1
    sfreq = 1000.  # Hz
    tstep = 1. / sfreq
    n_samples = 600
    times = np.linspace(tmin, tmin + n_samples * tstep, n_samples)

    # Generate times series for 2 dipoles
    stc = simulate_sparse_stc(fwd['src'], n_dipoles=2, times=times)

    # Generate noisy evoked data
    iir_filter = [1, -0.9]
    with warnings.catch_warnings(record=True):  # positive semidef
        evoked = simulate_evoked(fwd, stc, evoked_template.info, cov, snr,
                                 tmin=0.0, tmax=0.2, iir_filter=iir_filter)

    return evoked, stc


def test_metrics():
    """Test simulation metrics"""
    evoked, stc = _simulate_evoked()
    inverse_operator = read_inverse_operator(inv_fname)

    stc1 = apply_inverse(evoked, inverse_operator, lambda2, "dSPM")
    stc2 = apply_inverse(evoked, inverse_operator, lambda2, "MNE")

    E1_rms = source_estimate_quantification(stc1, stc1, metric='rms')
    E2_rms = source_estimate_quantification(stc2, stc2, metric='rms')
    E1_cos = source_estimate_quantification(stc1, stc1, metric='cosine')
    E2_cos = source_estimate_quantification(stc2, stc2, metric='cosine')

    # ### Tests to add
    assert_true(E1_rms == 0.)
    assert_true(E2_rms == 0.)
    assert_almost_equal(E1_cos, 0.)
    assert_almost_equal(E2_cos, 0.)
    stc_bad = stc2.copy().crop(0, 1)
    assert_raises(ValueError, source_estimate_quantification, stc1, stc_bad)
    stc_bad = stc2.copy()
    stc_bad.times -= 0.1
    assert_raises(ValueError, source_estimate_quantification, stc1, stc_bad)
    assert_raises(ValueError, source_estimate_quantification, stc1, stc2,
                  metric='foo')

run_tests_if_main()
