# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_almost_equal)
from nose.tools import assert_true, assert_raises
import warnings

from mne.datasets import testing
from mne import read_forward_solution
from mne.simulation import simulate_sparse_stc, simulate_evoked
from mne import read_cov
from mne.io import read_raw_fif
from mne import pick_types_forward, read_evokeds
from mne.utils import run_tests_if_main

warnings.simplefilter('always')

data_path = testing.data_path(download=False)
fwd_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
raw_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                    'data', 'test_raw.fif')
ave_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                    'data', 'test-ave.fif')
cov_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                    'data', 'test-cov.fif')


@testing.requires_testing_data
def test_simulate_evoked():
    """Test simulation of evoked data."""

    raw = read_raw_fif(raw_fname)
    fwd = read_forward_solution(fwd_fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])
    cov = read_cov(cov_fname)

    evoked_template = read_evokeds(ave_fname, condition=0, baseline=None)
    evoked_template.pick_types(meg=True, eeg=True, exclude=raw.info['bads'])

    snr = 6  # dB
    tmin = -0.1
    sfreq = 1000.  # Hz
    tstep = 1. / sfreq
    n_samples = 600
    times = np.linspace(tmin, tmin + n_samples * tstep, n_samples)

    # Generate times series for 2 dipoles
    stc = simulate_sparse_stc(fwd['src'], n_dipoles=2, times=times)
    stc._data *= 1e-9

    # Generate noisy evoked data
    iir_filter = [1, -0.9]
    evoked = simulate_evoked(fwd, stc, evoked_template.info, cov, snr,
                             tmin=0.0, tmax=0.2, iir_filter=iir_filter)
    assert_array_almost_equal(evoked.times, stc.times)
    assert_true(len(evoked.data) == len(fwd['sol']['data']))

    # make a vertex that doesn't exist in fwd, should throw error
    stc_bad = stc.copy()
    mv = np.max(fwd['src'][0]['vertno'][fwd['src'][0]['inuse']])
    stc_bad.vertices[0][0] = mv + 1

    assert_raises(RuntimeError, simulate_evoked, fwd, stc_bad,
                  evoked_template.info, cov, snr, tmin=0.0, tmax=0.2)
    evoked_1 = simulate_evoked(fwd, stc, evoked_template.info, cov, np.inf,
                               tmin=0.0, tmax=0.2)
    evoked_2 = simulate_evoked(fwd, stc, evoked_template.info, cov, np.inf,
                               tmin=0.0, tmax=0.2)
    assert_array_equal(evoked_1.data, evoked_2.data)

    # test snr definition in dB
    evoked_noise = simulate_evoked(fwd, stc, evoked_template.info, cov,
                                   snr=snr, tmin=None, tmax=None,
                                   iir_filter=None)
    evoked_clean = simulate_evoked(fwd, stc, evoked_template.info, cov,
                                   snr=np.inf, tmin=None, tmax=None,
                                   iir_filter=None)
    noise = evoked_noise.data - evoked_clean.data

    empirical_snr = 10 * np.log10(np.mean((evoked_clean.data ** 2).ravel()) /
                                  np.mean((noise ** 2).ravel()))

    assert_almost_equal(snr, empirical_snr, decimal=5)

    cov['names'] = cov.ch_names[:-2]  # Error channels are different.
    assert_raises(ValueError, simulate_evoked, fwd, stc, evoked_template.info,
                  cov, snr=3., tmin=None, tmax=None, iir_filter=None)

run_tests_if_main()
