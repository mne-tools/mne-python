# Author: Mark Wronkiewicz <wronk@uw.edu>
#         Yousra Bekhti <yousra.bekhti@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import warnings
from copy import deepcopy

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from nose.tools import assert_true, assert_raises

from mne import (read_forward_solution, read_source_spaces,
                 read_bem_solution, pick_types_forward, read_trans)
from mne.datasets import testing
from mne.simulation import simulate_sparse_stc, simulate_raw
from mne.io import Raw
from mne.utils import _TempDir, slow_test, run_tests_if_main

warnings.simplefilter('always')

data_path = testing.data_path(download=False)
fwd_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
bem_fname = op.join(data_path, 'subjects', 'sample', 'bem',
                    'sample-1280-1280-1280-bem-sol.fif')
raw_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                    'data', 'test_raw.fif')
cov_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-cov.fif')
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
bem_fname = op.join(data_path, 'subjects', 'sample', 'bem',
                    'sample-1280-1280-1280-bem-sol.fif')
src_fname = op.join(data_path, 'subjects', 'sample', 'bem',
                    'sample-oct-6-src.fif')


@slow_test
@testing.requires_testing_data
def test_simulate_raw():
    """Test simulation of raw data"""
    # Create object necessary to simulate raw
    raw_template = Raw(raw_fname)
    info = raw_template.info
    fwd = read_forward_solution(fwd_fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True, eeg=True,
                             exclude=raw_template.info['bads'])
    trans = read_trans(trans_fname)
    src = read_source_spaces(src_fname)
    bem = read_bem_solution(bem_fname)

    tmin = 0.0

    sfreq = info['sfreq']  # Hz
    tstep = 1. / sfreq
    n_samples = 100
    times = np.arange(tmin, tmin + tstep * n_samples, tstep)

    # Simulate STC
    stc = simulate_sparse_stc(fwd['src'], 1, times, random_state=42)

    raw_times = stc.times - stc.times[0]

    # Test raw simulation with basic parameters
    raw_sim = simulate_raw(info, stc, trans, src, bem, raw_times,
                           random_state=42)
    assert_array_almost_equal(raw_sim.info['sfreq'], 1. / stc.tstep,
                              err_msg='Raw and STC tstep must be equal')

    # Test raw simulation with parameters as filename where possible
    raw_sim_2 = simulate_raw(info, stc, trans_fname, src_fname, bem_fname,
                             raw_times)

    # Some numerical imprecision
    assert_array_almost_equal(raw_sim_2._data, raw_sim._data, decimal=5)

    # Test all simulated artifacts (after simulating head positions)
    # TODO: Make head positions that are more reasonable than simple 1mm
    #       deviations
    head_pos_sim = dict()
    shifts = [[0.001, 0., -0.001], [-0.001, 0.001, 0.], [0., -0.001, 0.001]]

    for time_key, shift in zip(raw_times[0:len(shifts)], shifts):
        # Create 4x4 matrix transform and normalize
        temp_trans = deepcopy(info['dev_head_t'])
        temp_trans['trans'][:3, 3] += shift
        head_pos_sim[time_key] = temp_trans['trans']

    raw_sim_3 = simulate_raw(info, stc, trans, src, bem, raw_times, ecg=True,
                             blink=True, head_pos=head_pos_sim)

    # Check that EOG channels exist and are not zero
    eog_noise = raw_sim_3._data[raw_sim_3.ch_names.index('EOG 061'), :]
    assert_true(np.any(eog_noise != np.zeros_like(eog_noise)))

    # TODO: Eventually, add ECG channels. Testing data raw file doesn't contain
    #       ECG channels yet.

    # Make extreme transform and make sure tests fail
    head_pos_sim_err = deepcopy(head_pos_sim)
    head_pos_sim_err[0.0][:, 3] = 1.  # 1m translation in all directions at t=0
    assert_raises(simulate_raw(info, stc, trans, src, bem, raw_times,
                               ecg=False, blink=False,
                               head_pos=head_pos_sim_err))

    # Test IO on processed data
    tempdir = _TempDir()
    test_outname = op.join(tempdir, 'sim_test_raw.fif')
    raw_sim.save(test_outname)

    raw_sim_loaded = Raw(test_outname, preload=True, proj=False,
                         allow_maxshield=True)
    # Some numerical imprecision since save uses 'single' fmt
    assert_allclose(raw_sim_loaded._data[:, :], raw_sim._data[:, :], rtol=1e-6,
                    atol=1e-20)

run_tests_if_main()
