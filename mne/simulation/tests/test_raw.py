# Author: Mark Wronkiewicz <wronk@uw.edu>
#         Yousra Bekhti <yousra.bekhti@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from mne.datasets import testing
from mne import (read_forward_solution, read_source_spaces, read_bem_solution,
                 pick_types_forward, read_cov)
from mne.io import Raw
from mne.simulation import simulate_sparse_stc, simulate_raw
from mne.utils import _TempDir

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


@testing.requires_testing_data
def test_simulate_raw():
    """ Test simulation of raw data """

    # Create object necessary to simulate raw
    raw = Raw(raw_fname)
    fwd = read_forward_solution(fwd_fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])
    cov = read_cov(cov_fname)
    bem = read_bem_solution(bem_fname)
    src = read_source_spaces(src_fname)

    tmin = -0.1
    sfreq = raw.info['sfreq']  # Hz
    tstep = 1. / sfreq
    n_samples = 100.
    times = np.arange(tmin, tmin + tstep * n_samples, tstep)
    trans = fwd['mri_head_t']
    times -= times[np.where(times < 0)[0][-1]]

    # Simulate STC
    stc = simulate_sparse_stc(fwd['src'], 1, times)

    # Test raw generation with object parameters
    raw_sim = simulate_raw(raw, stc, trans, src, bem, cov=cov)

    assert_array_almost_equal(raw.info['sfreq'], 1. / stc.tstep)

    # # Test raw generation with parameters as filename where possible
    # raw_sim = simulate_raw(raw, stc, trans_fname, src_fname, bem_fname)
    # XXX need to be fixed

    # Test all simulated artifact
    raw_sim = simulate_raw(raw, stc, trans, src, bem, ecg=True, blink=True)

    # Test io on processed data
    tempdir = _TempDir()
    test_outname = op.join(tempdir, 'test_raw_sim.fif')
    raw_sim.save(test_outname)
    raw_sim_loaded = Raw(test_outname, preload=True, proj=False,
                         allow_maxshield=True)
    # Some numerical imprecision since save uses 'single' fmt
    assert_allclose(raw_sim_loaded._data[:, :], raw_sim._data[:, :], rtol=1e-6,
                    atol=1e-20)
