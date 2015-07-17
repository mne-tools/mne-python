# Author: Mark Wronkiewicz <wronk@uw.edu>
#         Yousra Bekhti <yousra.bekhti@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
# from nose.tools import assert_true

from mne import (read_forward_solution, read_source_spaces,
                 read_bem_solution, pick_types_forward, read_trans)
from mne.datasets import testing
from mne.simulation import simulate_sparse_stc, simulate_raw
from mne.io import Raw
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
    raw_template = Raw(raw_fname)
    fwd = read_forward_solution(fwd_fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True, eeg=True,
                             exclude=raw_template.info['bads'])
    # cov = read_cov(cov_fname)
    bem = read_bem_solution(bem_fname)
    src = read_source_spaces(src_fname)
    trans = read_trans(trans_fname)

    tmin = -0.1
    sfreq = raw_template.info['sfreq']  # Hz
    tstep = 1. / sfreq
    n_samples = 100.
    times = np.arange(tmin, tmin + tstep * n_samples, tstep)

    # Simulate STC
    stc = simulate_sparse_stc(fwd['src'], 1, times)

    raw = simulate_raw(raw_template.info, stc, trans, src, bem, times)
    assert_array_almost_equal(raw_template.info['sfreq'], 1. / stc.tstep,
                              err_msg='Raw and STC tstep must be equal')

    # Test raw generation with parameters as filename where possible
    raw = simulate_raw(raw_template.info, stc, trans_fname, src_fname,
                       bem_fname, times)

    # Test all simulated artifacts (after simulating head positions)
    # TODO: Make head positions that are more reasonable than randomly changing
    #       position at every time point
    '''
    TODO
    head_pos_sim = dict()

    for time_key in raw_template.times:
        # Create 4x4 matrix transform and normalize
        temp_trans = np.zeros((4, 4))
        temp_trans[:3, :3] = np.random.rand(3, 3)
        temp_trans[:3, :3] /= np.linalg.norm(temp_trans[:3, :3], axis=0)
        temp_trans[-1, -1] = 1.
        head_pos_sim[time_key] = temp_trans

    raw = simulate_raw(raw_template.info, stc, trans, src, bem, times,
                       ecg=True, blink=True, head_pos=head_pos_sim)
    # Check that EOG and ECG channels exist and are not zero
    eog_noise = raw._data[raw.ch_names.index('EOG 061'), :]
    assert_true(np.any(eog_noise != np.zeros_like(eog_noise)))
    '''

    # Test io on processed data
    tempdir = _TempDir()
    test_outname = op.join(tempdir, 'test_raw_sim.fif')
    raw.save(test_outname)
    raw_sim_loaded = Raw(test_outname, preload=True, proj=False,
                         allow_maxshield=True)

    # Some numerical imprecision since save uses 'single' fmt
    assert_allclose(raw_sim_loaded._data[:, :], raw._data[:, :], rtol=1e-6,
                    atol=1e-20)
