# Author: Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD (3-clause)

from __future__ import division

import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true  # , assert_raises
import warnings

from mne.datasets import testing
from mne import read_forward_solution  # , read_label
# from mne.time_frequency import morlet
from mne.simulation import simulate_sparse_stc, generate_epochs
from mne.forward import restrict_forward_to_stc
from mne import read_cov
from mne.io import Raw
from mne import pick_types_forward
from mne.event import read_events

warnings.simplefilter('always')

data_path = testing.data_path(download=False)
fwd_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
raw_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data',
                    'test_raw.fif')
ave_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data',
                    'test-ave.fif')
cov_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data',
                    'test-cov.fif')
eve_name = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data',
                   'test-eve.fif')


@testing.requires_testing_data
def test_simulate_epochs():
    """ Test simulation of epoched data """

    raw = Raw(raw_fname, preload=True)
    raw = raw.pick_types(meg=True, eeg=True, exclude=raw.info['bads'])
    fwd = read_forward_solution(fwd_fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])
    cov = read_cov(cov_fname)
    eve = read_events(eve_name)
    # label_names = ['Aud-lh', 'Aud-rh']
    # labels = [read_label(op.join(data_path, 'MEG', 'sample', 'labels',
    #                      '%s.label' % label)) for label in label_names]

    n_epochs = 3
    snr = 5  # dB
    tmin = -0.1
    sfreq = raw.info['sfreq']  # Hz
    n_samples = 600.

    times = np.linspace(tmin, tmin + (n_samples - 1) / sfreq, n_samples)
    times -= times[np.where(times <= 0)[0][-1]]
    eve = eve[:n_epochs, :]

    stc = simulate_sparse_stc(fwd['src'], n_dipoles=1, times=times)

    fwd = restrict_forward_to_stc(fwd, stc)
    epochs = generate_epochs(fwd, [stc] * n_epochs, raw.info, cov, eve, snr)

    # TODO: Figure out why timing gets slightly off
    assert_array_almost_equal(epochs.times, stc.times)
    assert_true(len(epochs) == n_epochs)
    assert_true(epochs[0].get_data().shape[1] == len(fwd['sol']['data']))
    assert_true(epochs[0].get_data().shape[2] == len(stc.times))
