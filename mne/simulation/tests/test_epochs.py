# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
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
raw_fname = op.join(op.dirname('__file__'), '..', '..', 'io', 'tests', 'data',
                    'test_raw.fif')
ave_fname = op.join(op.dirname('__file__'), '..', '..', 'io', 'tests', 'data',
                    'test-ave.fif')
cov_fname = op.join(op.dirname('__file__'), '..', '..', 'io', 'tests', 'data',
                    'test-cov.fif')
eve_name =  op.join(op.dirname('__file__'), '..', '..', 'io', 'tests', 'data',
                    'test-eve.fif')


@testing.requires_testing_data
def test_simulate_epochs():
    """ Test simulation of epoched data """

    raw = Raw(raw_fname)
    fwd = read_forward_solution(fwd_fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])
    cov = read_cov(cov_fname)
    eve = read_events(eve_name)
    label_names = ['Aud-lh', 'Aud-rh']
    labels = [read_label(op.join(data_path, 'MEG', 'sample', 'labels',
                         '%s.label' % label)) for label in label_names]

    snr = 6  # dB
    tmin = -0.1
    sfreq = 1000.  # Hz
    tstep = 1. / sfreq
    n_samples = 600
    n_channels = fwd.info['n_chan']
    times = np.linspace(tmin, tmin + n_samples * tstep, n_samples)
    info = create_info(n_channels, sfreq, 'meg')

    # Generate times series from 2 Morlet wavelets
    stc_data = np.zeros((len(labels), len(times)))
    Ws = morlet(sfreq, [3, 10], n_cycles=[1, 1.5])
    stc_data[0][:len(Ws[0])] = np.real(Ws[0])
    stc_data[1][:len(Ws[1])] = np.real(Ws[1])
    stc_data *= 100 * 1e-9  # use nAm as unit

    # time translation
    stc_data[1] = np.roll(stc_data[1], 80)
    stc = generate_sparse_stc(fwd['src'], labels, stc_data, tmin, tstep,
                              random_state=0)

    # Generate noisy evoked data
    #with warnings.catch_warnings(record=True):
        #warnings.simplefilter('always')  # positive semidefinite warning
    epochs = generate_epochs(fwd, [stc] * 3, info, cov, snr, tmin=0.0,
                             tmax=0.2)
    assert_array_almost_equal(epochs.times, stc.times)
    assert_true(len(epochs[0].data) == len(fwd['sol']['data']))

    # make a vertex that doesn't exist in fwd, should throw error
    stc_bad = stc.copy()
    mv = np.max(fwd['src'][0]['vertno'][fwd['src'][0]['inuse']])
    stc_bad.vertices[0][0] = mv + 1
    assert_raises(RuntimeError, generate_evoked, fwd, stc_bad,
                  info, cov, snr, tmin=0.0, tmax=0.2)
