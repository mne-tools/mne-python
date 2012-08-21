# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true

from mne.datasets import sample
from mne import read_label, read_forward_solution
from mne.time_frequency import morlet
from mne.simulation import generate_sparse_stc, generate_evoked
import mne
from mne.fiff.pick import pick_types_evoked, pick_types_forward


examples_folder = op.join(op.dirname(__file__), '..', '..', '..' '/examples')
data_path = sample.data_path(examples_folder)
fwd_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-meg-eeg-oct-6-fwd.fif')
raw_fname = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests',
                    'data', 'test_raw.fif')
ave_fname = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests',
                    'data', 'test-ave.fif')
cov_fname = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests',
                    'data', 'test-cov.fif')


def test_simulate_evoked():
    """ Test simulation of evoked data """

    raw = mne.fiff.Raw(raw_fname)
    fwd = read_forward_solution(fwd_fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])
    cov = mne.read_cov(cov_fname)
    label_names = ['Aud-lh', 'Aud-rh']
    labels = [read_label(op.join(data_path, 'MEG', 'sample', 'labels',
                        '%s.label' % label)) for label in label_names]

    evoked_template = mne.fiff.read_evoked(ave_fname, setno=0, baseline=None)
    evoked_template = pick_types_evoked(evoked_template, meg=True, eeg=True,
                                        exclude=raw.info['bads'])

    snr = 6  # dB
    tmin = -0.1
    sfreq = 1000.  # Hz
    tstep = 1. / sfreq
    n_samples = 600
    times = np.linspace(tmin, tmin + n_samples * tstep, n_samples)

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
    iir_filter = [1, -0.9]
    evoked = generate_evoked(fwd, stc, evoked_template, cov, snr,
                             tmin=0.0, tmax=0.2, iir_filter=iir_filter)
    assert_array_almost_equal(evoked.times, stc.times)
    assert_true(len(evoked.data) == len(fwd['sol']['data']))
