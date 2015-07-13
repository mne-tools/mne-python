import os.path as op

import numpy as np
# from numpy.testing import assert_array_almost_equal
# from nose.tools import assert_true, assert_raises
import warnings

from mne.datasets import testing
from mne import read_label, read_forward_solution
from mne.time_frequency import morlet
from mne.simulation import generate_sparse_stc, generate_evoked
from mne import read_cov
from mne.io import Raw
from mne import pick_types_forward, read_evokeds
from mne.minimum_norm.inverse import (apply_inverse, read_inverse_operator)
from mne.simulation import source_estimate_quantification

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
def simulate_evoked():
    """ Simulate evoked data """

    raw = Raw(raw_fname)
    fwd = read_forward_solution(fwd_fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True, eeg=False,
                             exclude=raw.info['bads'])
    cov = read_cov(cov_fname)
    label_names = ['Aud-lh', 'Aud-rh']
    labels = [read_label(op.join(data_path, 'MEG', 'sample', 'labels',
                         '%s.label' % label)) for label in label_names]

    evoked_template = read_evokeds(ave_fname, condition=0, baseline=None)
    evoked_template.pick_types(meg=True, eeg=False, exclude=raw.info['bads'])

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

    return evoked, stc


def test_simulation_metrics():
    evoked, stc = simulate_evoked()
    inverse_operator = read_inverse_operator(inv_fname)

    stc1 = apply_inverse(evoked, inverse_operator, lambda2, "dSPM")
    stc2 = apply_inverse(evoked, inverse_operator, lambda2, "MNE")

    E_rms = source_estimate_quantification(stc1, stc2, metric='rms')
    E_corr = source_estimate_quantification(stc1, stc2, metric='corr')
    print E_rms, E_corr[0]
