import os.path as op
import warnings

from nose.tools import assert_true, assert_equal
from numpy.testing import assert_array_almost_equal
import numpy as np

from mne.io import read_raw_fif
from mne.io.proj import make_projector, activate_proj
from mne.preprocessing.ssp import compute_proj_ecg, compute_proj_eog
from mne.utils import run_tests_if_main

warnings.simplefilter('always')  # enable b/c these tests throw warnings

data_path = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
dur_use = 5.0
eog_times = np.array([0.5, 2.3, 3.6, 14.5])


def test_compute_proj_ecg():
    """Test computation of ECG SSP projectors."""
    raw = read_raw_fif(raw_fname).crop(0, 10)
    raw.load_data()
    for average in [False, True]:
        # For speed, let's not filter here (must also not reject then)
        projs, events = compute_proj_ecg(raw, n_mag=2, n_grad=2, n_eeg=2,
                                         ch_name='MEG 1531', bads=['MEG 2443'],
                                         average=average, avg_ref=True,
                                         no_proj=True, l_freq=None,
                                         h_freq=None, reject=None,
                                         tmax=dur_use, qrs_threshold=0.5,
                                         filter_length=6000)
        assert_true(len(projs) == 7)
        # heart rate at least 0.5 Hz, but less than 3 Hz
        assert_true(events.shape[0] > 0.5 * dur_use and
                    events.shape[0] < 3 * dur_use)
        ssp_ecg = [proj for proj in projs if proj['desc'].startswith('ECG')]
        # check that the first principal component have a certain minimum
        ssp_ecg = [proj for proj in ssp_ecg if 'PCA-01' in proj['desc']]
        thresh_eeg, thresh_axial, thresh_planar = .9, .3, .1
        for proj in ssp_ecg:
            if 'planar' in proj['desc']:
                assert_true(proj['explained_var'] > thresh_planar)
            elif 'axial' in proj['desc']:
                assert_true(proj['explained_var'] > thresh_axial)
            elif 'eeg' in proj['desc']:
                assert_true(proj['explained_var'] > thresh_eeg)
        # XXX: better tests

        # without setting a bad channel, this should throw a warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            projs, events = compute_proj_ecg(raw, n_mag=2, n_grad=2, n_eeg=2,
                                             ch_name='MEG 1531', bads=[],
                                             average=average, avg_ref=True,
                                             no_proj=True, l_freq=None,
                                             h_freq=None, tmax=dur_use)
        assert_true(len(w) >= 1)
        assert_equal(projs, None)


def test_compute_proj_eog():
    """Test computation of EOG SSP projectors."""
    raw = read_raw_fif(raw_fname).crop(0, 10)
    raw.load_data()
    for average in [False, True]:
        n_projs_init = len(raw.info['projs'])
        projs, events = compute_proj_eog(raw, n_mag=2, n_grad=2, n_eeg=2,
                                         bads=['MEG 2443'], average=average,
                                         avg_ref=True, no_proj=False,
                                         l_freq=None, h_freq=None,
                                         reject=None, tmax=dur_use,
                                         filter_length=6000)
        assert_true(len(projs) == (7 + n_projs_init))
        assert_true(np.abs(events.shape[0] -
                    np.sum(np.less(eog_times, dur_use))) <= 1)
        ssp_eog = [proj for proj in projs if proj['desc'].startswith('EOG')]
        # check that the first principal component have a certain minimum
        ssp_eog = [proj for proj in ssp_eog if 'PCA-01' in proj['desc']]
        thresh_eeg, thresh_axial, thresh_planar = .9, .3, .1
        for proj in ssp_eog:
            if 'planar' in proj['desc']:
                assert_true(proj['explained_var'] > thresh_planar)
            elif 'axial' in proj['desc']:
                assert_true(proj['explained_var'] > thresh_axial)
            elif 'eeg' in proj['desc']:
                assert_true(proj['explained_var'] > thresh_eeg)
        # XXX: better tests

        # This will throw a warning b/c simplefilter('always')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            projs, events = compute_proj_eog(raw, n_mag=2, n_grad=2, n_eeg=2,
                                             average=average, bads=[],
                                             avg_ref=True, no_proj=False,
                                             l_freq=None, h_freq=None,
                                             tmax=dur_use)
        assert_true(len(w) >= 1)
        assert_equal(projs, None)


def test_compute_proj_parallel():
    """Test computation of ExG projectors using parallelization."""
    raw_0 = read_raw_fif(raw_fname).crop(0, 10)
    raw_0.load_data()
    raw = raw_0.copy()
    projs, _ = compute_proj_eog(raw, n_mag=2, n_grad=2, n_eeg=2,
                                bads=['MEG 2443'], average=False,
                                avg_ref=True, no_proj=False, n_jobs=1,
                                l_freq=None, h_freq=None, reject=None,
                                tmax=dur_use, filter_length=6000)
    raw_2 = raw_0.copy()
    projs_2, _ = compute_proj_eog(raw_2, n_mag=2, n_grad=2, n_eeg=2,
                                  bads=['MEG 2443'], average=False,
                                  avg_ref=True, no_proj=False, n_jobs=2,
                                  l_freq=None, h_freq=None, reject=None,
                                  tmax=dur_use, filter_length=6000)
    projs = activate_proj(projs)
    projs_2 = activate_proj(projs_2)
    projs, _, _ = make_projector(projs, raw_2.info['ch_names'],
                                 bads=['MEG 2443'])
    projs_2, _, _ = make_projector(projs_2, raw_2.info['ch_names'],
                                   bads=['MEG 2443'])
    assert_array_almost_equal(projs, projs_2, 10)

run_tests_if_main()
