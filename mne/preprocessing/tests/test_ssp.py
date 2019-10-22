import os.path as op

import pytest
from numpy.testing import assert_array_almost_equal
import numpy as np

from mne.io import read_raw_fif, read_raw_ctf
from mne.io.proj import make_projector, activate_proj
from mne.preprocessing.ssp import compute_proj_ecg, compute_proj_eog
from mne.utils import run_tests_if_main
from mne.datasets import testing
from mne import pick_types

data_path = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
dur_use = 5.0
eog_times = np.array([0.5, 2.3, 3.6, 14.5])

ctf_fname = op.join(testing.data_path(download=False), 'CTF',
                    'testdata_ctf.ds')


def test_compute_proj_ecg():
    """Test computation of ECG SSP projectors."""
    raw = read_raw_fif(raw_fname).crop(0, 10)
    raw.load_data()
    for average in [False, True]:
        # For speed, let's not filter here (must also not reject then)
        projs, events = compute_proj_ecg(
            raw, n_mag=2, n_grad=2, n_eeg=2, ch_name='MEG 1531',
            bads=['MEG 2443'], average=average, avg_ref=True, no_proj=True,
            l_freq=None, h_freq=None, reject=None, tmax=dur_use,
            qrs_threshold=0.5, filter_length=6000)
        assert len(projs) == 7
        # heart rate at least 0.5 Hz, but less than 3 Hz
        assert (events.shape[0] > 0.5 * dur_use and
                events.shape[0] < 3 * dur_use)
        ssp_ecg = [proj for proj in projs if proj['desc'].startswith('ECG')]
        # check that the first principal component have a certain minimum
        ssp_ecg = [proj for proj in ssp_ecg if 'PCA-01' in proj['desc']]
        thresh_eeg, thresh_axial, thresh_planar = .9, .3, .1
        for proj in ssp_ecg:
            if 'planar' in proj['desc']:
                assert proj['explained_var'] > thresh_planar
            elif 'axial' in proj['desc']:
                assert proj['explained_var'] > thresh_axial
            elif 'eeg' in proj['desc']:
                assert proj['explained_var'] > thresh_eeg
        # XXX: better tests

        # without setting a bad channel, this should throw a warning
        with pytest.warns(RuntimeWarning, match='No good epochs found'):
            projs, events, drop_log = compute_proj_ecg(
                raw, n_mag=2, n_grad=2, n_eeg=2, ch_name='MEG 1531', bads=[],
                average=average, avg_ref=True, no_proj=True, l_freq=None,
                h_freq=None, tmax=dur_use, return_drop_log=True)
        assert projs is None
        assert len(events) == len(drop_log)


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
        assert (len(projs) == (7 + n_projs_init))
        assert (np.abs(events.shape[0] -
                np.sum(np.less(eog_times, dur_use))) <= 1)
        ssp_eog = [proj for proj in projs if proj['desc'].startswith('EOG')]
        # check that the first principal component have a certain minimum
        ssp_eog = [proj for proj in ssp_eog if 'PCA-01' in proj['desc']]
        thresh_eeg, thresh_axial, thresh_planar = .9, .3, .1
        for proj in ssp_eog:
            if 'planar' in proj['desc']:
                assert (proj['explained_var'] > thresh_planar)
            elif 'axial' in proj['desc']:
                assert (proj['explained_var'] > thresh_axial)
            elif 'eeg' in proj['desc']:
                assert (proj['explained_var'] > thresh_eeg)
        # XXX: better tests

        with pytest.warns(RuntimeWarning, match='longer'):
            projs, events = compute_proj_eog(raw, n_mag=2, n_grad=2, n_eeg=2,
                                             average=average, bads=[],
                                             avg_ref=True, no_proj=False,
                                             l_freq=None, h_freq=None,
                                             tmax=dur_use)
        assert projs is None


@pytest.mark.slowtest  # can be slow on OSX
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


def _check_projs_for_expected_channels(projs, n_mags, n_grads, n_eegs):
    for p in projs:
        if 'planar' in p['desc']:
            assert len(p['data']['col_names']) == n_grads
        elif 'axial' in p['desc']:
            assert len(p['data']['col_names']) == n_mags
        elif 'eeg' in p['desc']:
            assert len(p['data']['col_names']) == n_eegs


@pytest.mark.slowtest  # can be slow on OSX
@testing.requires_testing_data
def test_compute_proj_ctf():
    """Test to show that projector code completes on CTF data."""
    raw = read_raw_ctf(ctf_fname)
    raw.load_data()

    # expected channels per projector type
    n_mags = len(pick_types(raw.info, meg='mag', ref_meg=False,
                            exclude='bads'))
    n_grads = len(pick_types(raw.info, meg='grad', ref_meg=False,
                             exclude='bads'))
    n_eegs = len(pick_types(raw.info, meg=False, eeg=True, ref_meg=False,
                            exclude='bads'))

    # Test with and without gradient compensation
    for c in [0, 1]:
        raw.apply_gradient_compensation(c)
        for average in [False, True]:
            n_projs_init = len(raw.info['projs'])
            projs, events = compute_proj_eog(raw, n_mag=2, n_grad=2, n_eeg=2,
                                             average=average,
                                             ch_name='EEG059',
                                             avg_ref=True, no_proj=False,
                                             l_freq=None, h_freq=None,
                                             reject=None, tmax=dur_use,
                                             filter_length=6000)
            _check_projs_for_expected_channels(projs, n_mags, n_grads, n_eegs)
            assert len(projs) == (5 + n_projs_init)

            projs, events = compute_proj_ecg(raw, n_mag=1, n_grad=1, n_eeg=2,
                                             average=average,
                                             ch_name='EEG059',
                                             avg_ref=True, no_proj=False,
                                             l_freq=None, h_freq=None,
                                             reject=None, tmax=dur_use,
                                             filter_length=6000)
            _check_projs_for_expected_channels(projs, n_mags, n_grads, n_eegs)
            assert len(projs) == (4 + n_projs_init)


run_tests_if_main()
