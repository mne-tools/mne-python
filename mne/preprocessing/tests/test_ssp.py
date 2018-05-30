import os.path as op
import warnings

from nose.tools import assert_true, assert_equal
from numpy.testing import assert_array_almost_equal, assert_allclose
import numpy as np

from mne.io import read_raw_fif, read_raw_ctf
from mne.io.proj import make_projector, activate_proj
from mne.preprocessing import (compute_proj_ecg, compute_proj_eog,
                               create_ecg_epochs)
from mne.utils import run_tests_if_main
from mne.datasets import testing
from mne import (pick_types, compute_proj_epochs, compute_proj_evoked)

warnings.simplefilter('always')  # enable b/c these tests throw warnings

data_path = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
dur_use = 5.0
eog_times = np.array([0.5, 2.3, 3.6, 14.5])

testing_path = testing.data_path(download=False)
ctf_fname = op.join(testing_path, 'CTF', 'testdata_ctf.ds')
raw_sample_fname = op.join(testing_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw.fif')


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
        assert_true(events.shape[0] > 0.5 * dur_use and
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
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            projs, events, drop_log = compute_proj_ecg(
                raw, n_mag=2, n_grad=2, n_eeg=2, ch_name='MEG 1531', bads=[],
                average=average, avg_ref=True, no_proj=True, l_freq=None,
                h_freq=None, tmax=dur_use, return_drop_log=True)
        assert len(w) >= 1
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


def _check_projs_for_expected_channels(projs, n_mags, n_grads, n_eegs):
    for p in projs:
        if 'planar' in p['desc']:
            assert len(p['data']['col_names']) == n_grads
        elif 'axial' in p['desc']:
            assert len(p['data']['col_names']) == n_mags
        elif 'eeg' in p['desc']:
            assert len(p['data']['col_names']) == n_eegs


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


@testing.requires_testing_data
def test_compute_proj_xdawn():
    """Test computing projectors with Xdawn."""
    raw = read_raw_fif(raw_sample_fname).load_data().del_proj()
    raw.pick_types(meg='mag')

    # capture "true" / ideal epochs using the average in fairly clean data
    ecg_epochs = create_ecg_epochs(raw, l_freq=None, h_freq=None,
                                   tmin=-0.2, tmax=0.4)
    assert len(ecg_epochs) > 20
    assert ecg_epochs.info['projs'] == []
    proj_true = compute_proj_evoked(ecg_epochs.average(), n_mag=1)[0]
    proj_true = proj_true['data']['data'][0]
    assert proj_true.ndim == 1

    # create a huge environmental artifact
    rng = np.random.RandomState(0)
    artifact = rng.randn(len(proj_true))

    # ensure it's orthogonal to the proj of interest (cleans up some math)
    artifact -= np.dot(np.dot(artifact, proj_true), proj_true)
    artifact /= np.linalg.norm(artifact)
    assert_allclose(np.linalg.norm(proj_true), 1.)
    assert_allclose(np.linalg.norm(artifact), 1.)
    assert_allclose(np.dot(artifact, proj_true), 0., atol=1e-14)

    # add this artifact to our ECG epochs, with different amp per epoch
    # (similar to a low frequency/DC drift)
    amps = 2e-11 * rng.randn(len(ecg_epochs), 1, 1)
    ecg_epochs._data += artifact[np.newaxis, :, np.newaxis] * amps
    del amps

    # evaluate performance
    proj_epochs = compute_proj_epochs(
        ecg_epochs, n_mag=1)[0]
    proj_epochs_reg = compute_proj_epochs(
        ecg_epochs, n_mag=1, reg='diagonal_fixed')[0]
    assert_allclose(proj_epochs['data']['data'],
                    proj_epochs_reg['data']['data'])
    proj_evoked = compute_proj_evoked(
        ecg_epochs.average(), n_mag=1)[0]
    proj_xdawn = compute_proj_epochs(
        ecg_epochs, n_mag=1, fit_method='xdawn')[0]
    proj_xdawn_reg = compute_proj_epochs(
        ecg_epochs, n_mag=1, fit_method='xdawn', reg=0.9)[0]

    # Look at how much var explained by true proj and artifact
    for kind, proj, p_lim, v_lim in (
            ('epochs', proj_epochs, [0.00, 0.01], [0.9, 0.95]),  # bad
            ('epochs_reg', proj_epochs_reg, [0.00, 0.01], [0.9, 0.95]),  # bad
            ('evoked', proj_evoked, [0.2, 0.3], [0.97, 0.99]),  # better
            ('xdawn', proj_xdawn, [0.2, 0.3], [0.15, 0.2]),  # better
            ('xdawn_reg', proj_xdawn_reg, [0.3, 0.4], [0.01, 0.03])):  # best
        exp_var = proj['explained_var']
        assert v_lim[0] <= exp_var <= v_lim[1], kind
        proj = proj['data']['data'][0]
        proj_var = np.dot(proj_true, proj) ** 2
        assert p_lim[0] <= proj_var <= p_lim[1], kind
        arti_var = np.dot(artifact, proj) ** 2
        assert p_lim[1] < arti_var < 1.0, kind


run_tests_if_main()
