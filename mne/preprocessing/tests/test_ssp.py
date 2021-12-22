import os.path as op

import pytest
from numpy.testing import assert_array_almost_equal
import numpy as np

from mne.io import read_raw_fif, read_raw_ctf
from mne.io.proj import make_projector, activate_proj
from mne.preprocessing.ssp import compute_proj_ecg, compute_proj_eog
from mne.datasets import testing
from mne import pick_types

data_path = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
dur_use = 5.0
eog_times = np.array([0.5, 2.3, 3.6, 14.5])

ctf_fname = op.join(testing.data_path(download=False), 'CTF',
                    'testdata_ctf.ds')


@pytest.fixture()
def short_raw():
    """Create a short, picked raw instance."""
    raw = read_raw_fif(raw_fname).crop(0, 7).pick_types(
        meg=True, eeg=True, eog=True)
    raw.pick(raw.ch_names[:306:10] + raw.ch_names[306:]).load_data()
    raw.info.normalize_proj()
    return raw


@pytest.mark.parametrize('average', (True, False))
def test_compute_proj_ecg(short_raw, average):
    """Test computation of ECG SSP projectors."""
    raw = short_raw

    # For speed, let's not filter here (must also not reject then)
    with pytest.warns(RuntimeWarning, match='Attenuation'):
        projs, events = compute_proj_ecg(
            raw, n_mag=2, n_grad=2, n_eeg=2, ch_name='MEG 1531',
            bads=['MEG 2443'], average=average, avg_ref=True, no_proj=True,
            l_freq=None, h_freq=None, reject=None, tmax=dur_use,
            qrs_threshold=0.5, filter_length=1000)
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
            h_freq=None, tmax=dur_use, return_drop_log=True,
            # XXX can be removed once
            # XXX https://github.com/mne-tools/mne-python/issues/9273
            # XXX has been resolved:
            qrs_threshold=1e-15)
    assert projs == []
    assert len(events) == len(drop_log)


@pytest.mark.parametrize('average', [True, False])
def test_compute_proj_eog(average, short_raw):
    """Test computation of EOG SSP projectors."""
    raw = short_raw

    n_projs_init = len(raw.info['projs'])
    with pytest.warns(RuntimeWarning, match='Attenuation'):
        projs, events = compute_proj_eog(
            raw, n_mag=2, n_grad=2, n_eeg=2, bads=['MEG 2443'],
            average=average, avg_ref=True, no_proj=False, l_freq=None,
            h_freq=None, reject=None, tmax=dur_use, filter_length=1000)
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
        projs, events = compute_proj_eog(
            raw, n_mag=2, n_grad=2, n_eeg=2, average=average, bads=[],
            avg_ref=True, no_proj=False, l_freq=None, h_freq=None,
            tmax=dur_use)
    assert projs == []

    raw._data[raw.ch_names.index('EOG 061'), :] = 1.
    with pytest.warns(RuntimeWarning, match='filter.*longer than the signal'):
        projs, events = compute_proj_eog(raw=raw, tmax=dur_use,
                                         ch_name='EOG 061')


@pytest.mark.slowtest  # can be slow on OSX
def test_compute_proj_parallel(short_raw):
    """Test computation of ExG projectors using parallelization."""
    short_raw = short_raw.copy().pick(('eeg', 'eog')).resample(100)
    raw = short_raw.copy()
    with pytest.warns(RuntimeWarning, match='Attenuation'):
        projs, _ = compute_proj_eog(
            raw, n_eeg=2, bads=raw.ch_names[1:2], average=False,
            avg_ref=True, no_proj=False, n_jobs=1, l_freq=None, h_freq=None,
            reject=None, tmax=dur_use, filter_length=100)
    raw_2 = short_raw.copy()
    with pytest.warns(RuntimeWarning, match='Attenuation'):
        projs_2, _ = compute_proj_eog(
            raw_2, n_eeg=2, bads=raw.ch_names[1:2],
            average=False, avg_ref=True, no_proj=False, n_jobs=2,
            l_freq=None, h_freq=None, reject=None, tmax=dur_use,
            filter_length=100)
    projs = activate_proj(projs)
    projs_2 = activate_proj(projs_2)
    projs, _, _ = make_projector(projs, raw_2.info['ch_names'],
                                 bads=['MEG 2443'])
    projs_2, _, _ = make_projector(projs_2, raw_2.info['ch_names'],
                                   bads=['MEG 2443'])
    assert_array_almost_equal(projs, projs_2, 10)


def _check_projs_for_expected_channels(projs, n_mags, n_grads, n_eegs):
    assert projs is not None
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
    raw = read_raw_ctf(ctf_fname, preload=True)

    # expected channels per projector type
    mag_picks = pick_types(
        raw.info, meg='mag', ref_meg=False, exclude='bads')[::10]
    n_mags = len(mag_picks)
    grad_picks = pick_types(raw.info, meg='grad', ref_meg=False,
                            exclude='bads')[::10]
    n_grads = len(grad_picks)
    eeg_picks = pick_types(raw.info, meg=False, eeg=True, ref_meg=False,
                           exclude='bads')[2::3]
    n_eegs = len(eeg_picks)
    ref_picks = pick_types(raw.info, meg=False, ref_meg=True)
    raw.pick(np.sort(np.concatenate(
        [mag_picks, grad_picks, eeg_picks, ref_picks])))
    del mag_picks, grad_picks, eeg_picks, ref_picks

    # Test with and without gradient compensation
    raw.apply_gradient_compensation(0)
    n_projs_init = len(raw.info['projs'])
    with pytest.warns(RuntimeWarning, match='Attenuation'):
        projs, _ = compute_proj_eog(
            raw, n_mag=2, n_grad=2, n_eeg=2, average=True, ch_name='EEG059',
            avg_ref=True, no_proj=False, l_freq=None, h_freq=None,
            reject=None, tmax=dur_use, filter_length=1000)
    _check_projs_for_expected_channels(projs, n_mags, n_grads, n_eegs)
    assert len(projs) == (5 + n_projs_init)

    raw.apply_gradient_compensation(1)
    with pytest.warns(RuntimeWarning, match='Attenuation'):
        projs, _ = compute_proj_ecg(
            raw, n_mag=1, n_grad=1, n_eeg=2, average=True, ch_name='EEG059',
            avg_ref=True, no_proj=False, l_freq=None, h_freq=None,
            reject=None, tmax=dur_use, filter_length=1000)
    _check_projs_for_expected_channels(projs, n_mags, n_grads, n_eegs)
    assert len(projs) == (4 + n_projs_init)
