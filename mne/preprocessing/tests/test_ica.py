# Author: Denis Engemann <d.engemann@fz-juelich.de>
#         Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os
import os.path as op

from nose.tools import assert_true, assert_raises
from copy import deepcopy
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import stats
from itertools import product

from mne import fiff, Epochs, read_events, cov
from mne.preprocessing import ICA, ica_find_ecg_events, ica_find_eog_events,\
                              read_ica, run_ica
from mne.preprocessing.ica import score_funcs, _check_n_pca_components
from mne.utils import _TempDir, requires_sklearn
from mne.fiff.meas_info import Info

tempdir = _TempDir()

data_dir = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')
evoked_nf_name = op.join(data_dir, 'test-nf-ave.fif')
test_cov_name = op.join(data_dir, 'test-cov.fif')

event_id, tmin, tmax = 1, -0.2, 0.5
start, stop = 0, 8  # if stop is too small pca may fail in some cases, but
                    # we're okay on this file
raw = fiff.Raw(raw_fname, preload=True).crop(0, stop, False)

events = read_events(event_name)
picks = fiff.pick_types(raw.info, meg=True, stim=False, ecg=False, eog=False,
                        exclude='bads')

# for testing eog functionality
picks2 = fiff.pick_types(raw.info, meg=True, stim=False, ecg=False, eog=True,
                         exclude='bads')

reject = dict(grad=1000e-12, mag=4e-12, eeg=80e-6, eog=150e-6)
flat = dict(grad=1e-15, mag=1e-15)

test_cov = cov.read_cov(test_cov_name)
epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                baseline=(None, 0), preload=True)

epochs_eog = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks2,
                baseline=(None, 0), preload=True)

score_funcs_unsuited = ['pointbiserialr', 'ansari']


@requires_sklearn
def test_ica_core():
    """Test ICA on raw and epochs
    """
    # Most basic recovery
    raw_ = raw.crop(5.0)
    ica = ICA(n_components=5, max_pca_components=5,
              n_pca_components=5)
    ica.decompose_raw(raw_, picks=np.arange(ica.n_components))
    raw3 = ica.pick_sources_raw(raw_, exclude=[])
    assert_array_almost_equal(raw_._data[:ica.n_components_],
                              raw3._data[:ica.n_components_])
    del raw3, raw_

    # XXX. The None cases helped revealing bugs but are time consuming.
    noise_cov = [None, test_cov]
    # removed None cases to speed up...
    n_components = [3, 1.0]  # for future dbg add cases
    max_pca_components = [4]
    picks_ = [picks]
    iter_ica_params = product(noise_cov, n_components, max_pca_components,
                              picks_)

    # # test init catchers
    assert_raises(ValueError, ICA, n_components=3, max_pca_components=2)
    assert_raises(ValueError, ICA, n_components=1.3, max_pca_components=2)

    # test essential core functionality
    for n_cov, n_comp, max_n, pcks in iter_ica_params:
      # Test ICA raw
        ica = ICA(noise_cov=n_cov, n_components=n_comp,
                  max_pca_components=max_n, n_pca_components=max_n,
                  random_state=0)

        print ica  # to test repr

        # test fit checker
        assert_raises(RuntimeError, ica.get_sources_raw, raw)
        assert_raises(RuntimeError, ica.get_sources_epochs, epochs)

        # test decomposition
        ica.decompose_raw(raw, picks=pcks, start=start, stop=stop)
        print ica  # to test repr
        # test re-init exception
        assert_raises(RuntimeError, ica.decompose_raw, raw, picks=picks)

        sources = ica.get_sources_raw(raw)
        assert_true(sources.shape[0] == ica.n_components_)

        # test preload filter
        raw3 = raw.copy()
        raw3._preloaded = False
        assert_raises(ValueError, ica.pick_sources_raw, raw3,
                      include=[1, 2])

        for excl, incl in (([], []), ([], [1, 2]), ([1, 2], [])):
            raw2 = ica.pick_sources_raw(raw, exclude=excl, include=incl,
                                        copy=True)

            assert_array_almost_equal(raw2[:, :][1], raw[:, :][1])

        #######################################################################
        # test epochs decomposition

        # test re-init exception
        assert_raises(RuntimeError, ica.decompose_epochs, epochs, picks=picks)
        ica = ICA(noise_cov=n_cov, n_components=n_comp,
                  max_pca_components=max_n, n_pca_components=max_n,
                  random_state=0)

        ica.decompose_epochs(epochs, picks=picks)
        print ica  # to test repr
        # test pick block after epochs fit
        assert_raises(ValueError, ica.pick_sources_raw, raw)

        sources = ica.get_sources_epochs(epochs)
        assert_true(sources.shape[1] == ica.n_components_)

        assert_raises(ValueError, ica.find_sources_epochs, epochs,
                      target=np.arange(1))

        # test preload filter
        epochs3 = epochs.copy()
        epochs3.preload = False
        assert_raises(ValueError, ica.pick_sources_epochs, epochs3,
                      include=[1, 2])

        # test source picking
        for excl, incl in (([], []), ([], [1, 2]), ([1, 2], [])):
            epochs2 = ica.pick_sources_epochs(epochs, exclude=excl,
                                      include=incl, copy=True)

            assert_array_almost_equal(epochs2.get_data(),
                                      epochs.get_data())


@requires_sklearn
def test_ica_additional():
    """Test additional functionality
    """
    stop2 = 500

    test_cov2 = deepcopy(test_cov)
    ica = ICA(noise_cov=test_cov2, n_components=3, max_pca_components=4,
              n_pca_components=4)
    assert_true(ica.info is None)
    ica.decompose_raw(raw, picks[:5])
    assert_true(isinstance(ica.info, Info))
    assert_true(ica.n_components_ < 5)

    ica = ICA(n_components=3, max_pca_components=4,
              n_pca_components=4)
    assert_raises(RuntimeError, ica.save, '')
    ica.decompose_raw(raw, picks=None, start=start, stop=stop2)

    # epochs extraction from raw fit
    assert_raises(RuntimeError, ica.get_sources_epochs, epochs)
    # test reading and writing
    test_ica_fname = op.join(op.dirname(tempdir), 'ica_test.fif')
    for cov in (None, test_cov):
        ica = ICA(noise_cov=cov, n_components=3, max_pca_components=4,
                  n_pca_components=4)
        ica.decompose_raw(raw, picks=picks, start=start, stop=stop2)
        sources = ica.get_sources_epochs(epochs)
        assert_true(sources.shape[1] == ica.n_components_)

        for exclude in [[], [0]]:
            ica.exclude = [0]
            ica.save(test_ica_fname)
            ica_read = read_ica(test_ica_fname)
            assert_true(ica.exclude == ica_read.exclude)
            # test pick merge -- add components
            ica.pick_sources_raw(raw, exclude=[1])
            assert_true(ica.exclude == [0, 1])
            #                 -- only as arg
            ica.exclude = []
            ica.pick_sources_raw(raw, exclude=[0, 1])
            assert_true(ica.exclude == [0, 1])
            #                 -- remove duplicates
            ica.exclude += [1]
            ica.pick_sources_raw(raw, exclude=[0, 1])
            assert_true(ica.exclude == [0, 1])

            ica_raw = ica.sources_as_raw(raw)
            assert_true(ica.exclude == [ica_raw.ch_names.index(e) for e in
                                        ica_raw.info['bads']])

        ica.n_pca_components = 2
        ica.save(test_ica_fname)
        ica_read = read_ica(test_ica_fname)
        assert_true(ica.n_pca_components ==
                    ica_read.n_pca_components)
        ica.n_pca_components = 4
        ica_read.n_pca_components = 4

        ica.exclude = []
        ica.save(test_ica_fname)
        ica_read = read_ica(test_ica_fname)
        get = getattr
        for attr in ['mixing_matrix_', 'unmixing_matrix_', 'pca_components_',
                     'pca_mean_', 'pca_explained_variance_',
                     '_pre_whitener']:
            assert_array_almost_equal(get(ica, attr), get(ica_read, attr))

        assert_true(ica.ch_names == ica_read.ch_names)
        assert_true(isinstance(ica_read.info, Info))  # XXX improve later
        # assert_true(np.allclose(ica.mixing_matrix_, ica_read.mixing_matrix_,
        #                         rtol=1e-16, atol=1e-32))
        # assert_array_equal(ica.pca_components_,
        #                    ica_read.pca_components_)
        # assert_array_equal(ica.pca_mean_, ica_read.pca_mean_)
        # assert_array_equal(ica.pca_explained_variance_,
        #                    ica_read.pca_explained_variance_)
        # assert_array_equal(ica._pre_whitener, ica_read._pre_whitener)

        assert_raises(RuntimeError, ica_read.decompose_raw, raw)
        sources = ica.get_sources_raw(raw)
        sources2 = ica_read.get_sources_raw(raw)
        assert_array_almost_equal(sources, sources2)

        _raw1 = ica.pick_sources_raw(raw, exclude=[1])
        _raw2 = ica_read.pick_sources_raw(raw, exclude=[1])
        assert_array_almost_equal(_raw1[:, :][0], _raw2[:, :][0])

    os.remove(test_ica_fname)
    # check scrore funcs
    for name, func in score_funcs.items():
        if name in score_funcs_unsuited:
            continue
        scores = ica.find_sources_raw(raw, target='EOG 061', score_func=func,
                                      start=0, stop=10)
        assert_true(ica.n_components_ == len(scores))

    # check univariate stats
    scores = ica.find_sources_raw(raw, score_func=stats.skew)
    # check exception handling
    assert_raises(ValueError, ica.find_sources_raw, raw,
                  target=np.arange(1))

    params = []
    params += [(None, -1, slice(2), [0, 1])]  # varicance, kurtosis idx params
    params += [(None, 'MEG 1531')]  # ECG / EOG channel params
    for idx, ch_name in product(*params):
        ica.detect_artifacts(raw, start_find=0, stop_find=50, ecg_ch=ch_name,
                             eog_ch=ch_name, skew_criterion=idx,
                             var_criterion=idx, kurt_criterion=idx)
    ## score funcs epochs ##

    # check score funcs
    for name, func in score_funcs.items():
        if name in score_funcs_unsuited:
            continue
        scores = ica.find_sources_epochs(epochs_eog, target='EOG 061',
                                         score_func=func)
        assert_true(ica.n_components_ == len(scores))

    # check univariate stats
    scores = ica.find_sources_epochs(epochs, score_func=stats.skew)

    # check exception handling
    assert_raises(ValueError, ica.find_sources_epochs, epochs,
                  target=np.arange(1))

    # ecg functionality
    ecg_scores = ica.find_sources_raw(raw, target='MEG 1531',
                                      score_func='pearsonr')

    ecg_events = ica_find_ecg_events(raw, sources[np.abs(ecg_scores).argmax()])

    assert_true(ecg_events.ndim == 2)

    # eog functionality
    eog_scores = ica.find_sources_raw(raw, target='EOG 061',
                                      score_func='pearsonr')
    eog_events = ica_find_eog_events(raw, sources[np.abs(eog_scores).argmax()])

    assert_true(eog_events.ndim == 2)

    # Test ica fiff export
    ica_raw = ica.sources_as_raw(raw, start=0, stop=100)
    assert_true(ica_raw.last_samp - ica_raw.first_samp == 100)
    ica_chans = [ch for ch in ica_raw.ch_names if 'ICA' in ch]
    assert_true(ica.n_components_ == len(ica_chans))
    test_ica_fname = op.join(op.abspath(op.curdir), 'test_ica.fif')
    ica_raw.save(test_ica_fname, overwrite=True)
    ica_raw2 = fiff.Raw(test_ica_fname, preload=True)
    assert_array_almost_equal(ica_raw._data, ica_raw2._data)
    ica_raw2.close()
    os.remove(test_ica_fname)

    # Test ica epochs export
    ica_epochs = ica.sources_as_epochs(epochs)
    assert_true(ica_epochs.events.shape == epochs.events.shape)
    sources_epochs = ica.get_sources_epochs(epochs)
    assert_array_equal(ica_epochs.get_data(), sources_epochs)
    ica_chans = [ch for ch in ica_epochs.ch_names if 'ICA' in ch]
    assert_true(ica.n_components_ == len(ica_chans))
    assert_true(ica.n_components_ == ica_epochs.get_data().shape[1])
    assert_true(ica_epochs.raw is None)
    assert_true(ica_epochs.preload == True)

    # regression test for plot method
    assert_raises(ValueError, ica.plot_sources_raw, raw,
                  order=np.arange(50))
    assert_raises(ValueError, ica.plot_sources_epochs, epochs,
                  order=np.arange(50))

    # test float n pca components
    ica.pca_explained_variance_ = np.array([0.2] * 5)
    for ncomps, expected in [[0.3, 1], [0.9, 4], [1, 1]]:
        ncomps_ = _check_n_pca_components(ica, ncomps)
        assert_true(ncomps_ == expected)


def test_run_ica():
    """Test run_ica function"""
    params = []
    params += [(None, -1, slice(2), [0, 1])]  # varicance, kurtosis idx
    params += [(None, 'MEG 1531')]  # ECG / EOG channel params
    for idx, ch_name in product(*params):
        run_ica(raw, n_components=.9, start=0, stop=100, start_find=0,
                stop_find=50, ecg_ch=ch_name, eog_ch=ch_name,
                skew_criterion=idx, var_criterion=idx, kurt_criterion=idx)
