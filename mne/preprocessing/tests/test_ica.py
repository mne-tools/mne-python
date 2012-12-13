# Author: Denis Engemann <d.engemann@fz-juelich.de>
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
import tempfile
tempdir = tempfile.mkdtemp()

from mne import fiff, Epochs, read_events, cov
from mne.preprocessing import ICA, ica_find_ecg_events, ica_find_eog_events,\
                              read_ica
from mne.preprocessing.ica import score_funcs

have_sklearn = True
try:
    import sklearn
except ImportError:
    have_sklearn = False

sklearn_test = np.testing.dec.skipif(not have_sklearn,
                                     'scikit-learn not installed')

raw_fname = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data',
                    'test_raw.fif')
event_name = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests',
                     'data', 'test-eve.fif')
evoked_nf_name = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests',
                         'data', 'test-nf-ave.fif')

test_cov_name = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests',
                        'data', 'test-cov.fif')

event_id, tmin, tmax = 1, -0.2, 0.5
start, stop = 0, 8  # if stop is too small pca may fail in some cases, but
                    # we're okay on this file
raw = fiff.Raw(raw_fname, preload=True).crop(0, stop, False)

events = read_events(event_name)
picks = fiff.pick_types(raw.info, meg=True, stim=False,
                        ecg=False, eog=False, exclude=raw.info['bads'])

# for testing eog functionality
picks2 = fiff.pick_types(raw.info, meg=True, stim=False,
                        ecg=False, eog=True, exclude=raw.info['bads'])

reject = dict(grad=1000e-12, mag=4e-12, eeg=80e-6, eog=150e-6)
flat = dict(grad=1e-15, mag=1e-15)

test_cov = cov.read_cov(test_cov_name)
epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                baseline=(None, 0), preload=True)

epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                         baseline=(None, 0), preload=True)

epochs_eog = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks2,
                baseline=(None, 0), preload=True)


@sklearn_test
def test_ica_core():
    """Test ICA on raw and epochs
    """
    # setup parameter
    # XXX. The None cases helped revealing bugs but are time consuming.
    noise_cov = [None, test_cov]
    # removed None cases to speed up...
    n_components = [3, 1.0]  # for future dbg add cases
    max_n_components = [4]
    picks_ = [picks]
    iter_ica_params = product(noise_cov, n_components, max_n_components,
                           picks_)

    # test init catchers
    assert_raises(ValueError, ICA, n_components=3, max_n_components=2)
    assert_raises(ValueError, ICA, n_components=1.3, max_n_components=2)

    # test essential core functionality
    for n_cov, n_comp, max_n, pcks in iter_ica_params:
      # Test ICA raw
        ica = ICA(noise_cov=n_cov, n_components=n_comp, max_n_components=max_n,
                  random_state=0)

        print ica  # to test repr
        # test fit checker
        assert_raises(RuntimeError, ica.get_sources_raw, raw)
        assert_raises(RuntimeError, ica.get_sources_epochs, epochs)

        # test decomposition
        ica.decompose_raw(raw, picks=pcks, start=start, stop=stop)
        # test re-init exception
        assert_raises(RuntimeError, ica.decompose_raw, raw, picks=picks)

        sources = ica.get_sources_raw(raw)
        assert_true(sources.shape[0] == ica.n_components)

        # test preload filter
        raw3 = raw.copy()
        raw3._preloaded = False
        assert_raises(ValueError, ica.pick_sources_raw, raw3,
                      include=[1, 2], n_pca_components=ica.n_components)

        for excl, incl in (([], []), ([], [1, 2]), ([1, 2], [])):
            raw2 = ica.pick_sources_raw(raw, exclude=excl, include=incl,
                                        copy=True,
                                        n_pca_components=ica.n_components)

            assert_array_almost_equal(raw2[:, :][1], raw[:, :][1])

        #######################################################################
        # test epochs decomposition

        # test re-init exception
        assert_raises(RuntimeError, ica.decompose_epochs, epochs, picks=picks)
        ica = ICA(noise_cov=n_cov, n_components=n_comp, max_n_components=max_n,
                  random_state=0)

        ica.decompose_epochs(epochs, picks=picks)
        # test pick block after epochs fit
        assert_raises(ValueError, ica.pick_sources_raw, raw,
                    n_pca_components=ica.n_components)

        sources = ica.get_sources_epochs(epochs)
        assert_true(sources.shape[1] == ica.n_components)

        assert_raises(ValueError, ica.find_sources_epochs, epochs,
                      target=np.arange(1))

        # test preload filter
        epochs3 = epochs.copy()
        epochs3.preload = False
        assert_raises(ValueError, ica.pick_sources_epochs, epochs3,
                      include=[1, 2], n_pca_components=ica.n_components)

        # test source picking
        for excl, incl in (([], []), ([], [1, 2]), ([1, 2], [])):
            epochs2 = ica.pick_sources_epochs(epochs, exclude=excl,
                                              include=incl, copy=True,
                                              n_pca_components=ica.n_components)

            assert_array_almost_equal(epochs2.get_data(),
                                      epochs.get_data())


@sklearn_test
def test_ica_additional():
    """Test additional functionality
    """
    stop2 = 500

    test_cov2 = deepcopy(test_cov)
    ica = ICA(noise_cov=test_cov2, n_components=3, max_n_components=4)
    ica.decompose_raw(raw, picks[:5])
    assert_true(ica.n_components < 5)

    ica = ICA(n_components=3, max_n_components=4)
    assert_raises(RuntimeError, ica.save, '')
    ica.decompose_raw(raw, picks=None, start=start, stop=stop2)

    # epochs extraction from raw fit
    assert_raises(RuntimeError, ica.get_sources_epochs, epochs)

    # test reading and writing
    test_ica_fname = op.join(op.dirname(tempdir), 'ica_test.fif')
    for cov in (None, test_cov):
        ica = ICA(noise_cov=cov, n_components=3, max_n_components=4)
        ica.decompose_raw(raw, picks=picks, start=start, stop=stop2)
        sources = ica.get_sources_epochs(epochs)
        assert_true(sources.shape[1] == ica.n_components)

        ica.save(test_ica_fname)
        ica_read = read_ica(test_ica_fname)

        assert_true(ica.ch_names == ica_read.ch_names)

        try:
            a = ica._ica.components_
            b = ica_read._ica.components_
        except:
            a = ica._ica.unmixing_matrix_
            b = ica_read._ica.unmixing_matrix_

        assert_array_equal(a, b)
        assert_array_equal(ica._mixing, ica_read._mixing)
        assert_array_equal(ica._pca.components_,
                           ica_read._pca.components_)
        assert_array_equal(ica._pca.mean_,
                                  ica_read._pca.mean_)
        assert_array_equal(ica._pca.explained_variance_,
                                  ica_read._pca.explained_variance_)
        assert_array_equal(ica._pre_whitener,
                                  ica_read._pre_whitener)

        assert_raises(RuntimeError, ica_read.decompose_raw, raw)
        sources = ica.get_sources_raw(raw)
        sources2 = ica_read.get_sources_raw(raw)
        assert_array_almost_equal(sources, sources2)

        _raw1 = ica.pick_sources_raw(raw, exclude=[1], n_pca_components=4)
        _raw2 = ica_read.pick_sources_raw(raw, exclude=[1], n_pca_components=4)
        assert_array_almost_equal(_raw1[:, :][0], _raw2[:, :][0])

    os.remove(test_ica_fname)
    # score funcs raw

    sfunc_test = [ica.find_sources_raw(raw, target='EOG 061', score_func=n,
            start=0, stop=10) for  n, f in score_funcs.items()]

    # check lenght of scores
    [assert_true(ica.n_components == len(scores)) for scores in sfunc_test]

    # check univariate stats
    scores = ica.find_sources_raw(raw, score_func=stats.skew)
    # check exception handling
    assert_raises(ValueError, ica.find_sources_raw, raw,
                  target=np.arange(1))

    ## score funcs epochs ##

    # check lenght of scores
    sfunc_test = [ica.find_sources_epochs(epochs_eog, target='EOG 061',
                    score_func=n) for n, f in score_funcs.items()]

    # check lenght of scores
    [assert_true(ica.n_components == len(scores)) for scores in sfunc_test]

    # check univariat stats
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
    raw3 = raw.copy()
    raw3._preloaded = False
    assert_raises(ValueError, ica.export_sources, raw3, start=0, stop=100)
    ica_raw = ica.export_sources(raw, start=0, stop=100)
    assert_true(ica_raw.last_samp - ica_raw.first_samp == 100)
    ica_chans = [ch for ch in ica_raw.ch_names if 'ICA' in ch]
    assert_true(ica.n_components == len(ica_chans))
    test_ica_fname = op.join(op.abspath(op.curdir), 'test_ica.fif')
    ica_raw.save(test_ica_fname)
    ica_raw2 = fiff.Raw(test_ica_fname, preload=True)
    assert_array_almost_equal(ica_raw._data, ica_raw2._data)
    os.remove(test_ica_fname)

    # regression test for plot method
    assert_raises(ValueError, ica.plot_sources_raw, raw,
                  order=np.arange(50))
    assert_raises(ValueError, ica.plot_sources_epochs, epochs,
                  order=np.arange(50))
