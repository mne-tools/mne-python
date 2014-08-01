from __future__ import print_function

# Author: Denis Engemann <denis.engemann@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os
import os.path as op
from functools import wraps
import warnings

from nose.tools import assert_true, assert_raises, assert_equal
from copy import deepcopy
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from scipy import stats
from itertools import product

from mne import io, Epochs, read_events, pick_types
from mne.cov import read_cov
from mne.preprocessing import (ICA, ica_find_ecg_events, ica_find_eog_events,
                               read_ica, run_ica)
from mne.preprocessing.ica import score_funcs, _check_n_pca_components
from mne.io.meas_info import Info
from mne.utils import set_log_file, check_sklearn_version, _TempDir

warnings.simplefilter('always')  # enable b/c these tests throw warnings

tempdir = _TempDir()

data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')
evoked_nf_name = op.join(data_dir, 'test-nf-ave.fif')
test_cov_name = op.join(data_dir, 'test-cov.fif')

event_id, tmin, tmax = 1, -0.2, 0.2
start, stop = 0, 6  # if stop is too small pca may fail in some cases, but
                    # we're okay on this file

score_funcs_unsuited = ['pointbiserialr', 'ansari']
try:
    from sklearn.utils.validation import NonBLASDotWarning
    warnings.simplefilter('error', NonBLASDotWarning)
except:
    pass


def requires_sklearn(function):
    """Decorator to skip test if scikit-learn >= 0.12 is not available"""
    @wraps(function)
    def dec(*args, **kwargs):
        if not check_sklearn_version(min_version='0.12'):
            from nose.plugins.skip import SkipTest
            raise SkipTest('Test %s skipped, requires scikit-learn >= 0.12'
                           % function.__name__)
        ret = function(*args, **kwargs)
        return ret
    return dec


@requires_sklearn
def test_ica_full_data_recovery():
    """Test recovery of full data when no source is rejected"""
    # Most basic recovery
    raw = io.Raw(raw_fname, preload=True).crop(0, stop, False).crop(0.5)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')[:10]
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    evoked = epochs.average()
    n_channels = 5
    data = raw._data[:n_channels].copy()
    data_epochs = epochs.get_data()
    data_evoked = evoked.data
    for method in ['fastica']:
        stuff = [(2, n_channels, True), (2, n_channels // 2, False)]
        for n_components, n_pca_components, ok in stuff:
            ica = ICA(n_components=n_components,
                      max_pca_components=n_pca_components,
                      n_pca_components=n_pca_components,
                      method=method, max_iter=1)
            with warnings.catch_warnings(record=True):
                ica.fit(raw, picks=list(range(n_channels)))
            raw2 = ica.apply(raw, exclude=[], copy=True)
            if ok:
                assert_allclose(data[:n_channels], raw2._data[:n_channels],
                                rtol=1e-10, atol=1e-15)
            else:
                diff = np.abs(data[:n_channels] - raw2._data[:n_channels])
                assert_true(np.max(diff) > 1e-14)

            ica = ICA(n_components=n_components,
                      max_pca_components=n_pca_components,
                      n_pca_components=n_pca_components)
            with warnings.catch_warnings(record=True):
                ica.fit(epochs, picks=list(range(n_channels)))
            epochs2 = ica.apply(epochs, exclude=[], copy=True)
            data2 = epochs2.get_data()[:, :n_channels]
            if ok:
                assert_allclose(data_epochs[:, :n_channels], data2,
                                rtol=1e-10, atol=1e-15)
            else:
                diff = np.abs(data_epochs[:, :n_channels] - data2)
                assert_true(np.max(diff) > 1e-14)

            evoked2 = ica.apply(evoked, exclude=[], copy=True)
            data2 = evoked2.data[:n_channels]
            if ok:
                assert_allclose(data_evoked[:n_channels], data2,
                                rtol=1e-10, atol=1e-15)
            else:
                diff = np.abs(evoked.data[:n_channels] - data2)
                assert_true(np.max(diff) > 1e-14)
    assert_raises(ValueError, ICA, method='pizza-decomposision')


@requires_sklearn
def test_ica_rank_reduction():
    """Test recovery of full data when no source is rejected"""
    # Most basic recovery
    raw = io.Raw(raw_fname, preload=True).crop(0, stop, False).crop(0.5)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')[:10]
    n_components = 5
    max_pca_components = len(picks)
    for n_pca_components in [6, 10]:
        with warnings.catch_warnings(record=True):  # non-convergence
            warnings.simplefilter('always')
            ica = ICA(n_components=n_components,
                      max_pca_components=max_pca_components,
                      n_pca_components=n_pca_components,
                      method='fastica', max_iter=1).fit(raw, picks=picks)

        rank_before = raw.estimate_rank(picks=picks)
        assert_equal(rank_before, len(picks))
        raw_clean = ica.apply(raw, copy=True)
        rank_after = raw_clean.estimate_rank(picks=picks)
        # interaction between ICA rejection and PCA components difficult
        # to preduct. Rank_after often seems to be 1 higher then
        # n_pca_components
        assert_true(n_components < n_pca_components <= rank_after <=
                    rank_before)


@requires_sklearn
def test_ica_core():
    """Test ICA on raw and epochs"""
    raw = io.Raw(raw_fname, preload=True).crop(0, stop, False).crop(1.5)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    # XXX. The None cases helped revealing bugs but are time consuming.
    test_cov = read_cov(test_cov_name)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    noise_cov = [None, test_cov]
    # removed None cases to speed up...
    n_components = [2, 1.0]  # for future dbg add cases
    max_pca_components = [3]
    picks_ = [picks]
    methods = ['fastica']
    iter_ica_params = product(noise_cov, n_components, max_pca_components,
                              picks_, methods)

    # # test init catchers
    assert_raises(ValueError, ICA, n_components=3, max_pca_components=2)
    assert_raises(ValueError, ICA, n_components=2.3, max_pca_components=2)

    # test essential core functionality
    for n_cov, n_comp, max_n, pcks, method in iter_ica_params:
      # Test ICA raw
        ica = ICA(noise_cov=n_cov, n_components=n_comp,
                  max_pca_components=max_n, n_pca_components=max_n,
                  random_state=0, method=method, max_iter=1)

        print(ica)  # to test repr

        # test fit checker
        assert_raises(RuntimeError, ica.get_sources, raw)
        assert_raises(RuntimeError, ica.get_sources, epochs)

        # test decomposition
        with warnings.catch_warnings(record=True):
            ica.fit(raw, picks=pcks, start=start, stop=stop)
            repr(ica)  # to test repr

        # test re-fit
        unmixing1 = ica.unmixing_matrix_
        with warnings.catch_warnings(record=True):
            ica.fit(raw, picks=pcks, start=start, stop=stop)
        assert_array_almost_equal(unmixing1, ica.unmixing_matrix_)

        sources = ica.get_sources(raw)[:, :][0]
        assert_true(sources.shape[0] == ica.n_components_)

        # test preload filter
        raw3 = raw.copy()
        raw3.preload = False
        assert_raises(ValueError, ica.apply, raw3,
                      include=[1, 2])

        #######################################################################
        # test epochs decomposition
        ica = ICA(noise_cov=n_cov, n_components=n_comp,
                  max_pca_components=max_n, n_pca_components=max_n,
                  random_state=0)
        with warnings.catch_warnings(record=True):
            ica.fit(epochs, picks=picks)
        data = epochs.get_data()[:, 0, :]
        n_samples = np.prod(data.shape)
        assert_equal(ica.n_samples_, n_samples)
        print(ica)  # to test repr

        sources = ica.get_sources(epochs).get_data()
        assert_true(sources.shape[1] == ica.n_components_)

        assert_raises(ValueError, ica.score_sources, epochs,
                      target=np.arange(1))

        # test preload filter
        epochs3 = epochs.copy()
        epochs3.preload = False
        assert_raises(ValueError, ica.apply, epochs3,
                      include=[1, 2])

    # test for bug with whitener updating
    _pre_whitener = ica._pre_whitener.copy()
    epochs._data[:, 0, 10:15] *= 1e12
    ica.apply(epochs, copy=True)
    assert_array_equal(_pre_whitener, ica._pre_whitener)

    # test expl. var threshold leading to empty sel
    ica.n_components = 0.1
    assert_raises(RuntimeError, ica.fit, epochs)

    offender = 1, 2, 3,
    assert_raises(ValueError, ica.get_sources, offender)
    assert_raises(ValueError, ica.fit, offender)
    assert_raises(ValueError, ica.apply, offender)


@requires_sklearn
def test_ica_additional():
    """Test additional ICA functionality"""
    stop2 = 500
    raw = io.Raw(raw_fname, preload=True).crop(0, stop, False).crop(1.5)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    test_cov = read_cov(test_cov_name)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    # for testing eog functionality
    picks2 = pick_types(raw.info, meg=True, stim=False, ecg=False,
                        eog=True, exclude='bads')
    epochs_eog = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks2,
                        baseline=(None, 0), preload=True)

    test_cov2 = deepcopy(test_cov)
    ica = ICA(noise_cov=test_cov2, n_components=3, max_pca_components=4,
              n_pca_components=4)
    assert_true(ica.info is None)
    with warnings.catch_warnings(record=True):
        ica.fit(raw, picks[:5])
    assert_true(isinstance(ica.info, Info))
    assert_true(ica.n_components_ < 5)

    ica = ICA(n_components=3, max_pca_components=4,
              n_pca_components=4)
    assert_raises(RuntimeError, ica.save, '')
    with warnings.catch_warnings(record=True):
        ica.fit(raw, picks=None, start=start, stop=stop2)

    # test warnings on bad filenames
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        ica_badname = op.join(op.dirname(tempdir), 'test-bad-name.fif.gz')
        ica.save(ica_badname)
        read_ica(ica_badname)
    assert_true(len(w) == 2)

    # test decim
    ica = ICA(n_components=3, max_pca_components=4,
              n_pca_components=4)
    raw_ = raw.copy()
    for _ in range(3):
        raw_.append(raw_)
    n_samples = raw_._data.shape[1]
    with warnings.catch_warnings(record=True):
        ica.fit(raw, picks=None, decim=3)
    assert_true(raw_._data.shape[1], n_samples)

    # test expl var
    ica = ICA(n_components=1.0, max_pca_components=4,
              n_pca_components=4)
    with warnings.catch_warnings(record=True):
        ica.fit(raw, picks=None, decim=3)
    assert_true(ica.n_components_ == 4)

    # epochs extraction from raw fit
    assert_raises(RuntimeError, ica.get_sources, epochs)
    # test reading and writing
    test_ica_fname = op.join(op.dirname(tempdir), 'test-ica.fif')
    for cov in (None, test_cov):
        ica = ICA(noise_cov=cov, n_components=2, max_pca_components=4,
                  n_pca_components=4)
        with warnings.catch_warnings(record=True):  # ICA does not converge
            ica.fit(raw, picks=picks, start=start, stop=stop2)
        sources = ica.get_sources(epochs).get_data()
        assert_true(ica.mixing_matrix_.shape == (2, 2))
        assert_true(ica.unmixing_matrix_.shape == (2, 2))
        assert_true(ica.pca_components_.shape == (4, len(picks)))
        assert_true(sources.shape[1] == ica.n_components_)

        for exclude in [[], [0]]:
            ica.exclude = [0]
            ica.save(test_ica_fname)
            ica_read = read_ica(test_ica_fname)
            assert_true(ica.exclude == ica_read.exclude)

            ica.exclude = []
            ica.apply(raw, exclude=[1])
            assert_true(ica.exclude == [])

            ica.exclude = [0, 1]
            ica.apply(raw, exclude=[1])
            assert_true(ica.exclude == [0, 1])

            ica_raw = ica.get_sources(raw)
            assert_true(ica.exclude == [ica_raw.ch_names.index(e) for e in
                                        ica_raw.info['bads']])

        # test filtering
        d1 = ica_raw._data[0].copy()
        with warnings.catch_warnings(record=True):  # dB warning
            ica_raw.filter(4, 20)
        assert_true((d1 != ica_raw._data[0]).any())
        d1 = ica_raw._data[0].copy()
        with warnings.catch_warnings(record=True):  # dB warning
            ica_raw.notch_filter([10])
        assert_true((d1 != ica_raw._data[0]).any())

        ica.n_pca_components = 2
        ica.save(test_ica_fname)
        ica_read = read_ica(test_ica_fname)
        assert_true(ica.n_pca_components == ica_read.n_pca_components)

        # check type consistency
        attrs = ('mixing_matrix_ unmixing_matrix_ pca_components_ '
                 'pca_explained_variance_ _pre_whitener')
        f = lambda x, y: getattr(x, y).dtype
        for attr in attrs.split():
            assert_equal(f(ica_read, attr), f(ica, attr))

        ica.n_pca_components = 4
        ica_read.n_pca_components = 4

        ica.exclude = []
        ica.save(test_ica_fname)
        ica_read = read_ica(test_ica_fname)
        for attr in ['mixing_matrix_', 'unmixing_matrix_', 'pca_components_',
                     'pca_mean_', 'pca_explained_variance_',
                     '_pre_whitener']:
            assert_array_almost_equal(getattr(ica, attr),
                                      getattr(ica_read, attr))

        assert_true(ica.ch_names == ica_read.ch_names)
        assert_true(isinstance(ica_read.info, Info))

        sources = ica.get_sources(raw)[:, :][0]
        sources2 = ica_read.get_sources(raw)[:, :][0]
        assert_array_almost_equal(sources, sources2)

        _raw1 = ica.apply(raw, exclude=[1])
        _raw2 = ica_read.apply(raw, exclude=[1])
        assert_array_almost_equal(_raw1[:, :][0], _raw2[:, :][0])

    os.remove(test_ica_fname)
    # check scrore funcs
    for name, func in score_funcs.items():
        if name in score_funcs_unsuited:
            continue
        scores = ica.score_sources(raw, target='EOG 061', score_func=func,
                                   start=0, stop=10)
        assert_true(ica.n_components_ == len(scores))

    # check univariate stats
    scores = ica.score_sources(raw, score_func=stats.skew)
    # check exception handling
    assert_raises(ValueError, ica.score_sources, raw,
                  target=np.arange(1))

    params = []
    params += [(None, -1, slice(2), [0, 1])]  # varicance, kurtosis idx params
    params += [(None, 'MEG 1531')]  # ECG / EOG channel params
    for idx, ch_name in product(*params):
        ica.detect_artifacts(raw, start_find=0, stop_find=50, ecg_ch=ch_name,
                             eog_ch=ch_name, skew_criterion=idx,
                             var_criterion=idx, kurt_criterion=idx)
    with warnings.catch_warnings(record=True):
        idx, scores = ica.find_bads_ecg(raw, method='ctps')
        assert_equal(len(scores), ica.n_components_)
        idx, scores = ica.find_bads_ecg(raw, method='correlation')
        assert_equal(len(scores), ica.n_components_)
        idx, scores = ica.find_bads_ecg(epochs, method='ctps')
        assert_equal(len(scores), ica.n_components_)
        assert_raises(ValueError, ica.find_bads_ecg, epochs.average(),
                      method='ctps')
        assert_raises(ValueError, ica.find_bads_ecg, raw,
                      method='crazy-coupling')

        idx, scores = ica.find_bads_eog(raw)
        assert_equal(len(scores), ica.n_components_)
        raw.info['chs'][raw.ch_names.index('EOG 061') - 1]['kind'] = 202
        idx, scores = ica.find_bads_eog(raw)
        assert_true(isinstance(scores, list))
        assert_equal(len(scores[0]), ica.n_components_)

    # check score funcs
    for name, func in score_funcs.items():
        if name in score_funcs_unsuited:
            continue
        scores = ica.score_sources(epochs_eog, target='EOG 061',
                                   score_func=func)
        assert_true(ica.n_components_ == len(scores))

    # check univariate stats
    scores = ica.score_sources(epochs, score_func=stats.skew)

    # check exception handling
    assert_raises(ValueError, ica.score_sources, epochs,
                  target=np.arange(1))

    # ecg functionality
    ecg_scores = ica.score_sources(raw, target='MEG 1531',
                                   score_func='pearsonr')

    with warnings.catch_warnings(record=True):  # filter attenuation warning
        ecg_events = ica_find_ecg_events(raw,
                                         sources[np.abs(ecg_scores).argmax()])

    assert_true(ecg_events.ndim == 2)

    # eog functionality
    eog_scores = ica.score_sources(raw, target='EOG 061',
                                   score_func='pearsonr')
    with warnings.catch_warnings(record=True):  # filter attenuation warning
        eog_events = ica_find_eog_events(raw,
                                         sources[np.abs(eog_scores).argmax()])

    assert_true(eog_events.ndim == 2)

    # Test ica fiff export
    ica_raw = ica.get_sources(raw, start=0, stop=100)
    assert_true(ica_raw.last_samp - ica_raw.first_samp == 100)
    assert_true(len(ica_raw._filenames) == 0)  # API consistency
    ica_chans = [ch for ch in ica_raw.ch_names if 'ICA' in ch]
    assert_true(ica.n_components_ == len(ica_chans))
    test_ica_fname = op.join(op.abspath(op.curdir), 'test-ica_raw.fif')
    ica.n_components = np.int32(ica.n_components)
    ica_raw.save(test_ica_fname, overwrite=True)
    ica_raw2 = io.Raw(test_ica_fname, preload=True)
    assert_allclose(ica_raw._data, ica_raw2._data, rtol=1e-5, atol=1e-4)
    ica_raw2.close()
    os.remove(test_ica_fname)

    # Test ica epochs export
    ica_epochs = ica.get_sources(epochs)
    assert_true(ica_epochs.events.shape == epochs.events.shape)
    ica_chans = [ch for ch in ica_epochs.ch_names if 'ICA' in ch]
    assert_true(ica.n_components_ == len(ica_chans))
    assert_true(ica.n_components_ == ica_epochs.get_data().shape[1])
    assert_true(ica_epochs.raw is None)
    assert_true(ica_epochs.preload is True)

    # test float n pca components
    ica.pca_explained_variance_ = np.array([0.2] * 5)
    ica.n_components_ = 0
    for ncomps, expected in [[0.3, 1], [0.9, 4], [1, 1]]:
        ncomps_ = _check_n_pca_components(ica, ncomps)
        assert_true(ncomps_ == expected)


@requires_sklearn
def test_run_ica():
    """Test run_ica function"""
    raw = io.Raw(raw_fname, preload=True).crop(0, stop, False).crop(1.5)
    params = []
    params += [(None, -1, slice(2), [0, 1])]  # varicance, kurtosis idx
    params += [(None, 'MEG 1531')]  # ECG / EOG channel params
    for idx, ch_name in product(*params):
        warnings.simplefilter('always')
        with warnings.catch_warnings(record=True):
            run_ica(raw, n_components=2, start=0, stop=6, start_find=0,
                    stop_find=5, ecg_ch=ch_name, eog_ch=ch_name,
                    skew_criterion=idx, var_criterion=idx, kurt_criterion=idx)


@requires_sklearn
def test_ica_reject_buffer():
    """Test ICA data raw buffer rejection"""
    raw = io.Raw(raw_fname, preload=True).crop(0, stop, False).crop(1.5)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    ica = ICA(n_components=3, max_pca_components=4, n_pca_components=4)
    raw._data[2, 1000:1005] = 5e-12
    drop_log = op.join(op.dirname(tempdir), 'ica_drop.log')
    set_log_file(drop_log, overwrite=True)
    with warnings.catch_warnings(record=True):
        ica.fit(raw, picks[:5], reject=dict(mag=2.5e-12), decim=2,
                tstep=0.01, verbose=True)
    assert_true(raw._data[:5, ::2].shape[1] - 4 == ica.n_samples_)
    with open(drop_log) as fid:
        log = [l for l in fid if 'detected' in l]
    assert_equal(len(log), 1)


@requires_sklearn
def test_ica_twice():
    """Test running ICA twice"""
    raw = io.Raw(raw_fname, preload=True).crop(0, stop, False).crop(1.5)
    picks = pick_types(raw.info, meg='grad', exclude='bads')
    n_components = 0.9
    max_pca_components = None
    n_pca_components = 1.1
    with warnings.catch_warnings(record=True):
        ica1 = ICA(n_components=n_components,
                   max_pca_components=max_pca_components,
                   n_pca_components=n_pca_components, random_state=0)

        ica1.fit(raw, picks=picks, decim=3)
        raw_new = ica1.apply(raw, n_pca_components=n_pca_components)
        ica2 = ICA(n_components=n_components,
                   max_pca_components=max_pca_components,
                   n_pca_components=1.0, random_state=0)
        ica2.fit(raw_new, picks=picks, decim=3)
        assert_equal(ica1.n_components_, ica2.n_components_)
