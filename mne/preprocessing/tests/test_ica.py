from __future__ import print_function

# Author: Denis Engemann <denis.engemann@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os
import os.path as op
import warnings

from nose.tools import (assert_true, assert_raises, assert_equal, assert_false,
                        assert_not_equal, assert_is_none)
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from scipy import stats
from itertools import product

from mne import (Epochs, read_events, pick_types, create_info, EpochsArray,
                 EvokedArray, Annotations)
from mne.cov import read_cov
from mne.preprocessing import (ICA, ica_find_ecg_events, ica_find_eog_events,
                               read_ica, run_ica)
from mne.preprocessing.ica import (get_score_funcs, corrmap, _sort_components,
                                   _ica_explained_variance)
from mne.io import read_raw_fif, Info, RawArray
from mne.io.meas_info import _kind_dict
from mne.io.pick import _DATA_CH_TYPES_SPLIT
from mne.tests.common import assert_naming
from mne.utils import (catch_logging, _TempDir, requires_sklearn, slow_test,
                       run_tests_if_main)

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings

data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')
test_cov_name = op.join(data_dir, 'test-cov.fif')

event_id, tmin, tmax = 1, -0.2, 0.2
# if stop is too small pca may fail in some cases, but we're okay on this file
start, stop = 0, 6
score_funcs_unsuited = ['pointbiserialr', 'ansari']
try:
    from sklearn.utils.validation import NonBLASDotWarning
    warnings.simplefilter('error', NonBLASDotWarning)
except:
    pass


@requires_sklearn
def test_ica_full_data_recovery():
    """Test recovery of full data when no source is rejected."""
    # Most basic recovery
    raw = read_raw_fif(raw_fname).crop(0.5, stop).load_data()
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')[:10]
    with warnings.catch_warnings(record=True):  # bad proj
        epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), preload=True)
    evoked = epochs.average()
    n_channels = 5
    data = raw._data[:n_channels].copy()
    data_epochs = epochs.get_data()
    data_evoked = evoked.data
    raw.annotations = Annotations([0.5], [0.5], ['BAD'])
    for method in ['fastica']:
        stuff = [(2, n_channels, True), (2, n_channels // 2, False)]
        for n_components, n_pca_components, ok in stuff:
            ica = ICA(n_components=n_components,
                      max_pca_components=n_pca_components,
                      n_pca_components=n_pca_components,
                      method=method, max_iter=1)
            with warnings.catch_warnings(record=True):
                ica.fit(raw, picks=list(range(n_channels)))
            raw2 = ica.apply(raw.copy(), exclude=[])
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
            epochs2 = ica.apply(epochs.copy(), exclude=[])
            data2 = epochs2.get_data()[:, :n_channels]
            if ok:
                assert_allclose(data_epochs[:, :n_channels], data2,
                                rtol=1e-10, atol=1e-15)
            else:
                diff = np.abs(data_epochs[:, :n_channels] - data2)
                assert_true(np.max(diff) > 1e-14)

            evoked2 = ica.apply(evoked.copy(), exclude=[])
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
    """Test recovery ICA rank reduction."""
    # Most basic recovery
    raw = read_raw_fif(raw_fname).crop(0.5, stop).load_data()
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
        raw_clean = ica.apply(raw.copy())
        rank_after = raw_clean.estimate_rank(picks=picks)
        # interaction between ICA rejection and PCA components difficult
        # to preduct. Rank_after often seems to be 1 higher then
        # n_pca_components
        assert_true(n_components < n_pca_components <= rank_after <=
                    rank_before)


@requires_sklearn
def test_ica_reset():
    """Test ICA resetting."""
    raw = read_raw_fif(raw_fname).crop(0.5, stop).load_data()
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')[:10]

    run_time_attrs = (
        '_pre_whitener',
        'unmixing_matrix_',
        'mixing_matrix_',
        'n_components_',
        'n_samples_',
        'pca_components_',
        'pca_explained_variance_',
        'pca_mean_'
    )
    with warnings.catch_warnings(record=True):
        ica = ICA(
            n_components=3, max_pca_components=3, n_pca_components=3,
            method='fastica', max_iter=1).fit(raw, picks=picks)

    assert_true(all(hasattr(ica, attr) for attr in run_time_attrs))
    assert_not_equal(ica.labels_, None)
    ica._reset()
    assert_true(not any(hasattr(ica, attr) for attr in run_time_attrs))
    assert_not_equal(ica.labels_, None)


@requires_sklearn
def test_ica_core():
    """Test ICA on raw and epochs."""
    raw = read_raw_fif(raw_fname).crop(1.5, stop).load_data()
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
        assert_raises(ValueError, ica.__contains__, 'mag')

        print(ica)  # to test repr

        # test fit checker
        assert_raises(RuntimeError, ica.get_sources, raw)
        assert_raises(RuntimeError, ica.get_sources, epochs)

        # test decomposition
        with warnings.catch_warnings(record=True):
            ica.fit(raw, picks=pcks, start=start, stop=stop)
            repr(ica)  # to test repr
        assert_true('mag' in ica)  # should now work without error

        # test re-fit
        unmixing1 = ica.unmixing_matrix_
        with warnings.catch_warnings(record=True):
            ica.fit(raw, picks=pcks, start=start, stop=stop)
        assert_array_almost_equal(unmixing1, ica.unmixing_matrix_)

        raw_sources = ica.get_sources(raw)
        # test for #3804
        assert_equal(raw_sources._filenames, [None])
        print(raw_sources)

        sources = raw_sources[:, :][0]
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
    ica.apply(epochs.copy())
    assert_array_equal(_pre_whitener, ica._pre_whitener)

    # test expl. var threshold leading to empty sel
    ica.n_components = 0.1
    assert_raises(RuntimeError, ica.fit, epochs)

    offender = 1, 2, 3,
    assert_raises(ValueError, ica.get_sources, offender)
    assert_raises(ValueError, ica.fit, offender)
    assert_raises(ValueError, ica.apply, offender)


@slow_test
@requires_sklearn
def test_ica_additional():
    """Test additional ICA functionality."""
    import matplotlib.pyplot as plt
    tempdir = _TempDir()
    stop2 = 500
    raw = read_raw_fif(raw_fname).crop(1.5, stop).load_data()
    # XXX This breaks the tests :(
    # raw.info['bads'] = [raw.ch_names[1]]
    test_cov = read_cov(test_cov_name)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    # test if n_components=None works
    with warnings.catch_warnings(record=True):
        ica = ICA(n_components=None,
                  max_pca_components=None,
                  n_pca_components=None, random_state=0)
        ica.fit(epochs, picks=picks, decim=3)
    # for testing eog functionality
    picks2 = pick_types(raw.info, meg=True, stim=False, ecg=False,
                        eog=True, exclude='bads')
    epochs_eog = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks2,
                        baseline=(None, 0), preload=True)

    test_cov2 = test_cov.copy()
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
        ica.fit(raw, picks=[1, 2, 3, 4, 5], start=start, stop=stop2)

    # test corrmap
    ica2 = ica.copy()
    ica3 = ica.copy()
    corrmap([ica, ica2], (0, 0), threshold='auto', label='blinks', plot=True,
            ch_type="mag")
    corrmap([ica, ica2], (0, 0), threshold=2, plot=False, show=False)
    assert_true(ica.labels_["blinks"] == ica2.labels_["blinks"])
    assert_true(0 in ica.labels_["blinks"])
    # test retrieval of component maps as arrays
    components = ica.get_components()
    template = components[:, 0]
    EvokedArray(components, ica.info, tmin=0.).plot_topomap([0])

    corrmap([ica, ica3], template, threshold='auto', label='blinks', plot=True,
            ch_type="mag")
    assert_true(ica2.labels_["blinks"] == ica3.labels_["blinks"])

    plt.close('all')

    # test warnings on bad filenames
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        ica_badname = op.join(op.dirname(tempdir), 'test-bad-name.fif.gz')
        ica.save(ica_badname)
        read_ica(ica_badname)
    assert_naming(w, 'test_ica.py', 2)

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
    ica_var = _ica_explained_variance(ica, raw, normalize=True)
    assert_true(np.all(ica_var[:-1] >= ica_var[1:]))

    # test ica sorting
    ica.exclude = [0]
    ica.labels_ = dict(blink=[0], think=[1])
    ica_sorted = _sort_components(ica, [3, 2, 1, 0], copy=True)
    assert_equal(ica_sorted.exclude, [3])
    assert_equal(ica_sorted.labels_, dict(blink=[3], think=[2]))

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
            ica.exclude = exclude
            ica.labels_ = {'foo': [0]}
            ica.save(test_ica_fname)
            ica_read = read_ica(test_ica_fname)
            assert_true(ica.exclude == ica_read.exclude)
            assert_equal(ica.labels_, ica_read.labels_)
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
        ica_raw.filter(4, 20, l_trans_bandwidth='auto',
                       h_trans_bandwidth='auto', filter_length='auto',
                       phase='zero', fir_window='hamming')
        assert_equal(ica_raw.info['lowpass'], 20.)
        assert_equal(ica_raw.info['highpass'], 4.)
        assert_true((d1 != ica_raw._data[0]).any())
        d1 = ica_raw._data[0].copy()
        ica_raw.notch_filter([10], filter_length='auto', trans_bandwidth=10,
                             phase='zero', fir_window='hamming')
        assert_true((d1 != ica_raw._data[0]).any())

        ica.n_pca_components = 2
        ica.method = 'fake'
        ica.save(test_ica_fname)
        ica_read = read_ica(test_ica_fname)
        assert_true(ica.n_pca_components == ica_read.n_pca_components)
        assert_equal(ica.method, ica_read.method)
        assert_equal(ica.labels_, ica_read.labels_)

        # check type consistency
        attrs = ('mixing_matrix_ unmixing_matrix_ pca_components_ '
                 'pca_explained_variance_ _pre_whitener')

        def f(x, y):
            return getattr(x, y).dtype

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
    for name, func in get_score_funcs().items():
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

    evoked = epochs.average()
    evoked_data = evoked.data.copy()
    raw_data = raw[:][0].copy()
    epochs_data = epochs.get_data().copy()
    with warnings.catch_warnings(record=True):
        idx, scores = ica.find_bads_ecg(raw, method='ctps')
        assert_equal(len(scores), ica.n_components_)
        idx, scores = ica.find_bads_ecg(raw, method='correlation')
        assert_equal(len(scores), ica.n_components_)

        idx, scores = ica.find_bads_eog(raw)
        assert_equal(len(scores), ica.n_components_)

        idx, scores = ica.find_bads_ecg(epochs, method='ctps')
        assert_equal(len(scores), ica.n_components_)
        assert_raises(ValueError, ica.find_bads_ecg, epochs.average(),
                      method='ctps')
        assert_raises(ValueError, ica.find_bads_ecg, raw,
                      method='crazy-coupling')

        raw.info['chs'][raw.ch_names.index('EOG 061') - 1]['kind'] = 202
        idx, scores = ica.find_bads_eog(raw)
        assert_true(isinstance(scores, list))
        assert_equal(len(scores[0]), ica.n_components_)

        idx, scores = ica.find_bads_eog(evoked, ch_name='MEG 1441')
        assert_equal(len(scores), ica.n_components_)

        idx, scores = ica.find_bads_ecg(evoked, method='correlation')
        assert_equal(len(scores), ica.n_components_)

    assert_array_equal(raw_data, raw[:][0])
    assert_array_equal(epochs_data, epochs.get_data())
    assert_array_equal(evoked_data, evoked.data)

    # check score funcs
    for name, func in get_score_funcs().items():
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
    assert_equal(len(ica_raw._filenames), 1)  # API consistency
    ica_chans = [ch for ch in ica_raw.ch_names if 'ICA' in ch]
    assert_true(ica.n_components_ == len(ica_chans))
    test_ica_fname = op.join(op.abspath(op.curdir), 'test-ica_raw.fif')
    ica.n_components = np.int32(ica.n_components)
    ica_raw.save(test_ica_fname, overwrite=True)
    ica_raw2 = read_raw_fif(test_ica_fname, preload=True)
    assert_allclose(ica_raw._data, ica_raw2._data, rtol=1e-5, atol=1e-4)
    ica_raw2.close()
    os.remove(test_ica_fname)

    # Test ica epochs export
    ica_epochs = ica.get_sources(epochs)
    assert_true(ica_epochs.events.shape == epochs.events.shape)
    ica_chans = [ch for ch in ica_epochs.ch_names if 'ICA' in ch]
    assert_true(ica.n_components_ == len(ica_chans))
    assert_true(ica.n_components_ == ica_epochs.get_data().shape[1])
    assert_true(ica_epochs._raw is None)
    assert_true(ica_epochs.preload is True)

    # test float n pca components
    ica.pca_explained_variance_ = np.array([0.2] * 5)
    ica.n_components_ = 0
    for ncomps, expected in [[0.3, 1], [0.9, 4], [1, 1]]:
        ncomps_ = ica._check_n_pca_components(ncomps)
        assert_true(ncomps_ == expected)

    ica = ICA()
    ica.fit(raw, picks=picks[:5])
    ica.find_bads_ecg(raw)
    ica.find_bads_eog(epochs, ch_name='MEG 0121')
    assert_array_equal(raw_data, raw[:][0])

    raw.drop_channels(['MEG 0122'])
    assert_raises(RuntimeError, ica.find_bads_eog, raw)
    assert_raises(RuntimeError, ica.find_bads_ecg, raw)


@requires_sklearn
def test_run_ica():
    """Test run_ica function."""
    raw = read_raw_fif(raw_fname).crop(1.5, stop).load_data()
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
    """Test ICA data raw buffer rejection."""
    raw = read_raw_fif(raw_fname).crop(1.5, stop).load_data()
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    ica = ICA(n_components=3, max_pca_components=4, n_pca_components=4)
    raw._data[2, 1000:1005] = 5e-12
    with catch_logging() as drop_log:
        with warnings.catch_warnings(record=True):
            ica.fit(raw, picks[:5], reject=dict(mag=2.5e-12), decim=2,
                    tstep=0.01, verbose=True, reject_by_annotation=False)
        assert_true(raw._data[:5, ::2].shape[1] - 4 == ica.n_samples_)
    log = [l for l in drop_log.getvalue().split('\n') if 'detected' in l]
    assert_equal(len(log), 1)


@requires_sklearn
def test_ica_twice():
    """Test running ICA twice."""
    raw = read_raw_fif(raw_fname).crop(1.5, stop).load_data()
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


@requires_sklearn
def test_fit_params():
    """Test fit_params for ICA."""
    assert_raises(ValueError, ICA, fit_params=dict(extended=True))
    fit_params = {}
    ICA(fit_params=fit_params)  # test no side effects
    assert_equal(fit_params, {})


@requires_sklearn
def test_bad_channels():
    """Test exception when unsupported channels are used."""
    chs = [i for i in _kind_dict]
    data_chs = _DATA_CH_TYPES_SPLIT + ['eog']
    chs_bad = list(set(chs) - set(data_chs))
    info = create_info(len(chs), 500, chs)
    data = np.random.rand(len(chs), 50)
    raw = RawArray(data, info)
    data = np.random.rand(100, len(chs), 50)
    epochs = EpochsArray(data, info)

    n_components = 0.9
    ica = ICA(n_components=n_components, method='fastica')

    for inst in [raw, epochs]:
        for ch in chs_bad:
            # Test case for only bad channels
            picks_bad1 = pick_types(inst.info, meg=False,
                                    **{str(ch): True})
            # Test case for good and bad channels
            picks_bad2 = pick_types(inst.info, meg=True,
                                    **{str(ch): True})
            assert_raises(ValueError, ica.fit, inst, picks=picks_bad1)
            assert_raises(ValueError, ica.fit, inst, picks=picks_bad2)
        assert_raises(ValueError, ica.fit, inst, picks=[])


@requires_sklearn
def test_eog_channel():
    """Test that EOG channel is included when performing ICA."""
    raw = read_raw_fif(raw_fname, preload=True)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=True, ecg=False,
                       eog=True, exclude='bads')
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    n_components = 0.9
    ica = ICA(n_components=n_components, method='fastica')
    # Test case for MEG and EOG data. Should have EOG channel
    for inst in [raw, epochs]:
        picks1a = pick_types(inst.info, meg=True, stim=False, ecg=False,
                             eog=False, exclude='bads')[:4]
        picks1b = pick_types(inst.info, meg=False, stim=False, ecg=False,
                             eog=True, exclude='bads')
        picks1 = np.append(picks1a, picks1b)
        ica.fit(inst, picks=picks1)
        assert_true(any('EOG' in ch for ch in ica.ch_names))
    # Test case for MEG data. Should have no EOG channel
    for inst in [raw, epochs]:
        picks1 = pick_types(inst.info, meg=True, stim=False, ecg=False,
                            eog=False, exclude='bads')[:5]
        ica.fit(inst, picks=picks1)
        assert_false(any('EOG' in ch for ch in ica.ch_names))


@requires_sklearn
def test_max_pca_components_none():
    raw = read_raw_fif(raw_fname).crop(1.5, stop).load_data()
    events = read_events(event_name)
    picks = pick_types(raw.info, eeg=True, meg=False)
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)

    max_pca_components = None
    n_components = 10
    random_state = 12345

    tempdir = _TempDir()
    output_fname = op.join(tempdir, 'test_ica-ica.fif')

    ica = ICA(max_pca_components=max_pca_components,
              n_components=n_components, random_state=random_state)
    ica.fit(epochs)
    ica.save(output_fname)

    ica = read_ica(output_fname)

    # ICA.fit() replaced max_pca_components, which was previously None,
    # with the appropriate integer value.
    assert_equal(ica.max_pca_components, epochs.info['nchan'])
    assert_equal(ica.n_components, 10)


@requires_sklearn
def test_n_components_none():
    raw = read_raw_fif(raw_fname).crop(1.5, stop).load_data()
    events = read_events(event_name)
    picks = pick_types(raw.info, eeg=True, meg=False)
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)

    max_pca_components = 10
    n_components = None
    random_state = 12345

    tempdir = _TempDir()
    output_fname = op.join(tempdir, 'test_ica-ica.fif')

    ica = ICA(max_pca_components=max_pca_components,
              n_components=n_components, random_state=random_state)
    ica.fit(epochs)
    ica.save(output_fname)

    ica = read_ica(output_fname)

    # ICA.fit() replaced max_pca_components, which was previously None,
    # with the appropriate integer value.
    assert_equal(ica.max_pca_components, 10)
    assert_is_none(ica.n_components)


@requires_sklearn
def test_n_components_and_max_pca_components_none():
    raw = read_raw_fif(raw_fname).crop(1.5, stop).load_data()
    events = read_events(event_name)
    picks = pick_types(raw.info, eeg=True, meg=False)
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)

    max_pca_components = None
    n_components = None
    random_state = 12345

    tempdir = _TempDir()
    output_fname = op.join(tempdir, 'test_ica-ica.fif')

    ica = ICA(max_pca_components=max_pca_components,
              n_components=n_components, random_state=random_state)
    ica.fit(epochs)
    ica.save(output_fname)

    ica = read_ica(output_fname)

    # ICA.fit() replaced max_pca_components, which was previously None,
    # with the appropriate integer value.
    assert_equal(ica.max_pca_components, epochs.info['nchan'])
    assert_is_none(ica.n_components)


run_tests_if_main()
