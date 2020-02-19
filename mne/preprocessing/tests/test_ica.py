# Author: Denis Engemann <denis.engemann@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

from itertools import product
import os
import os.path as op
from unittest import SkipTest

import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose, assert_equal)
from scipy import stats, linalg
import matplotlib.pyplot as plt

from mne import (Epochs, read_events, pick_types, create_info, EpochsArray,
                 EvokedArray, Annotations, pick_channels_regexp)
from mne.cov import read_cov
from mne.preprocessing import (ICA, ica_find_ecg_events, ica_find_eog_events,
                               read_ica, run_ica)
from mne.preprocessing.ica import (get_score_funcs, corrmap, _sort_components,
                                   _ica_explained_variance, read_ica_eeglab)
from mne.io import read_raw_fif, Info, RawArray, read_raw_ctf, read_raw_eeglab
from mne.io.meas_info import _kind_dict
from mne.io.pick import _DATA_CH_TYPES_SPLIT
from mne.io.eeglab.eeglab import _check_load_mat
from mne.rank import _compute_rank_int
from mne.utils import (catch_logging, _TempDir, requires_sklearn,
                       run_tests_if_main)
from mne.datasets import testing
from mne.event import make_fixed_length_events

data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')
test_cov_name = op.join(data_dir, 'test-cov.fif')

test_base_dir = testing.data_path(download=False)
ctf_fname = op.join(test_base_dir, 'CTF', 'testdata_ctf.ds')

fif_fname = op.join(test_base_dir, 'MEG', 'sample',
                    'sample_audvis_trunc_raw.fif')
eeglab_fname = op.join(test_base_dir, 'EEGLAB', 'test_raw.set')
eeglab_montage = op.join(test_base_dir, 'EEGLAB', 'test_chans.locs')

ctf_fname2 = op.join(test_base_dir, 'CTF', 'catch-alp-good-f.ds')

event_id, tmin, tmax = 1, -0.2, 0.2
# if stop is too small pca may fail in some cases, but we're okay on this file
start, stop = 0, 6
score_funcs_unsuited = ['pointbiserialr', 'ansari']


def _skip_check_picard(method):
    if method == 'picard':
        try:
            import picard  # noqa, analysis:ignore
        except Exception as exp:
            raise SkipTest("Picard is not installed (%s)." % (exp,))


@requires_sklearn
@pytest.mark.parametrize("method", ["fastica", "picard"])
def test_ica_full_data_recovery(method):
    """Test recovery of full data when no source is rejected."""
    # Most basic recovery
    _skip_check_picard(method)
    raw = read_raw_fif(raw_fname).crop(0.5, stop).load_data()
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')[:10]
    with pytest.warns(RuntimeWarning, match='projection'):
        epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), preload=True)
    evoked = epochs.average()
    n_channels = 5
    data = raw._data[:n_channels].copy()
    data_epochs = epochs.get_data()
    data_evoked = evoked.data
    raw.set_annotations(Annotations([0.5], [0.5], ['BAD']))
    methods = [method]
    for method in methods:
        stuff = [(2, n_channels, True), (2, n_channels // 2, False)]
        for n_components, n_pca_components, ok in stuff:
            ica = ICA(n_components=n_components, random_state=0,
                      max_pca_components=n_pca_components,
                      n_pca_components=n_pca_components,
                      method=method, max_iter=1)
            with pytest.warns(UserWarning, match=None):  # sometimes warns
                ica.fit(raw, picks=list(range(n_channels)))
            _assert_ica_attributes(ica)
            raw2 = ica.apply(raw.copy(), exclude=[])
            if ok:
                assert_allclose(data[:n_channels], raw2._data[:n_channels],
                                rtol=1e-10, atol=1e-15)
            else:
                diff = np.abs(data[:n_channels] - raw2._data[:n_channels])
                assert (np.max(diff) > 1e-14)

            ica = ICA(n_components=n_components, method=method,
                      max_pca_components=n_pca_components,
                      n_pca_components=n_pca_components, random_state=0)
            with pytest.warns(None):  # sometimes warns
                ica.fit(epochs, picks=list(range(n_channels)))
            _assert_ica_attributes(ica)
            epochs2 = ica.apply(epochs.copy(), exclude=[])
            data2 = epochs2.get_data()[:, :n_channels]
            if ok:
                assert_allclose(data_epochs[:, :n_channels], data2,
                                rtol=1e-10, atol=1e-15)
            else:
                diff = np.abs(data_epochs[:, :n_channels] - data2)
                assert (np.max(diff) > 1e-14)

            evoked2 = ica.apply(evoked.copy(), exclude=[])
            data2 = evoked2.data[:n_channels]
            if ok:
                assert_allclose(data_evoked[:n_channels], data2,
                                rtol=1e-10, atol=1e-15)
            else:
                diff = np.abs(evoked.data[:n_channels] - data2)
                assert (np.max(diff) > 1e-14)
    pytest.raises(ValueError, ICA, method='pizza-decomposision')


@pytest.mark.parametrize("method", ["fastica", "picard"])
def test_ica_simple(method):
    """Test that ICA recovers the unmixing matrix in a simple case."""
    if method == "fastica":
        try:
            import sklearn  # noqa: F401
        except ImportError:
            raise SkipTest("scikit-learn not installed")
    _skip_check_picard(method)
    n_components = 3
    n_samples = 1000
    rng = np.random.RandomState(0)
    S = rng.laplace(size=(n_components, n_samples))
    A = rng.randn(n_components, n_components)
    data = np.dot(A, S)
    ica = ICA(n_components=n_components, method=method, random_state=0)
    ica._fit(data, n_components, 0)
    transform = np.dot(np.dot(ica.unmixing_matrix_, ica.pca_components_), A)
    amari_distance = np.mean(np.sum(np.abs(transform), axis=1) /
                             np.max(np.abs(transform), axis=1) - 1.)
    assert amari_distance < 0.1


@requires_sklearn
@pytest.mark.parametrize("method", ["infomax", "fastica", "picard"])
def test_ica_n_iter_(method):
    """Test that ICA.n_iter_ is set after fitting."""
    _skip_check_picard(method)

    raw = read_raw_fif(raw_fname).crop(0.5, stop).load_data()
    n_components = 3
    max_iter = 1
    ica = ICA(n_components=n_components, max_iter=max_iter, method=method)

    if method == 'infomax':
        ica.fit(raw)
    else:
        with pytest.warns(UserWarning, match='did not converge'):
            ica.fit(raw)

    assert_equal(ica.n_iter_, max_iter)

    # Test I/O roundtrip.
    tempdir = _TempDir()
    output_fname = op.join(tempdir, 'test_ica-ica.fif')
    _assert_ica_attributes(ica)
    ica.save(output_fname)
    ica = read_ica(output_fname)
    _assert_ica_attributes(ica)

    assert_equal(ica.n_iter_, max_iter)


@requires_sklearn
@pytest.mark.parametrize("method", ["fastica", "picard"])
def test_ica_rank_reduction(method):
    """Test recovery ICA rank reduction."""
    _skip_check_picard(method)
    # Most basic recovery
    raw = read_raw_fif(raw_fname).crop(0.5, stop).load_data()
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')[:10]
    n_components = 5
    max_pca_components = len(picks)
    for n_pca_components in [6, 10]:
        with pytest.warns(UserWarning, match='did not converge'):
            ica = ICA(n_components=n_components,
                      max_pca_components=max_pca_components,
                      n_pca_components=n_pca_components,
                      method=method, max_iter=1).fit(raw, picks=picks)

        rank_before = _compute_rank_int(raw.copy().pick(picks), proj=False)
        assert_equal(rank_before, len(picks))
        raw_clean = ica.apply(raw.copy())
        rank_after = _compute_rank_int(raw_clean.copy().pick(picks),
                                       proj=False)
        # interaction between ICA rejection and PCA components difficult
        # to preduct. Rank_after often seems to be 1 higher then
        # n_pca_components
        assert (n_components < n_pca_components <= rank_after <=
                rank_before)


@requires_sklearn
@pytest.mark.parametrize("method", ["fastica", "picard"])
def test_ica_reset(method):
    """Test ICA resetting."""
    _skip_check_picard(method)
    raw = read_raw_fif(raw_fname).crop(0.5, stop).load_data()
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')[:10]

    run_time_attrs = (
        'pre_whitener_',
        'unmixing_matrix_',
        'mixing_matrix_',
        'n_components_',
        'n_samples_',
        'pca_components_',
        'pca_explained_variance_',
        'pca_mean_',
        'n_iter_'
    )
    with pytest.warns(UserWarning, match='did not converge'):
        ica = ICA(
            n_components=3, max_pca_components=3, n_pca_components=3,
            method=method, max_iter=1).fit(raw, picks=picks)

    assert (all(hasattr(ica, attr) for attr in run_time_attrs))
    assert ica.labels_ is not None
    ica._reset()
    assert (not any(hasattr(ica, attr) for attr in run_time_attrs))
    assert ica.labels_ is not None


@requires_sklearn
@pytest.mark.parametrize("method", ["fastica", "picard"])
def test_ica_core(method):
    """Test ICA on raw and epochs."""
    _skip_check_picard(method)
    raw = read_raw_fif(raw_fname).crop(1.5, stop).load_data()

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
    methods = [method]
    iter_ica_params = product(noise_cov, n_components, max_pca_components,
                              picks_, methods)

    # # test init catchers
    pytest.raises(ValueError, ICA, n_components=3, max_pca_components=2)
    pytest.raises(ValueError, ICA, n_components=2.3, max_pca_components=2)

    # test essential core functionality
    for n_cov, n_comp, max_n, pcks, method in iter_ica_params:
        # Test ICA raw
        ica = ICA(noise_cov=n_cov, n_components=n_comp,
                  max_pca_components=max_n, n_pca_components=max_n,
                  random_state=0, method=method, max_iter=1)
        pytest.raises(ValueError, ica.__contains__, 'mag')

        print(ica)  # to test repr

        # test fit checker
        pytest.raises(RuntimeError, ica.get_sources, raw)
        pytest.raises(RuntimeError, ica.get_sources, epochs)

        # Test error upon empty epochs fitting
        with pytest.raises(RuntimeError, match='none were found'):
            ica.fit(epochs[0:0])

        # test decomposition
        with pytest.warns(UserWarning, match='did not converge'):
            ica.fit(raw, picks=pcks, start=start, stop=stop)
        repr(ica)  # to test repr
        assert ('mag' in ica)  # should now work without error

        # test re-fit
        unmixing1 = ica.unmixing_matrix_
        with pytest.warns(UserWarning, match='did not converge'):
            ica.fit(raw, picks=pcks, start=start, stop=stop)
        assert_array_almost_equal(unmixing1, ica.unmixing_matrix_)

        raw_sources = ica.get_sources(raw)
        # test for #3804
        assert_equal(raw_sources._filenames, [None])
        print(raw_sources)

        # test for gh-6271 (scaling of ICA traces)
        fig = raw_sources.plot()
        assert len(fig.axes[0].lines) in (4, 5, 6)
        for line in fig.axes[0].lines:
            y = line.get_ydata()
            if len(y) > 2:  # actual data, not markers
                assert np.ptp(y) < 15
        plt.close('all')

        sources = raw_sources[:, :][0]
        assert (sources.shape[0] == ica.n_components_)

        # test preload filter
        raw3 = raw.copy()
        raw3.preload = False
        pytest.raises(RuntimeError, ica.apply, raw3,
                      include=[1, 2])

        #######################################################################
        # test epochs decomposition
        ica = ICA(noise_cov=n_cov, n_components=n_comp,
                  max_pca_components=max_n, n_pca_components=max_n,
                  random_state=0, method=method)
        with pytest.warns(None):  # sometimes warns
            ica.fit(epochs, picks=picks)
        _assert_ica_attributes(ica)
        data = epochs.get_data()[:, 0, :]
        n_samples = np.prod(data.shape)
        assert_equal(ica.n_samples_, n_samples)
        print(ica)  # to test repr

        sources = ica.get_sources(epochs).get_data()
        assert (sources.shape[1] == ica.n_components_)

        pytest.raises(ValueError, ica.score_sources, epochs,
                      target=np.arange(1))

        # test preload filter
        epochs3 = epochs.copy()
        epochs3.preload = False
        pytest.raises(RuntimeError, ica.apply, epochs3,
                      include=[1, 2])

    # test for bug with whitener updating
    _pre_whitener = ica.pre_whitener_.copy()
    epochs._data[:, 0, 10:15] *= 1e12
    ica.apply(epochs.copy())
    assert_array_equal(_pre_whitener, ica.pre_whitener_)

    # test expl. var threshold leading to empty sel
    ica.n_components = 0.1
    pytest.raises(RuntimeError, ica.fit, epochs)

    offender = 1, 2, 3,
    pytest.raises(ValueError, ica.get_sources, offender)
    pytest.raises(TypeError, ica.fit, offender)
    pytest.raises(TypeError, ica.apply, offender)


@requires_sklearn
@pytest.mark.slowtest
@pytest.mark.parametrize("method", ["picard", "fastica"])
def test_ica_additional(method):
    """Test additional ICA functionality."""
    _skip_check_picard(method)

    tempdir = _TempDir()
    stop2 = 500
    raw = read_raw_fif(raw_fname).crop(1.5, stop).load_data()
    raw.del_proj()  # avoid warnings
    raw.set_annotations(Annotations([0.5], [0.5], ['BAD']))
    # XXX This breaks the tests :(
    # raw.info['bads'] = [raw.ch_names[1]]
    test_cov = read_cov(test_cov_name)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')[1::2]
    epochs = Epochs(raw, events, None, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True, proj=False)
    epochs.decimate(3, verbose='error')
    assert len(epochs) == 4

    # test if n_components=None works
    ica = ICA(n_components=None, max_pca_components=None,
              n_pca_components=None, random_state=0, method=method, max_iter=1)
    with pytest.warns(UserWarning, match='did not converge'):
        ica.fit(epochs)
    _assert_ica_attributes(ica)
    # for testing eog functionality
    picks2 = np.concatenate([picks, pick_types(raw.info, False, eog=True)])
    epochs_eog = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks2,
                        baseline=(None, 0), preload=True)
    del picks2

    test_cov2 = test_cov.copy()
    ica = ICA(noise_cov=test_cov2, n_components=3, max_pca_components=4,
              n_pca_components=4, method=method)
    assert (ica.info is None)
    with pytest.warns(RuntimeWarning, match='normalize_proj'):
        ica.fit(raw, picks[:5])
    _assert_ica_attributes(ica)
    assert (isinstance(ica.info, Info))
    assert (ica.n_components_ < 5)

    ica = ICA(n_components=3, max_pca_components=4, method=method,
              n_pca_components=4, random_state=0)
    pytest.raises(RuntimeError, ica.save, '')

    ica.fit(raw, picks=[1, 2, 3, 4, 5], start=start, stop=stop2)
    _assert_ica_attributes(ica)

    # check passing a ch_name to find_bads_ecg
    with pytest.warns(RuntimeWarning, match='longer'):
        _, scores_1 = ica.find_bads_ecg(raw)
        _, scores_2 = ica.find_bads_ecg(raw, raw.ch_names[1])
    assert scores_1[0] != scores_2[0]

    # test corrmap
    ica2 = ica.copy()
    ica3 = ica.copy()
    corrmap([ica, ica2], (0, 0), threshold='auto', label='blinks', plot=True,
            ch_type="mag")
    with pytest.raises(RuntimeError, match='No component detected'):
        corrmap([ica, ica2], (0, 0), threshold=2, plot=False, show=False,)
    corrmap([ica, ica2], (0, 0), threshold=0.5, plot=False, show=False)
    assert (ica.labels_["blinks"] == ica2.labels_["blinks"])
    assert (0 in ica.labels_["blinks"])
    # test retrieval of component maps as arrays
    components = ica.get_components()
    template = components[:, 0]
    EvokedArray(components, ica.info, tmin=0.).plot_topomap([0], time_unit='s')

    corrmap([ica, ica3], template, threshold='auto', label='blinks', plot=True,
            ch_type="mag")
    assert (ica2.labels_["blinks"] == ica3.labels_["blinks"])

    plt.close('all')

    # No match
    bad_ica = ica2.copy()
    bad_ica.mixing_matrix_[:] = 0.
    with pytest.warns(RuntimeWarning, match='divide'):
        with catch_logging() as log:
            corrmap([ica, bad_ica], (0, 0), threshold=0.5, plot=False,
                    show=False, verbose=True)
    log = log.getvalue()
    assert 'No maps selected' in log

    # make sure a single threshold in a list works
    corrmap([ica, ica3], template, threshold=[0.5], label='blinks', plot=True,
            ch_type="mag")

    ica_different_channels = ICA(n_components=2, random_state=0).fit(
        raw, picks=[2, 3, 4, 5])
    pytest.raises(ValueError, corrmap, [ica_different_channels, ica], (0, 0))

    # test warnings on bad filenames
    ica_badname = op.join(op.dirname(tempdir), 'test-bad-name.fif.gz')
    with pytest.warns(RuntimeWarning, match='-ica.fif'):
        ica.save(ica_badname)
    with pytest.warns(RuntimeWarning, match='-ica.fif'):
        read_ica(ica_badname)

    # test decim
    ica = ICA(n_components=3, max_pca_components=4,
              n_pca_components=4, method=method, max_iter=1)
    raw_ = raw.copy()
    for _ in range(3):
        raw_.append(raw_)
    n_samples = raw_._data.shape[1]
    with pytest.warns(UserWarning, match='did not converge'):
        ica.fit(raw, picks=picks[:5], decim=3)
    _assert_ica_attributes(ica)
    assert raw_._data.shape[1] == n_samples

    # test expl var
    ica = ICA(n_components=1.0, max_pca_components=4,
              n_pca_components=4, method=method, max_iter=1)
    with pytest.warns(UserWarning, match='did not converge'):
        ica.fit(raw, picks=None, decim=3)
    _assert_ica_attributes(ica)
    assert (ica.n_components_ == 4)
    ica_var = _ica_explained_variance(ica, raw, normalize=True)
    assert (np.all(ica_var[:-1] >= ica_var[1:]))

    # test ica sorting
    ica.exclude = [0]
    ica.labels_ = dict(blink=[0], think=[1])
    ica_sorted = _sort_components(ica, [3, 2, 1, 0], copy=True)
    assert_equal(ica_sorted.exclude, [3])
    assert_equal(ica_sorted.labels_, dict(blink=[3], think=[2]))

    # epochs extraction from raw fit
    pytest.raises(RuntimeError, ica.get_sources, epochs)
    # test reading and writing
    test_ica_fname = op.join(op.dirname(tempdir), 'test-ica.fif')
    for cov in (None, test_cov):
        ica = ICA(noise_cov=cov, n_components=2, max_pca_components=4,
                  n_pca_components=4, method=method, max_iter=1)
        with pytest.warns(None):  # ICA does not converge
            ica.fit(raw, picks=picks[:10], start=start, stop=stop2)
        _assert_ica_attributes(ica)
        sources = ica.get_sources(epochs).get_data()
        assert (ica.mixing_matrix_.shape == (2, 2))
        assert (ica.unmixing_matrix_.shape == (2, 2))
        assert (ica.pca_components_.shape == (4, 10))
        assert (sources.shape[1] == ica.n_components_)

        for exclude in [[], [0], np.array([1, 2, 3])]:
            ica.exclude = exclude
            ica.labels_ = {'foo': [0]}
            ica.save(test_ica_fname)
            ica_read = read_ica(test_ica_fname)
            assert (list(ica.exclude) == ica_read.exclude)
            assert_equal(ica.labels_, ica_read.labels_)
            ica.apply(raw)
            ica.exclude = []
            ica.apply(raw, exclude=[1])
            assert (ica.exclude == [])

            ica.exclude = [0, 1]
            ica.apply(raw, exclude=[1])
            assert (ica.exclude == [0, 1])

            ica_raw = ica.get_sources(raw)
            assert (ica.exclude == [ica_raw.ch_names.index(e) for e in
                                    ica_raw.info['bads']])

        # test filtering
        d1 = ica_raw._data[0].copy()
        ica_raw.filter(4, 20, fir_design='firwin2')
        assert_equal(ica_raw.info['lowpass'], 20.)
        assert_equal(ica_raw.info['highpass'], 4.)
        assert ((d1 != ica_raw._data[0]).any())
        d1 = ica_raw._data[0].copy()
        ica_raw.notch_filter([10], trans_bandwidth=10, fir_design='firwin')
        assert ((d1 != ica_raw._data[0]).any())

        ica.n_pca_components = 2
        ica.method = 'fake'
        ica.save(test_ica_fname)
        ica_read = read_ica(test_ica_fname)
        assert (ica.n_pca_components == ica_read.n_pca_components)
        assert_equal(ica.method, ica_read.method)
        assert_equal(ica.labels_, ica_read.labels_)

        # check type consistency
        attrs = ('mixing_matrix_ unmixing_matrix_ pca_components_ '
                 'pca_explained_variance_ pre_whitener_')

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
                     'pre_whitener_']:
            assert_array_almost_equal(getattr(ica, attr),
                                      getattr(ica_read, attr))

        assert (ica.ch_names == ica_read.ch_names)
        assert (isinstance(ica_read.info, Info))

        sources = ica.get_sources(raw)[:, :][0]
        sources2 = ica_read.get_sources(raw)[:, :][0]
        assert_array_almost_equal(sources, sources2)

        _raw1 = ica.apply(raw, exclude=[1])
        _raw2 = ica_read.apply(raw, exclude=[1])
        assert_array_almost_equal(_raw1[:, :][0], _raw2[:, :][0])

    os.remove(test_ica_fname)
    # check score funcs
    for name, func in get_score_funcs().items():
        if name in score_funcs_unsuited:
            continue
        scores = ica.score_sources(raw, target='EOG 061', score_func=func,
                                   start=0, stop=10)
        assert (ica.n_components_ == len(scores))

    # check univariate stats
    scores = ica.score_sources(raw, start=0, stop=50, score_func=stats.skew)
    # check exception handling
    pytest.raises(ValueError, ica.score_sources, raw,
                  target=np.arange(1))

    params = []
    params += [(None, -1, slice(2), [0, 1])]  # variance, kurtosis params
    params += [(None, 'MEG 1531')]  # ECG / EOG channel params
    for idx, ch_name in product(*params):
        ica.detect_artifacts(raw, start_find=0, stop_find=50, ecg_ch=ch_name,
                             eog_ch=ch_name, skew_criterion=idx,
                             var_criterion=idx, kurt_criterion=idx)

    # Make sure detect_artifacts marks the right components.
    # For int criterion, the doc says "E.g. range(2) would return the two
    # sources with the highest score". Assert that's what it does.
    # Only test for skew, since it's always the same code.
    ica.exclude = []
    ica.detect_artifacts(raw, start_find=0, stop_find=50, ecg_ch=None,
                         eog_ch=None, skew_criterion=0,
                         var_criterion=None, kurt_criterion=None)
    assert np.abs(scores[ica.exclude]) == np.max(np.abs(scores))

    evoked = epochs.average()
    evoked_data = evoked.data.copy()
    raw_data = raw[:][0].copy()
    epochs_data = epochs.get_data().copy()

    with pytest.warns(RuntimeWarning, match='longer'):
        idx, scores = ica.find_bads_ecg(raw, method='ctps')
    assert_equal(len(scores), ica.n_components_)
    with pytest.warns(RuntimeWarning, match='longer'):
        idx, scores = ica.find_bads_ecg(raw, method='correlation')
    assert_equal(len(scores), ica.n_components_)

    with pytest.warns(RuntimeWarning, match='longer'):
        idx, scores = ica.find_bads_eog(raw)
    assert_equal(len(scores), ica.n_components_)

    idx, scores = ica.find_bads_ecg(epochs, method='ctps')

    assert_equal(len(scores), ica.n_components_)
    pytest.raises(ValueError, ica.find_bads_ecg, epochs.average(),
                  method='ctps')
    pytest.raises(ValueError, ica.find_bads_ecg, raw,
                  method='crazy-coupling')

    with pytest.warns(RuntimeWarning, match='longer'):
        idx, scores = ica.find_bads_eog(raw)
    assert_equal(len(scores), ica.n_components_)

    raw.info['chs'][raw.ch_names.index('EOG 061') - 1]['kind'] = 202
    with pytest.warns(RuntimeWarning, match='longer'):
        idx, scores = ica.find_bads_eog(raw)
    assert (isinstance(scores, list))
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
        assert (ica.n_components_ == len(scores))

    # check univariate stats
    scores = ica.score_sources(epochs, score_func=stats.skew)

    # check exception handling
    pytest.raises(ValueError, ica.score_sources, epochs,
                  target=np.arange(1))

    # ecg functionality
    ecg_scores = ica.score_sources(raw, target='MEG 1531',
                                   score_func='pearsonr')

    with pytest.warns(RuntimeWarning, match='longer'):
        ecg_events = ica_find_ecg_events(
            raw, sources[np.abs(ecg_scores).argmax()])
    assert (ecg_events.ndim == 2)

    # eog functionality
    eog_scores = ica.score_sources(raw, target='EOG 061',
                                   score_func='pearsonr')
    with pytest.warns(RuntimeWarning, match='longer'):
        eog_events = ica_find_eog_events(
            raw, sources[np.abs(eog_scores).argmax()])
    assert (eog_events.ndim == 2)

    # Test ica fiff export
    ica_raw = ica.get_sources(raw, start=0, stop=100)
    assert (ica_raw.last_samp - ica_raw.first_samp == 100)
    assert_equal(len(ica_raw._filenames), 1)  # API consistency
    ica_chans = [ch for ch in ica_raw.ch_names if 'ICA' in ch]
    assert (ica.n_components_ == len(ica_chans))
    test_ica_fname = op.join(op.abspath(op.curdir), 'test-ica_raw.fif')
    ica.n_components = np.int32(ica.n_components)
    ica_raw.save(test_ica_fname, overwrite=True)
    ica_raw2 = read_raw_fif(test_ica_fname, preload=True)
    assert_allclose(ica_raw._data, ica_raw2._data, rtol=1e-5, atol=1e-4)
    ica_raw2.close()
    os.remove(test_ica_fname)

    # Test ica epochs export
    ica_epochs = ica.get_sources(epochs)
    assert (ica_epochs.events.shape == epochs.events.shape)
    ica_chans = [ch for ch in ica_epochs.ch_names if 'ICA' in ch]
    assert (ica.n_components_ == len(ica_chans))
    assert (ica.n_components_ == ica_epochs.get_data().shape[1])
    assert (ica_epochs._raw is None)
    assert (ica_epochs.preload is True)

    # test float n pca components
    ica.pca_explained_variance_ = np.array([0.2] * 5)
    ica.n_components_ = 0
    for ncomps, expected in [[0.3, 1], [0.9, 4], [1, 1]]:
        ncomps_ = ica._check_n_pca_components(ncomps)
        assert (ncomps_ == expected)

    ica = ICA(method=method)
    with pytest.warns(None):  # sometimes does not converge
        ica.fit(raw, picks=picks[:5])
    _assert_ica_attributes(ica)
    with pytest.warns(RuntimeWarning, match='longer'):
        ica.find_bads_ecg(raw)
    ica.find_bads_eog(epochs, ch_name='MEG 0121')
    assert_array_equal(raw_data, raw[:][0])

    raw.drop_channels(['MEG 0122'])
    pytest.raises(RuntimeError, ica.find_bads_eog, raw)
    with pytest.warns(RuntimeWarning, match='longer'):
        pytest.raises(RuntimeError, ica.find_bads_ecg, raw)


@requires_sklearn
@pytest.mark.parametrize("method", ("fastica", "picard", "infomax"))
@pytest.mark.parametrize("idx", (None, -1, slice(2), [0, 1]))
@pytest.mark.parametrize("ch_name", (None, 'MEG 1531'))
def test_detect_artifacts_replacement_of_run_ica(method, idx, ch_name):
    """Test replacement workflow for deprecated run_ica() function."""
    _skip_check_picard(method)
    raw = read_raw_fif(raw_fname).crop(1.5, stop).load_data()
    ica = ICA(n_components=2, method=method)
    ica.fit(raw)
    ica.detect_artifacts(raw, start_find=0, stop_find=5, ecg_ch=ch_name,
                         eog_ch=ch_name, skew_criterion=idx,
                         var_criterion=idx, kurt_criterion=idx)


@requires_sklearn
def test_run_ica_deprecation():
    """Test that run_ica() has been deprecated."""
    raw = read_raw_fif(raw_fname).crop(1.5, stop).load_data()
    method = 'fastica'
    idx = None
    ch_name = None
    with pytest.warns(DeprecationWarning, match='run_ica() is deprecated'):
        run_ica(raw, n_components=2, start=0, stop=0.5, start_find=0,
                stop_find=5, ecg_ch=ch_name, eog_ch=ch_name, method=method,
                skew_criterion=idx, var_criterion=idx, kurt_criterion=idx)


@requires_sklearn
@pytest.mark.parametrize("method", ["fastica", "picard"])
def test_ica_reject_buffer(method):
    """Test ICA data raw buffer rejection."""
    _skip_check_picard(method)
    raw = read_raw_fif(raw_fname).crop(1.5, stop).load_data()
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    raw._data[2, 1000:1005] = 5e-12
    ica = ICA(n_components=3, max_pca_components=4, n_pca_components=4,
              method=method)
    with catch_logging() as drop_log:
        ica.fit(raw, picks[:5], reject=dict(mag=2.5e-12), decim=2,
                tstep=0.01, verbose=True, reject_by_annotation=False)
        assert (raw._data[:5, ::2].shape[1] - 4 == ica.n_samples_)
    log = [l for l in drop_log.getvalue().split('\n') if 'detected' in l]
    assert_equal(len(log), 1)
    _assert_ica_attributes(ica)


@requires_sklearn
@pytest.mark.parametrize("method", ["fastica", "picard"])
def test_ica_twice(method):
    """Test running ICA twice."""
    _skip_check_picard(method)
    raw = read_raw_fif(raw_fname).crop(1.5, stop).load_data()
    picks = pick_types(raw.info, meg='grad', exclude='bads')
    n_components = 0.9
    max_pca_components = None
    n_pca_components = 1.1
    ica1 = ICA(n_components=n_components, method=method,
               max_pca_components=max_pca_components,
               n_pca_components=n_pca_components, random_state=0)

    ica1.fit(raw, picks=picks, decim=3)
    raw_new = ica1.apply(raw, n_pca_components=n_pca_components)
    ica2 = ICA(n_components=n_components, method=method,
               max_pca_components=max_pca_components,
               n_pca_components=1.0, random_state=0)
    ica2.fit(raw_new, picks=picks, decim=3)
    assert_equal(ica1.n_components_, ica2.n_components_)


@requires_sklearn
@pytest.mark.parametrize("method", ["fastica", "picard", "infomax"])
def test_fit_params(method, tmpdir):
    """Test fit_params for ICA."""
    _skip_check_picard(method)
    fit_params = {}
    ICA(fit_params=fit_params, method=method)  # test no side effects
    assert fit_params == {}

    # Test I/O roundtrip.
    # Only picard and infomax support the "extended" keyword, so limit the
    # tests to those.
    if method in ['picard', 'infomax']:
        tmpdir = str(tmpdir)
        output_fname = op.join(tmpdir, 'test_ica-ica.fif')

        raw = read_raw_fif(raw_fname).crop(0.5, stop).load_data()
        n_components = 3
        max_iter = 1
        fit_params = dict(extended=True)
        ica = ICA(fit_params=fit_params, n_components=n_components,
                  max_iter=max_iter, method=method)
        fit_params_after_instantiation = ica.fit_params

        if method == 'infomax':
            ica.fit(raw)
        else:
            with pytest.warns(UserWarning, match='did not converge'):
                ica.fit(raw)

        ica.save(output_fname)
        ica = read_ica(output_fname)

        assert ica.fit_params == fit_params_after_instantiation


@requires_sklearn
@pytest.mark.parametrize("method", ["fastica", "picard"])
@pytest.mark.parametrize("allow_ref_meg", [True, False])
def test_bad_channels(method, allow_ref_meg):
    """Test exception when unsupported channels are used."""
    _skip_check_picard(method)
    chs = [i for i in _kind_dict]
    info = create_info(len(chs), 500, chs)
    rng = np.random.RandomState(0)
    data = rng.rand(len(chs), 50)
    raw = RawArray(data, info)
    data = rng.rand(100, len(chs), 50)
    epochs = EpochsArray(data, info)

    n_components = 0.9
    data_chs = list(_DATA_CH_TYPES_SPLIT + ('eog',))
    if allow_ref_meg:
        data_chs.append('ref_meg')
    chs_bad = list(set(chs) - set(data_chs))
    ica = ICA(n_components=n_components, method=method,
              allow_ref_meg=allow_ref_meg)
    for inst in [raw, epochs]:
        for ch in chs_bad:
            if allow_ref_meg:
                # Test case for only bad channels
                picks_bad1 = pick_types(inst.info, meg=False,
                                        ref_meg=False,
                                        **{str(ch): True})
                # Test case for good and bad channels
                picks_bad2 = pick_types(inst.info, meg=True,
                                        ref_meg=True,
                                        **{str(ch): True})
            else:
                # Test case for only bad channels
                picks_bad1 = pick_types(inst.info, meg=False,
                                        **{str(ch): True})
                # Test case for good and bad channels
                picks_bad2 = pick_types(inst.info, meg=True,
                                        **{str(ch): True})

            pytest.raises(ValueError, ica.fit, inst, picks=picks_bad1)
            pytest.raises(ValueError, ica.fit, inst, picks=picks_bad2)
        pytest.raises(ValueError, ica.fit, inst, picks=[])


@requires_sklearn
@pytest.mark.parametrize("method", ["fastica", "picard"])
def test_eog_channel(method):
    """Test that EOG channel is included when performing ICA."""
    _skip_check_picard(method)
    raw = read_raw_fif(raw_fname, preload=True)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=True, ecg=False,
                       eog=True, exclude='bads')
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    n_components = 0.9
    ica = ICA(n_components=n_components, method=method)
    # Test case for MEG and EOG data. Should have EOG channel
    for inst in [raw, epochs]:
        picks1a = pick_types(inst.info, meg=True, stim=False, ecg=False,
                             eog=False, exclude='bads')[:4]
        picks1b = pick_types(inst.info, meg=False, stim=False, ecg=False,
                             eog=True, exclude='bads')
        picks1 = np.append(picks1a, picks1b)
        ica.fit(inst, picks=picks1)
        assert (any('EOG' in ch for ch in ica.ch_names))
        _assert_ica_attributes(ica)
    # Test case for MEG data. Should have no EOG channel
    for inst in [raw, epochs]:
        picks1 = pick_types(inst.info, meg=True, stim=False, ecg=False,
                            eog=False, exclude='bads')[:5]
        ica.fit(inst, picks=picks1)
        _assert_ica_attributes(ica)
        assert not any('EOG' in ch for ch in ica.ch_names)


@requires_sklearn
@pytest.mark.parametrize("method", ["fastica", "picard"])
def test_max_pca_components_none(method):
    """Test max_pca_components=None."""
    _skip_check_picard(method)
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
    ica = ICA(max_pca_components=max_pca_components, method=method,
              n_components=n_components, random_state=random_state)
    with pytest.warns(None):
        ica.fit(epochs)
    _assert_ica_attributes(ica)
    ica.save(output_fname)

    ica = read_ica(output_fname)

    # ICA.fit() replaced max_pca_components, which was previously None,
    # with the appropriate integer value.
    assert_equal(ica.max_pca_components, epochs.info['nchan'])
    assert_equal(ica.n_components, 10)


@requires_sklearn
@pytest.mark.parametrize("method", ["fastica", "picard"])
def test_n_components_none(method):
    """Test n_components=None."""
    _skip_check_picard(method)
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
    ica = ICA(max_pca_components=max_pca_components, method=method,
              n_components=n_components, random_state=random_state)
    with pytest.warns(None):
        ica.fit(epochs)
    _assert_ica_attributes(ica)
    ica.save(output_fname)

    ica = read_ica(output_fname)
    _assert_ica_attributes(ica)

    # ICA.fit() replaced max_pca_components, which was previously None,
    # with the appropriate integer value.
    assert_equal(ica.max_pca_components, 10)
    assert ica.n_components is None


@requires_sklearn
@pytest.mark.parametrize("method", ["fastica", "picard"])
def test_n_components_and_max_pca_components_none(method):
    """Test n_components and max_pca_components=None."""
    _skip_check_picard(method)
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
    ica = ICA(max_pca_components=max_pca_components, method=method,
              n_components=n_components, random_state=random_state)
    with pytest.warns(None):  # convergence
        ica.fit(epochs)
    ica.save(output_fname)
    _assert_ica_attributes(ica)

    ica = read_ica(output_fname)
    _assert_ica_attributes(ica)

    # ICA.fit() replaced max_pca_components, which was previously None,
    # with the appropriate integer value.
    assert_equal(ica.max_pca_components, epochs.info['nchan'])
    assert ica.n_components is None


@requires_sklearn
@testing.requires_testing_data
def test_ica_ctf():
    """Test run ICA computation on ctf data with/without compensation."""
    method = 'fastica'
    raw = read_raw_ctf(ctf_fname, preload=True)
    events = make_fixed_length_events(raw, 99999)
    for comp in [0, 1]:
        raw.apply_gradient_compensation(comp)
        epochs = Epochs(raw, events, None, -0.2, 0.2, preload=True)
        evoked = epochs.average()

        # test fit
        for inst in [raw, epochs]:
            ica = ICA(n_components=2, random_state=0, max_iter=2,
                      method=method)
            with pytest.warns(UserWarning, match='did not converge'):
                ica.fit(inst)
            _assert_ica_attributes(ica)

        # test apply and get_sources
        for inst in [raw, epochs, evoked]:
            ica.apply(inst)
            ica.get_sources(inst)

    # test mixed compensation case
    raw.apply_gradient_compensation(0)
    ica = ICA(n_components=2, random_state=0, max_iter=2, method=method)
    with pytest.warns(UserWarning, match='did not converge'):
        ica.fit(raw)
    _assert_ica_attributes(ica)
    raw.apply_gradient_compensation(1)
    epochs = Epochs(raw, events, None, -0.2, 0.2, preload=True)
    evoked = epochs.average()
    for inst in [raw, epochs, evoked]:
        with pytest.raises(RuntimeError, match='Compensation grade of ICA'):
            ica.apply(inst)
        with pytest.raises(RuntimeError, match='Compensation grade of ICA'):
            ica.get_sources(inst)


@requires_sklearn
@testing.requires_testing_data
def test_ica_labels():
    """Test ICA labels."""
    # The CTF data are uniquely well suited to testing the ICA.find_bads_
    # methods
    raw = read_raw_ctf(ctf_fname, preload=True)
    # derive reference ICA components and append them to raw
    icarf = ICA(n_components=2, random_state=0, max_iter=2, allow_ref_meg=True)
    with pytest.warns(UserWarning, match='did not converge'):
        icarf.fit(raw.copy().pick_types(meg=False, ref_meg=True))
    icacomps = icarf.get_sources(raw)
    # rename components so they are auto-detected by find_bads_ref
    icacomps.rename_channels({c: 'REF_' + c for c in icacomps.ch_names})
    # and add them to raw
    raw.add_channels([icacomps])
    # set the appropriate EEG channels to EOG and ECG
    raw.set_channel_types({'EEG057': 'eog', 'EEG058': 'eog', 'EEG059': 'ecg'})
    ica = ICA(n_components=4, random_state=0, max_iter=2, method='fastica')
    with pytest.warns(UserWarning, match='did not converge'):
        ica.fit(raw)
    _assert_ica_attributes(ica)

    ica.find_bads_eog(raw, l_freq=None, h_freq=None)
    picks = list(pick_types(raw.info, meg=False, eog=True))
    for idx, ch in enumerate(picks):
        assert '{}/{}/{}'.format('eog', idx, raw.ch_names[ch]) in ica.labels_
    assert 'eog' in ica.labels_
    for key in ('ecg', 'ref_meg', 'ecg/ECG-MAG'):
        assert key not in ica.labels_

    ica.find_bads_ecg(raw, l_freq=None, h_freq=None, method='correlation')
    picks = list(pick_types(raw.info, meg=False, ecg=True))
    for idx, ch in enumerate(picks):
        assert '{}/{}/{}'.format('ecg', idx, raw.ch_names[ch]) in ica.labels_
    for key in ('ecg', 'eog'):
        assert key in ica.labels_
    for key in ('ref_meg', 'ecg/ECG-MAG'):
        assert key not in ica.labels_

    ica.find_bads_ref(raw, l_freq=None, h_freq=None)
    picks = pick_channels_regexp(raw.ch_names, 'REF_ICA*')
    for idx, ch in enumerate(picks):
        assert '{}/{}/{}'.format('ref_meg', idx,
                                 raw.ch_names[ch]) in ica.labels_
    for key in ('ecg', 'eog', 'ref_meg'):
        assert key in ica.labels_
    assert 'ecg/ECG-MAG' not in ica.labels_

    ica.find_bads_ecg(raw, l_freq=None, h_freq=None)
    for key in ('ecg', 'eog', 'ref_meg', 'ecg/ECG-MAG'):
        assert key in ica.labels_


@requires_sklearn
@testing.requires_testing_data
def test_ica_eeg():
    """Test ICA on EEG."""
    method = 'fastica'
    raw_fif = read_raw_fif(fif_fname, preload=True)
    raw_eeglab = read_raw_eeglab(input_fname=eeglab_fname,
                                 preload=True)
    for raw in [raw_fif, raw_eeglab]:
        events = make_fixed_length_events(raw, 99999, start=0, stop=0.3,
                                          duration=0.1)
        picks_meg = pick_types(raw.info, meg=True, eeg=False)[:2]
        picks_eeg = pick_types(raw.info, meg=False, eeg=True)[:2]
        picks_all = []
        picks_all.extend(picks_meg)
        picks_all.extend(picks_eeg)
        epochs = Epochs(raw, events, None, -0.1, 0.1, preload=True)
        evoked = epochs.average()

        for picks in [picks_meg, picks_eeg, picks_all]:
            if len(picks) == 0:
                continue
            # test fit
            for inst in [raw, epochs]:
                ica = ICA(n_components=2, random_state=0, max_iter=2,
                          method=method)
                with pytest.warns(None):
                    ica.fit(inst, picks=picks)
                _assert_ica_attributes(ica)

            # test apply and get_sources
            for inst in [raw, epochs, evoked]:
                ica.apply(inst)
                ica.get_sources(inst)

    with pytest.warns(RuntimeWarning, match='MISC channel'):
        raw = read_raw_ctf(ctf_fname2, preload=True)
    events = make_fixed_length_events(raw, 99999, start=0, stop=0.2,
                                      duration=0.1)
    picks_meg = pick_types(raw.info, meg=True, eeg=False)[:2]
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)[:2]
    picks_all = picks_meg + picks_eeg
    for comp in [0, 1]:
        raw.apply_gradient_compensation(comp)
        epochs = Epochs(raw, events, None, -0.1, 0.1, preload=True)
        evoked = epochs.average()

        for picks in [picks_meg, picks_eeg, picks_all]:
            if len(picks) == 0:
                continue
            # test fit
            for inst in [raw, epochs]:
                ica = ICA(n_components=2, random_state=0, max_iter=2,
                          method=method)
                with pytest.warns(None):
                    ica.fit(inst)
                _assert_ica_attributes(ica)

            # test apply and get_sources
            for inst in [raw, epochs, evoked]:
                ica.apply(inst)
                ica.get_sources(inst)


@testing.requires_testing_data
def test_read_ica_eeglab():
    """Test read_ica_eeglab function."""
    fname = op.join(test_base_dir, "EEGLAB", "test_raw.set")
    fname_cleaned_matlab = op.join(test_base_dir, "EEGLAB",
                                   "test_raw.cleaned.set")

    raw = read_raw_eeglab(fname, preload=True)
    raw_eeg = _check_load_mat(fname, None)
    raw_cleaned_matlab = read_raw_eeglab(fname_cleaned_matlab,
                                         preload=True)

    mark_to_remove = ["manual"]
    comp_info = raw_eeg.marks["comp_info"]

    if len(comp_info["flags"].shape) > 1:
        ind_comp_to_drop = [np.where(flags)[0]
                            for flags, label in zip(comp_info["flags"],
                                                    comp_info["label"])
                            if label in mark_to_remove]
        ind_comp_to_drop = np.unique(np.concatenate(ind_comp_to_drop))
    else:
        ind_comp_to_drop = np.where(comp_info["flags"])[0]

    ica = read_ica_eeglab(fname)
    _assert_ica_attributes(ica)
    raw_cleaned = ica.apply(raw.copy(), exclude=ind_comp_to_drop)

    assert_allclose(raw_cleaned_matlab.get_data(), raw_cleaned.get_data(),
                    rtol=1e-05, atol=1e-08)


def _assert_ica_attributes(ica):
    """Assert some attributes of ICA objects."""
    __tracebackhide__ = True
    # This tests properties, but also serves as documentation of
    # the shapes these arrays can obtain and how they obtain them

    # Pre-whitener
    n_ch = len(ica.ch_names)
    assert ica.pre_whitener_.shape == (
        n_ch, n_ch if ica.noise_cov is not None else 1)

    # PCA
    n_pca = ica.max_pca_components
    assert ica.pca_components_.shape == (n_pca, n_ch), 'PCA shape'
    assert_allclose(np.dot(ica.pca_components_, ica.pca_components_.T),
                    np.eye(n_pca), atol=1e-6, err_msg='PCA orthogonality')
    assert ica.pca_mean_.shape == (n_ch,)

    # Mixing/unmixing
    assert ica.unmixing_matrix_.shape == (ica.n_components_,) * 2, \
        'Unmixing shape'
    assert ica.mixing_matrix_.shape == (ica.n_components_,) * 2, \
        'Mixing shape'
    mix_unmix = np.dot(ica.mixing_matrix_, ica.unmixing_matrix_)
    s = linalg.svdvals(ica.unmixing_matrix_)
    nz = len(s) - (s > s[0] * 1e-12).sum()
    want = np.eye(ica.n_components_)
    want[:nz] = 0
    assert_allclose(mix_unmix, want, atol=1e-6, err_msg='Mixing as pinv')


run_tests_if_main()
