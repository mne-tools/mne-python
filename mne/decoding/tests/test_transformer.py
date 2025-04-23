# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)

pytest.importorskip("sklearn")

from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from mne import Epochs, EpochsArray, create_info, io, pick_types, read_events
from mne.decoding import (
    FilterEstimator,
    LinearModel,
    PSDEstimator,
    Scaler,
    TemporalFilter,
    UnsupervisedSpatialFilter,
    Vectorizer,
)
from mne.defaults import DEFAULTS
from mne.utils import use_log_level

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
start, stop = 0, 8
data_dir = Path(__file__).parents[2] / "io" / "tests" / "data"
raw_fname = data_dir / "test_raw.fif"
event_name = data_dir / "test-eve.fif"
info = create_info(2, 1000.0, "eeg")


@pytest.mark.parametrize(
    "info, method",
    [
        (True, None),
        (True, dict(mag=5, grad=10, eeg=20)),
        (False, "mean"),
        (False, "median"),
    ],
)
def test_scaler(info, method):
    """Test methods of Scaler."""
    raw = io.read_raw_fif(raw_fname)
    events = read_events(event_name)
    picks = pick_types(
        raw.info, meg=True, stim=False, ecg=False, eog=False, exclude="bads"
    )
    picks = picks[1:13:3]

    epochs = Epochs(
        raw, events, event_id, tmin, tmax, picks=picks, baseline=(None, 0), preload=True
    )
    epochs_data = epochs.get_data(copy=False)
    y = epochs.events[:, -1]

    epochs_data_t = epochs_data.transpose([1, 0, 2])

    if info:
        info = epochs.info
    scaler = Scaler(info, method)
    X = scaler.fit_transform(epochs_data, y)
    assert_equal(X.shape, epochs_data.shape)
    if method is None or isinstance(method, dict):
        sd = DEFAULTS["scalings"] if method is None else method
        stds = np.zeros(len(picks))
        for key in ("mag", "grad"):
            stds[pick_types(epochs.info, meg=key)] = 1.0 / sd[key]
        stds[pick_types(epochs.info, meg=False, eeg=True)] = 1.0 / sd["eeg"]
        means = np.zeros(len(epochs.ch_names))
    elif method == "mean":
        stds = np.array([np.std(ch_data) for ch_data in epochs_data_t])
        means = np.array([np.mean(ch_data) for ch_data in epochs_data_t])
    else:  # median
        percs = np.array(
            [np.percentile(ch_data, [25, 50, 75]) for ch_data in epochs_data_t]
        )
        stds = percs[:, 2] - percs[:, 0]
        means = percs[:, 1]
    assert_allclose(
        X * stds[:, np.newaxis] + means[:, np.newaxis],
        epochs_data,
        rtol=1e-12,
        atol=1e-20,
        err_msg=method,
    )

    X2 = scaler.fit(epochs_data, y).transform(epochs_data)
    assert_array_equal(X, X2)

    # inverse_transform
    Xi = scaler.inverse_transform(X)
    assert_array_almost_equal(epochs_data, Xi)

    # Test init exception
    x = Scaler(None, None)
    with pytest.raises(ValueError):
        x.fit(epochs_data, y)
    pytest.raises(ValueError, scaler.fit, "foo", y)
    pytest.raises(ValueError, scaler.transform, "foo")
    epochs_bad = Epochs(
        raw,
        events,
        event_id,
        0,
        0.01,
        baseline=None,
        picks=np.arange(len(raw.ch_names)),
    )  # non-data chs
    scaler = Scaler(epochs_bad.info, None)
    pytest.raises(ValueError, scaler.fit, epochs_bad.get_data(copy=False), y)


def test_filterestimator():
    """Test methods of FilterEstimator."""
    raw = io.read_raw_fif(raw_fname)
    events = read_events(event_name)
    picks = pick_types(
        raw.info, meg=True, stim=False, ecg=False, eog=False, exclude="bads"
    )
    picks = picks[1:13:3]
    epochs = Epochs(
        raw, events, event_id, tmin, tmax, picks=picks, baseline=(None, 0), preload=True
    )
    epochs_data = epochs.get_data(copy=False)

    # Add tests for different combinations of l_freq and h_freq
    filt = FilterEstimator(epochs.info, l_freq=40, h_freq=80)
    y = epochs.events[:, -1]
    X = filt.fit_transform(epochs_data, y)
    assert X.shape == epochs_data.shape
    assert_array_equal(filt.fit(epochs_data, y).transform(epochs_data), X)

    filt = FilterEstimator(
        epochs.info,
        l_freq=None,
        h_freq=40,
        filter_length="auto",
        l_trans_bandwidth="auto",
        h_trans_bandwidth="auto",
    )
    y = epochs.events[:, -1]
    X = filt.fit_transform(epochs_data, y)

    filt = FilterEstimator(epochs.info, l_freq=1, h_freq=1)
    y = epochs.events[:, -1]
    with pytest.warns(RuntimeWarning, match="longer than the signal"):
        pytest.raises(ValueError, filt.fit_transform, epochs_data, y)

    filt = FilterEstimator(
        epochs.info,
        l_freq=40,
        h_freq=None,
        filter_length="auto",
        l_trans_bandwidth="auto",
        h_trans_bandwidth="auto",
    )
    X = filt.fit_transform(epochs_data, y)

    # Test init exception
    pytest.raises(ValueError, filt.fit, "foo", y)
    pytest.raises(ValueError, filt.transform, "foo")


def test_psdestimator():
    """Test methods of PSDEstimator."""
    raw = io.read_raw_fif(raw_fname)
    events = read_events(event_name)
    picks = pick_types(
        raw.info, meg=True, stim=False, ecg=False, eog=False, exclude="bads"
    )
    picks = picks[1:13:3]
    epochs = Epochs(
        raw, events, event_id, tmin, tmax, picks=picks, baseline=(None, 0), preload=True
    )
    epochs_data = epochs.get_data(copy=False)
    psd = PSDEstimator(2 * np.pi, 0, np.inf)
    y = epochs.events[:, -1]
    assert not hasattr(psd, "fitted_")
    X = psd.fit_transform(epochs_data, y)
    assert psd.fitted_

    assert X.shape[0] == epochs_data.shape[0]
    assert_array_equal(psd.fit(epochs_data, y).transform(epochs_data), X)

    # Test init exception
    with pytest.raises(ValueError):
        psd.fit("foo", y)
    with pytest.raises(ValueError):
        psd.transform("foo")


def test_vectorizer():
    """Test Vectorizer."""
    data = np.random.rand(150, 18, 6)
    vect = Vectorizer()
    result = vect.fit_transform(data)
    assert_equal(result.ndim, 2)

    # check inverse_trasnform
    orig_data = vect.inverse_transform(result)
    assert_equal(orig_data.ndim, 3)
    assert_array_equal(orig_data, data)
    assert_array_equal(vect.inverse_transform(result[1:]), data[1:])

    # check with different shape
    assert_equal(vect.fit_transform(np.random.rand(150, 18, 6, 3)).shape, (150, 324))
    assert_equal(vect.fit_transform(data[1:]).shape, (149, 108))

    # check if raised errors are working correctly
    X = np.random.default_rng(0).standard_normal((105, 12, 3))
    y = np.arange(X.shape[0]) % 2
    pytest.raises(ValueError, vect.transform, X[..., np.newaxis])
    pytest.raises(ValueError, vect.inverse_transform, X[:, :-1])

    # And that pipelines work properly
    X_arr = EpochsArray(X, create_info(12, 1000.0, "eeg"))
    vect.fit(X_arr)
    clf = make_pipeline(Vectorizer(), StandardScaler(), LinearModel())
    clf.fit(X_arr, y)


def test_unsupervised_spatial_filter():
    """Test unsupervised spatial filter."""
    raw = io.read_raw_fif(raw_fname)
    events = read_events(event_name)
    picks = pick_types(
        raw.info, meg=True, stim=False, ecg=False, eog=False, exclude="bads"
    )
    picks = picks[1:13:3]
    epochs = Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        picks=picks,
        preload=True,
        baseline=None,
        verbose=False,
    )

    # Test estimator (must be a transformer)
    X = epochs.get_data(copy=False)
    usf = UnsupervisedSpatialFilter(KernelRidge(2))
    with pytest.raises(ValueError, match="transform"):
        usf.fit(X)

    # Test fit
    n_components = 4
    usf = UnsupervisedSpatialFilter(PCA(n_components))
    usf.fit(X)
    usf1 = UnsupervisedSpatialFilter(PCA(n_components))

    # test transform
    assert_equal(usf.transform(X).ndim, 3)
    # test fit_transform
    assert_array_almost_equal(usf.transform(X), usf1.fit_transform(X))
    assert_equal(usf.transform(X).shape[1], n_components)
    assert_array_almost_equal(usf.inverse_transform(usf.transform(X)), X)

    # Test with average param
    usf = UnsupervisedSpatialFilter(PCA(4), average=True)
    usf.fit_transform(X)
    usf = UnsupervisedSpatialFilter(PCA(4), 2)
    with pytest.raises(TypeError, match="average must be"):
        usf.fit(X)


def test_temporal_filter():
    """Test methods of TemporalFilter."""
    X = np.random.rand(5, 5, 1200)

    # Test init test
    values = (
        ("10hz", None, 100.0, "auto"),
        (5.0, "10hz", 100.0, "auto"),
        (10.0, 20.0, 5.0, "auto"),
        (None, None, 100.0, "5hz"),
    )
    for low, high, sf, ltrans in values:
        filt = TemporalFilter(low, high, sf, ltrans, fir_design="firwin")
        pytest.raises(ValueError, filt.fit_transform, X)

    # Add tests for different combinations of l_freq and h_freq
    for low, high in ((5.0, 15.0), (None, 15.0), (5.0, None)):
        filt = TemporalFilter(low, high, sfreq=100.0, fir_design="firwin")
        Xt = filt.fit_transform(X)
        assert_array_equal(filt.fit_transform(X), Xt)
        assert X.shape == Xt.shape

    # Test fit and transform numpy type check
    with pytest.raises(ValueError):
        filt.transform("foo")

    # Test with 2 dimensional data array
    X = np.random.rand(101, 500)
    filt = TemporalFilter(
        l_freq=25.0, h_freq=50.0, sfreq=1000.0, filter_length=150, fir_design="firwin2"
    )
    with use_log_level("error"):  # warning about transition bandwidth
        assert_equal(filt.fit_transform(X).shape, X.shape)


def test_bad_triage():
    """Test for gh-10924."""
    filt = TemporalFilter(l_freq=8, h_freq=60, sfreq=160.0)
    # Used to fail with "ValueError: Effective band-stop frequency (135.0) is
    # too high (maximum based on Nyquist is 80.0)"
    assert not hasattr(filt, "fitted_")
    filt.fit_transform(np.zeros((1, 1, 481)))
    assert filt.fitted_


@pytest.mark.filterwarnings("ignore:.*filter_length.*")
@parametrize_with_checks(
    [
        FilterEstimator(info, l_freq=1, h_freq=10),
        PSDEstimator(),
        Scaler(scalings="mean"),
        # Not easy to test Scaler(info) b/c number of channels must match
        TemporalFilter(),
        UnsupervisedSpatialFilter(PCA()),
        Vectorizer(),
    ]
)
def test_sklearn_compliance(estimator, check):
    """Test LinearModel compliance with sklearn."""
    ignores = []
    if estimator.__class__.__name__ == "FilterEstimator":
        ignores += [
            "check_estimators_overwrite_params",  # we modify self.info
            "check_methods_sample_order_invariance",
        ]
    if estimator.__class__.__name__.startswith(("PSD", "Temporal")):
        ignores += [
            "check_transformers_unfitted",  # allow unfitted transform
            "check_methods_sample_order_invariance",
        ]
    if any(ignore in str(check) for ignore in ignores):
        return
    check(estimator)
