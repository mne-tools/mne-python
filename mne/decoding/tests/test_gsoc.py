import numpy as np
import os.path as op
from numpy.testing import (assert_array_equal, assert_equal,
                          assert_array_almost_equal)
from nose.tools import assert_raises
from mne import (io, Epochs, read_events, pick_types,
                compute_raw_covariance)
from mne.decoding.gsoc import (_EpochsTransformerMixin,
                               UnsupervisedSpatialFilter,
                               XdawnTransformer, Vectorizer)
from mne.decoding.transformer import EpochsVectorizer
from mne.utils import run_tests_if_main, requires_sklearn

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')

tmin, tmax = -0.1, 0.2
event_id = dict(cond2=2, cond3=3)


def _get_data():
    raw = io.read_raw_fif(raw_fname, add_eeg_ref=False, verbose=False,
                          preload=True)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False,
                       ecg=False, eog=False,
                       exclude='bads')[::8]
    return raw, events, picks


def test_epochs_transformer_mixin():
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, preload=True,
                    baseline=None, verbose=False)
    ch_names = [epochs.ch_names[p] for p in picks]
    epochs.pick_channels(ch_names)

    # Test _rehsape method wrong input
    etm = _EpochsTransformerMixin(n_chan=epochs.info['nchan'])
    assert_raises(ValueError, etm._reshape, raw)

    # Test _reshape correctness
    X, y = EpochsVectorizer().fit_transform(epochs)
    assert_array_equal(etm._reshape(X), epochs._data)
    assert_equal(etm._reshape(X).ndim, epochs._data.ndim)


@requires_sklearn
def test_unsupervised_spatial_filter():
    from sklearn.decomposition import PCA
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)

    # Test fit
    X, y = EpochsVectorizer().fit_transform(epochs)
    usf = UnsupervisedSpatialFilter(PCA(5), n_chan=epochs.info['nchan'])
    usf.fit(X)
    usf1 = UnsupervisedSpatialFilter(PCA(5), n_chan=epochs.info['nchan'])

    # test transform
    assert_equal(usf.transform(X).ndim, 3)

    # test fit_trasnform
    print(usf.transform(X), usf1.fit_transform(X))
    assert_array_almost_equal(usf.transform(X), usf1.fit_transform(X))

    # assert shape
    assert_equal(usf.transform(X).shape[1], 5)


@requires_sklearn
def test_xdawn_fit():
    """Test Xdawn fit."""
    # get data
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)
    e = EpochsVectorizer()
    X, y = e.fit_transform(epochs)
    print(y)
    # =========== Basic Fit test =================
    # test base xdawn
    xd = XdawnTransformer(n_chan=epochs.info['nchan'], n_components=2,
                          signal_cov=None, reg=None)
    xd.fit(X, y)

    # ========== with signal cov provided ====================
    # provide covariance object
    signal_cov = compute_raw_covariance(raw, picks=picks)
    xd = XdawnTransformer(n_chan=epochs.info['nchan'], n_components=2,
                          signal_cov=signal_cov, reg=None)
    xd.fit(X, y)
    # provide ndarray
    signal_cov = np.eye(len(picks))
    xd = XdawnTransformer(n_chan=epochs.info['nchan'], n_components=2,
                          signal_cov=signal_cov, reg=None)
    xd.fit(X, y)
    # provide ndarray of bad shape
    signal_cov = np.eye(len(picks) - 1)
    xd = XdawnTransformer(n_chan=epochs.info['nchan'], n_components=2,
                          signal_cov=signal_cov, reg=None)
    assert_raises(ValueError, xd.fit, X, y)
    # provide another type
    signal_cov = 42
    xd = XdawnTransformer(n_chan=epochs.info['nchan'], n_components=2,
                          signal_cov=signal_cov, reg=None)
    assert_raises(ValueError, xd.fit, X, y)
    # fit with y as None results in error
    xd = XdawnTransformer(n_chan=epochs.info['nchan'], n_components=2,
                          signal_cov=None, reg=None)
    assert_raises(ValueError, xd.fit, X, None)


@requires_sklearn
def test_xdawn_transform_and_inverse_transform():
    """Test Xdawn apply and transform."""
    # get data
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)
    e = EpochsVectorizer()
    X, y = e.fit_transform(epochs)
    n_components = 2
    # Fit Xdawn
    xd = XdawnTransformer(n_chan=epochs.info['nchan'],
                          n_components=n_components)
    xd.fit(X, y)

    # transform
    xd.transform(X)
    # transform on someting else
    assert_raises(ValueError, xd.transform, 42)

    # inverse transform testing
    xd.inverse_transform(X)

    # should raise an error if not np.ndarray
    assert_raises(ValueError, xd.inverse_transform, 42)

def test_vectorizer():
    """Test Vectorizer."""

    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)
    vector_data = Vectorizer().fit_transform(epochs._data)
    assert_equal(vector_data.ndim, 2)

run_tests_if_main()
