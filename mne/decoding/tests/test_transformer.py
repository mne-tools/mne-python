# Author: Mainak Jas <mainak@neuro.hut.fi>
#         Romain Trachel <trachelr@gmail.com>
#
# License: BSD (3-clause)

import warnings
import os.path as op
import numpy as np

from nose.tools import assert_true, assert_raises
from numpy.testing import (assert_array_equal, assert_equal,
                           assert_array_almost_equal)

from mne import io, read_events, Epochs, pick_types
from mne.decoding import Scaler, FilterEstimator
from mne.decoding import (PSDEstimator, EpochsVectorizer, Vectorizer,
                          UnsupervisedSpatialFilter, TemporalFilter)
from mne.utils import requires_sklearn_0_15

warnings.simplefilter('always')  # enable b/c these tests throw warnings

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
start, stop = 0, 8

data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')


def test_scaler():
    """Test methods of Scaler."""
    raw = io.read_raw_fif(raw_fname, preload=False, add_eeg_ref=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]

    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True, add_eeg_ref=False)
    epochs_data = epochs.get_data()
    scaler = Scaler(epochs.info)
    y = epochs.events[:, -1]

    # np invalid divide value warnings
    with warnings.catch_warnings(record=True):
        X = scaler.fit_transform(epochs_data, y)
        assert_true(X.shape == epochs_data.shape)
        X2 = scaler.fit(epochs_data, y).transform(epochs_data)

    assert_array_equal(X2, X)

    # Test inverse_transform
    with warnings.catch_warnings(record=True):  # invalid value in mult
        Xi = scaler.inverse_transform(X, y)
    assert_array_equal(epochs_data, Xi)

    # Test init exception
    assert_raises(ValueError, scaler.fit, epochs, y)
    assert_raises(ValueError, scaler.transform, epochs, y)


def test_filterestimator():
    """Test methods of FilterEstimator."""
    raw = io.read_raw_fif(raw_fname, preload=False, add_eeg_ref=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True, add_eeg_ref=False)
    epochs_data = epochs.get_data()

    # Add tests for different combinations of l_freq and h_freq
    filt = FilterEstimator(epochs.info, l_freq=40, h_freq=80,
                           filter_length='auto',
                           l_trans_bandwidth='auto', h_trans_bandwidth='auto')
    y = epochs.events[:, -1]
    with warnings.catch_warnings(record=True):  # stop freq attenuation warning
        X = filt.fit_transform(epochs_data, y)
        assert_true(X.shape == epochs_data.shape)
        assert_array_equal(filt.fit(epochs_data, y).transform(epochs_data), X)

    filt = FilterEstimator(epochs.info, l_freq=None, h_freq=40,
                           filter_length='auto',
                           l_trans_bandwidth='auto', h_trans_bandwidth='auto')
    y = epochs.events[:, -1]
    with warnings.catch_warnings(record=True):  # stop freq attenuation warning
        X = filt.fit_transform(epochs_data, y)

    filt = FilterEstimator(epochs.info, l_freq=1, h_freq=1)
    y = epochs.events[:, -1]
    with warnings.catch_warnings(record=True):  # stop freq attenuation warning
        assert_raises(ValueError, filt.fit_transform, epochs_data, y)

    filt = FilterEstimator(epochs.info, l_freq=40, h_freq=None,
                           filter_length='auto',
                           l_trans_bandwidth='auto', h_trans_bandwidth='auto')
    with warnings.catch_warnings(record=True):  # stop freq attenuation warning
        X = filt.fit_transform(epochs_data, y)

    # Test init exception
    assert_raises(ValueError, filt.fit, epochs, y)
    assert_raises(ValueError, filt.transform, epochs, y)


def test_psdestimator():
    """Test methods of PSDEstimator."""
    raw = io.read_raw_fif(raw_fname, preload=False, add_eeg_ref=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True, add_eeg_ref=False)
    epochs_data = epochs.get_data()
    psd = PSDEstimator(2 * np.pi, 0, np.inf)
    y = epochs.events[:, -1]
    X = psd.fit_transform(epochs_data, y)

    assert_true(X.shape[0] == epochs_data.shape[0])
    assert_array_equal(psd.fit(epochs_data, y).transform(epochs_data), X)

    # Test init exception
    assert_raises(ValueError, psd.fit, epochs, y)
    assert_raises(ValueError, psd.transform, epochs, y)


def test_epochs_vectorizer():
    """Test methods of EpochsVectorizer."""
    raw = io.read_raw_fif(raw_fname, preload=False, add_eeg_ref=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]
    with warnings.catch_warnings(record=True):
        epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), preload=True, add_eeg_ref=False)
    epochs_data = epochs.get_data()
    with warnings.catch_warnings(record=True):  # deprecation
        vector = EpochsVectorizer(epochs.info)
    y = epochs.events[:, -1]
    X = vector.fit_transform(epochs_data, y)

    # Check data dimensions
    assert_true(X.shape[0] == epochs_data.shape[0])
    assert_true(X.shape[1] == epochs_data.shape[1] * epochs_data.shape[2])

    assert_array_equal(vector.fit(epochs_data, y).transform(epochs_data), X)

    # Check if data is preserved
    n_times = epochs_data.shape[2]
    assert_array_equal(epochs_data[0, 0, 0:n_times], X[0, 0:n_times])

    # Check inverse transform
    Xi = vector.inverse_transform(X, y)
    assert_true(Xi.shape[0] == epochs_data.shape[0])
    assert_true(Xi.shape[1] == epochs_data.shape[1])
    assert_array_equal(epochs_data[0, 0, 0:n_times], Xi[0, 0, 0:n_times])

    # check if inverse transform works with different number of epochs
    Xi = vector.inverse_transform(epochs_data[0], y)
    assert_true(Xi.shape[1] == epochs_data.shape[1])
    assert_true(Xi.shape[2] == epochs_data.shape[2])

    # Test init exception
    assert_raises(ValueError, vector.fit, epochs, y)
    assert_raises(ValueError, vector.transform, epochs, y)


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
    assert_equal(vect.fit_transform(np.random.rand(150, 18, 6, 3)).shape,
                 (150, 324))
    assert_equal(vect.fit_transform(data[1:]).shape, (149, 108))

    # check if raised errors are working correctly
    vect.fit(np.random.rand(105, 12, 3))
    assert_raises(ValueError, vect.transform, np.random.rand(105, 12, 3, 1))
    assert_raises(ValueError, vect.inverse_transform,
                  np.random.rand(102, 12, 12))


@requires_sklearn_0_15
def test_unsupervised_spatial_filter():
    """Test unsupervised spatial filter."""
    from sklearn.decomposition import PCA
    from sklearn.kernel_ridge import KernelRidge
    raw = io.read_raw_fif(raw_fname, preload=False, add_eeg_ref=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False,
                    add_eeg_ref=False)

    # Test estimator
    assert_raises(ValueError, UnsupervisedSpatialFilter, KernelRidge(2))

    # Test fit
    X = epochs.get_data()
    n_components = 4
    usf = UnsupervisedSpatialFilter(PCA(n_components))
    usf.fit(X)
    usf1 = UnsupervisedSpatialFilter(PCA(n_components))

    # test transform
    assert_equal(usf.transform(X).ndim, 3)
    # test fit_transform
    assert_array_almost_equal(usf.transform(X), usf1.fit_transform(X))
    # assert shape
    assert_equal(usf.transform(X).shape[1], n_components)

    # Test with average param
    usf = UnsupervisedSpatialFilter(PCA(4), average=True)
    usf.fit_transform(X)
    assert_raises(ValueError, UnsupervisedSpatialFilter, PCA(4), 2)


def test_temporal_filter():
    """Test methods of TemporalFilter."""
    X = np.random.rand(5, 5, 1200)

    # Test init test
    values = (('10hz', None, 100., 'auto'), (5., '10hz', 100., 'auto'),
              (10., 20., 5., 'auto'), (None, None, 100., '5hz'))
    for low, high, sf, ltrans in values:
        filt = TemporalFilter(low, high, sf, ltrans)
        assert_raises(ValueError, filt.fit_transform, X)

    # Add tests for different combinations of l_freq and h_freq
    for low, high in ((5., 15.), (None, 15.), (5., None)):
        filt = TemporalFilter(low, high, sfreq=100.)
        Xt = filt.fit_transform(X)
        assert_array_equal(filt.fit_transform(X), Xt)
        assert_true(X.shape == Xt.shape)

    # Test fit and transform numpy type check
    with warnings.catch_warnings(record=True):
        assert_raises(TypeError, filt.transform, [1, 2])

    # Test with 2 dimensional data array
    X = np.random.rand(101, 500)
    filt = TemporalFilter(l_freq=25., h_freq=50., sfreq=1000.,
                          filter_length=150)
    assert_equal(filt.fit_transform(X).shape, X.shape)
