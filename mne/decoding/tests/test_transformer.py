# Author: Mainak Jas <mainak@neuro.hut.fi>
#         Romain Trachel <trachelr@gmail.com>
#
# License: BSD-3-Clause

import os.path as op
import numpy as np

import pytest
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_allclose, assert_equal)

from mne import io, read_events, Epochs, pick_types
from mne.decoding import (Scaler, FilterEstimator, PSDEstimator, Vectorizer,
                          UnsupervisedSpatialFilter, TemporalFilter)
from mne.defaults import DEFAULTS
from mne.utils import requires_sklearn, check_version

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
start, stop = 0, 8

data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')


@pytest.mark.parametrize('info, method', [
    (True, None),
    (True, dict(mag=5, grad=10, eeg=20)),
    (False, 'mean'),
    (False, 'median'),
])
def test_scaler(info, method):
    """Test methods of Scaler."""
    raw = io.read_raw_fif(raw_fname)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]

    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    epochs_data = epochs.get_data()
    y = epochs.events[:, -1]

    epochs_data_t = epochs_data.transpose([1, 0, 2])
    if method in ('mean', 'median'):
        if not check_version('sklearn'):
            with pytest.raises(ImportError, match='No module'):
                Scaler(info, method)
            return

    if info:
        info = epochs.info
    scaler = Scaler(info, method)
    X = scaler.fit_transform(epochs_data, y)
    assert_equal(X.shape, epochs_data.shape)
    if method is None or isinstance(method, dict):
        sd = DEFAULTS['scalings'] if method is None else method
        stds = np.zeros(len(picks))
        for key in ('mag', 'grad'):
            stds[pick_types(epochs.info, meg=key)] = 1. / sd[key]
        stds[pick_types(epochs.info, meg=False, eeg=True)] = 1. / sd['eeg']
        means = np.zeros(len(epochs.ch_names))
    elif method == 'mean':
        stds = np.array([np.std(ch_data) for ch_data in epochs_data_t])
        means = np.array([np.mean(ch_data) for ch_data in epochs_data_t])
    else:  # median
        percs = np.array([np.percentile(ch_data, [25, 50, 75])
                          for ch_data in epochs_data_t])
        stds = percs[:, 2] - percs[:, 0]
        means = percs[:, 1]
    assert_allclose(X * stds[:, np.newaxis] + means[:, np.newaxis],
                    epochs_data, rtol=1e-12, atol=1e-20, err_msg=method)

    X2 = scaler.fit(epochs_data, y).transform(epochs_data)
    assert_array_equal(X, X2)

    # inverse_transform
    Xi = scaler.inverse_transform(X)
    assert_array_almost_equal(epochs_data, Xi)

    # Test init exception
    pytest.raises(ValueError, Scaler, None, None)
    pytest.raises(TypeError, scaler.fit, epochs, y)
    pytest.raises(TypeError, scaler.transform, epochs)
    epochs_bad = Epochs(raw, events, event_id, 0, 0.01, baseline=None,
                        picks=np.arange(len(raw.ch_names)))  # non-data chs
    scaler = Scaler(epochs_bad.info, None)
    pytest.raises(ValueError, scaler.fit, epochs_bad.get_data(), y)


def test_filterestimator():
    """Test methods of FilterEstimator."""
    raw = io.read_raw_fif(raw_fname)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    epochs_data = epochs.get_data()

    # Add tests for different combinations of l_freq and h_freq
    filt = FilterEstimator(epochs.info, l_freq=40, h_freq=80)
    y = epochs.events[:, -1]
    X = filt.fit_transform(epochs_data, y)
    assert (X.shape == epochs_data.shape)
    assert_array_equal(filt.fit(epochs_data, y).transform(epochs_data), X)

    filt = FilterEstimator(epochs.info, l_freq=None, h_freq=40,
                           filter_length='auto',
                           l_trans_bandwidth='auto', h_trans_bandwidth='auto')
    y = epochs.events[:, -1]
    X = filt.fit_transform(epochs_data, y)

    filt = FilterEstimator(epochs.info, l_freq=1, h_freq=1)
    y = epochs.events[:, -1]
    with pytest.warns(RuntimeWarning, match='longer than the signal'):
        pytest.raises(ValueError, filt.fit_transform, epochs_data, y)

    filt = FilterEstimator(epochs.info, l_freq=40, h_freq=None,
                           filter_length='auto',
                           l_trans_bandwidth='auto', h_trans_bandwidth='auto')
    X = filt.fit_transform(epochs_data, y)

    # Test init exception
    pytest.raises(ValueError, filt.fit, epochs, y)
    pytest.raises(ValueError, filt.transform, epochs)


def test_psdestimator():
    """Test methods of PSDEstimator."""
    raw = io.read_raw_fif(raw_fname)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    epochs_data = epochs.get_data()
    psd = PSDEstimator(2 * np.pi, 0, np.inf)
    y = epochs.events[:, -1]
    X = psd.fit_transform(epochs_data, y)

    assert (X.shape[0] == epochs_data.shape[0])
    assert_array_equal(psd.fit(epochs_data, y).transform(epochs_data), X)

    # Test init exception
    pytest.raises(ValueError, psd.fit, epochs, y)
    pytest.raises(ValueError, psd.transform, epochs)


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
    pytest.raises(ValueError, vect.transform, np.random.rand(105, 12, 3, 1))
    pytest.raises(ValueError, vect.inverse_transform,
                  np.random.rand(102, 12, 12))


@requires_sklearn
def test_unsupervised_spatial_filter():
    """Test unsupervised spatial filter."""
    from sklearn.decomposition import PCA
    from sklearn.kernel_ridge import KernelRidge
    raw = io.read_raw_fif(raw_fname)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)

    # Test estimator
    pytest.raises(ValueError, UnsupervisedSpatialFilter, KernelRidge(2))

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
    assert_equal(usf.transform(X).shape[1], n_components)
    assert_array_almost_equal(usf.inverse_transform(usf.transform(X)), X)

    # Test with average param
    usf = UnsupervisedSpatialFilter(PCA(4), average=True)
    usf.fit_transform(X)
    pytest.raises(ValueError, UnsupervisedSpatialFilter, PCA(4), 2)


def test_temporal_filter():
    """Test methods of TemporalFilter."""
    X = np.random.rand(5, 5, 1200)

    # Test init test
    values = (('10hz', None, 100., 'auto'), (5., '10hz', 100., 'auto'),
              (10., 20., 5., 'auto'), (None, None, 100., '5hz'))
    for low, high, sf, ltrans in values:
        filt = TemporalFilter(low, high, sf, ltrans, fir_design='firwin')
        pytest.raises(ValueError, filt.fit_transform, X)

    # Add tests for different combinations of l_freq and h_freq
    for low, high in ((5., 15.), (None, 15.), (5., None)):
        filt = TemporalFilter(low, high, sfreq=100., fir_design='firwin')
        Xt = filt.fit_transform(X)
        assert_array_equal(filt.fit_transform(X), Xt)
        assert (X.shape == Xt.shape)

    # Test fit and transform numpy type check
    with pytest.raises(ValueError, match='Data to be filtered must be'):
        filt.transform([1, 2])

    # Test with 2 dimensional data array
    X = np.random.rand(101, 500)
    filt = TemporalFilter(l_freq=25., h_freq=50., sfreq=1000.,
                          filter_length=150, fir_design='firwin2')
    assert_equal(filt.fit_transform(X).shape, X.shape)
