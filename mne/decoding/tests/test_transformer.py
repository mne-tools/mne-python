# Author: Mainak Jas <mainak@neuro.hut.fi>
#         Romain Trachel <trachelr@gmail.com>
#
# License: BSD (3-clause)

import warnings
import os.path as op
import numpy as np

from nose.tools import assert_true, assert_raises
from numpy.testing import assert_array_equal

from mne import io, read_events, Epochs, pick_types
from mne.decoding import Scaler, FilterEstimator
from mne.decoding import PSDEstimator, EpochsVectorizer

warnings.simplefilter('always')  # enable b/c these tests throw warnings

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
start, stop = 0, 8

data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')


def test_scaler():
    """Test methods of Scaler
    """
    raw = io.read_raw_fif(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]

    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
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
    """Test methods of FilterEstimator
    """
    raw = io.read_raw_fif(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    epochs_data = epochs.get_data()

    # Add tests for different combinations of l_freq and h_freq
    filt = FilterEstimator(epochs.info, l_freq=1, h_freq=40)
    y = epochs.events[:, -1]
    with warnings.catch_warnings(record=True):  # stop freq attenuation warning
        X = filt.fit_transform(epochs_data, y)
        assert_true(X.shape == epochs_data.shape)
        assert_array_equal(filt.fit(epochs_data, y).transform(epochs_data), X)

    filt = FilterEstimator(epochs.info, l_freq=0, h_freq=40)
    y = epochs.events[:, -1]
    with warnings.catch_warnings(record=True):  # stop freq attenuation warning
        X = filt.fit_transform(epochs_data, y)

    filt = FilterEstimator(epochs.info, l_freq=1, h_freq=1)
    y = epochs.events[:, -1]
    with warnings.catch_warnings(record=True):  # stop freq attenuation warning
        assert_raises(ValueError, filt.fit_transform, epochs_data, y)

    filt = FilterEstimator(epochs.info, l_freq=1, h_freq=None)
    with warnings.catch_warnings(record=True):  # stop freq attenuation warning
        X = filt.fit_transform(epochs_data, y)

    # Test init exception
    assert_raises(ValueError, filt.fit, epochs, y)
    assert_raises(ValueError, filt.transform, epochs, y)


def test_psdestimator():
    """Test methods of PSDEstimator
    """
    raw = io.read_raw_fif(raw_fname, preload=False)
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

    assert_true(X.shape[0] == epochs_data.shape[0])
    assert_array_equal(psd.fit(epochs_data, y).transform(epochs_data), X)

    # Test init exception
    assert_raises(ValueError, psd.fit, epochs, y)
    assert_raises(ValueError, psd.transform, epochs, y)


def test_epochs_vectorizer():
    """Test methods of EpochsVectorizer
    """
    raw = io.read_raw_fif(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]
    with warnings.catch_warnings(record=True):
        epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), preload=True)
    epochs_data = epochs.get_data()
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
