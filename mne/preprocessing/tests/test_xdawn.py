# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import os.path as op
from nose.tools import assert_equal, assert_raises, assert_true
from numpy.testing import assert_array_equal, assert_array_almost_equal
from mne import Epochs, read_events, pick_types, compute_raw_covariance
from mne.io import read_raw_fif
from mne.utils import requires_sklearn, run_tests_if_main
from mne.preprocessing.xdawn import Xdawn, _XdawnTransformer

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')

tmin, tmax = -0.1, 0.2
event_id = dict(cond2=2, cond3=3)


def _get_data():
    """Get data."""
    raw = read_raw_fif(raw_fname, verbose=False, preload=True)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False,
                       ecg=False, eog=False,
                       exclude='bads')[::8]
    return raw, events, picks


def test_xdawn():
    """Test init of xdawn."""
    # Init xdawn with good parameters
    Xdawn(n_components=2, correct_overlap='auto', signal_cov=None, reg=None)
    # Init xdawn with bad parameters
    assert_raises(ValueError, Xdawn, correct_overlap=42)


def test_xdawn_fit():
    """Test Xdawn fit."""
    # Get data
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)
    # =========== Basic Fit test =================
    # Test base xdawn
    xd = Xdawn(n_components=2, correct_overlap='auto')
    xd.fit(epochs)
    # With these parameters, the overlap correction must be False
    assert_equal(xd.correct_overlap_, False)
    # No overlap correction should give averaged evoked
    evoked = epochs['cond2'].average()
    assert_array_equal(evoked.data, xd.evokeds_['cond2'].data)

    # ========== with signal cov provided ====================
    # Provide covariance object
    signal_cov = compute_raw_covariance(raw, picks=picks)
    xd = Xdawn(n_components=2, correct_overlap=False,
               signal_cov=signal_cov)
    xd.fit(epochs)
    # Provide ndarray
    signal_cov = np.eye(len(picks))
    xd = Xdawn(n_components=2, correct_overlap=False,
               signal_cov=signal_cov)
    xd.fit(epochs)
    # Provide ndarray of bad shape
    signal_cov = np.eye(len(picks) - 1)
    xd = Xdawn(n_components=2, correct_overlap=False,
               signal_cov=signal_cov)
    assert_raises(ValueError, xd.fit, epochs)
    # Provide another type
    signal_cov = 42
    xd = Xdawn(n_components=2, correct_overlap=False,
               signal_cov=signal_cov)
    assert_raises(ValueError, xd.fit, epochs)
    # Fit with baseline correction and overlap correction should throw an
    # error
    # XXX This is a buggy test, the epochs here don't overlap
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=(None, 0), verbose=False)

    xd = Xdawn(n_components=2, correct_overlap=True)
    assert_raises(ValueError, xd.fit, epochs)


def test_xdawn_apply_transform():
    """Test Xdawn apply and transform."""
    # Get data
    raw, events, picks = _get_data()
    raw.pick_types(eeg=True, meg=False)
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    preload=True, baseline=None,
                    verbose=False)
    n_components = 2
    # Fit Xdawn
    xd = Xdawn(n_components=n_components, correct_overlap=False)
    xd.fit(epochs)

    # Apply on different types of instances
    for inst in [raw, epochs.average(), epochs]:
        denoise = xd.apply(inst)
    # Apply on other thing should raise an error
    assert_raises(ValueError, xd.apply, 42)

    # Transform on epochs
    xd.transform(epochs)
    # Transform on ndarray
    xd.transform(epochs._data)
    # Transform on someting else
    assert_raises(ValueError, xd.transform, 42)

    # Check numerical results with shuffled epochs
    np.random.seed(0)  # random makes unstable linalg
    idx = np.arange(len(epochs))
    np.random.shuffle(idx)
    xd.fit(epochs[idx])
    denoise_shfl = xd.apply(epochs)
    assert_array_almost_equal(denoise['cond2']._data,
                              denoise_shfl['cond2']._data)


@requires_sklearn
def test_xdawn_regularization():
    """Test Xdawn with regularization."""
    # Get data
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)

    # Test with overlapping events.
    # modify events to simulate one overlap
    events = epochs.events
    sel = np.where(events[:, 2] == 2)[0][:2]
    modified_event = events[sel[0]]
    modified_event[0] += 1
    epochs.events[sel[1]] = modified_event
    # Fit and check that overlap was found and applied
    xd = Xdawn(n_components=2, correct_overlap='auto', reg='oas')
    xd.fit(epochs)
    assert_equal(xd.correct_overlap_, True)
    evoked = epochs['cond2'].average()
    assert_true(np.sum(np.abs(evoked.data - xd.evokeds_['cond2'].data)))

    # With covariance regularization
    for reg in [.1, 0.1, 'ledoit_wolf', 'oas']:
        xd = Xdawn(n_components=2, correct_overlap=False,
                   signal_cov=np.eye(len(picks)), reg=reg)
        xd.fit(epochs)
    # With bad shrinkage
    xd = Xdawn(n_components=2, correct_overlap=False,
               signal_cov=np.eye(len(picks)), reg=2)
    assert_raises(ValueError, xd.fit, epochs)


@requires_sklearn
def test_XdawnTransformer():
    """Test _XdawnTransformer."""
    # Get data
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)
    X = epochs._data
    y = epochs.events[:, -1]
    # Fit
    xdt = _XdawnTransformer()
    xdt.fit(X, y)
    assert_raises(ValueError, xdt.fit, X, y[1:])
    assert_raises(ValueError, xdt.fit, 'foo')

    # Provide covariance object
    signal_cov = compute_raw_covariance(raw, picks=picks)
    xdt = _XdawnTransformer(signal_cov=signal_cov)
    xdt.fit(X, y)
    # Provide ndarray
    signal_cov = np.eye(len(picks))
    xdt = _XdawnTransformer(signal_cov=signal_cov)
    xdt.fit(X, y)
    # Provide ndarray of bad shape
    signal_cov = np.eye(len(picks) - 1)
    xdt = _XdawnTransformer(signal_cov=signal_cov)
    assert_raises(ValueError, xdt.fit, X, y)
    # Provide another type
    signal_cov = 42
    xdt = _XdawnTransformer(signal_cov=signal_cov)
    assert_raises(ValueError, xdt.fit, X, y)

    # Fit with y as None
    xdt = _XdawnTransformer()
    xdt.fit(X)

    # Compare xdawn and _XdawnTransformer
    xd = Xdawn(correct_overlap=False)
    xd.fit(epochs)

    xdt = _XdawnTransformer()
    xdt.fit(X, y)
    assert_array_almost_equal(xd.filters_['cond2'][:, :2],
                              xdt.filters_.reshape(2, 2, 8)[0].T)

    # Transform testing
    xdt.transform(X[1:, ...])  # different number of epochs
    xdt.transform(X[:, :, 1:])  # different number of time
    assert_raises(ValueError, xdt.transform, X[:, 1:, :])
    Xt = xdt.transform(X)
    assert_raises(ValueError, xdt.transform, 42)

    # Inverse transform testing
    Xinv = xdt.inverse_transform(Xt)
    assert_equal(Xinv.shape, X.shape)
    xdt.inverse_transform(Xt[1:, ...])
    xdt.inverse_transform(Xt[:, :, 1:])
    # should raise an error if not correct number of components
    assert_raises(ValueError, xdt.inverse_transform, Xt[:, 1:, :])
    assert_raises(ValueError, xdt.inverse_transform, 42)


run_tests_if_main()
