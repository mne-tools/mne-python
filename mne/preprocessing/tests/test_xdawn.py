# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import os.path as op
import sys

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_allclose)
import pytest
from scipy import linalg, stats

from mne import (Epochs, read_events, pick_types, compute_raw_covariance,
                 create_info, EpochsArray)
from mne.decoding import Vectorizer
from mne.io import read_raw_fif
from mne.utils import (requires_sklearn, run_tests_if_main, check_version,
                       _get_numpy_libs)
from mne.preprocessing.xdawn import Xdawn, _XdawnTransformer

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')

tmin, tmax = -0.1, 0.2
event_id = dict(cond2=2, cond3=3)


def _get_data():
    """Get data."""
    raw = read_raw_fif(raw_fname, verbose=False, preload=True)
    raw.set_eeg_reference(projection=True)
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
    pytest.raises(ValueError, Xdawn, correct_overlap=42)


def test_xdawn_picks():
    """Test picking with Xdawn."""
    data = np.random.RandomState(0).randn(10, 2, 10)
    info = create_info(2, 1000., ('eeg', 'misc'))
    epochs = EpochsArray(data, info)
    xd = Xdawn(correct_overlap=False)
    xd.fit(epochs)
    epochs_out = xd.apply(epochs)['1']
    assert epochs_out.info['ch_names'] == epochs.ch_names
    assert not (epochs_out.get_data()[:, 0] != data[:, 0]).any()
    assert_array_equal(epochs_out.get_data()[:, 1], data[:, 1])


def test_xdawn_fit():
    """Test Xdawn fit."""
    # Get data
    raw, events, picks = _get_data()
    raw.del_proj()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)
    # =========== Basic Fit test =================
    # Test base xdawn
    xd = Xdawn(n_components=2, correct_overlap='auto')
    xd.fit(epochs)
    # With these parameters, the overlap correction must be False
    assert not xd.correct_overlap_
    # No overlap correction should give averaged evoked
    evoked = epochs['cond2'].average()
    assert_array_equal(evoked.data, xd.evokeds_['cond2'].data)

    assert_allclose(np.linalg.norm(xd.filters_['cond2'], axis=1), 1)

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
    pytest.raises(ValueError, xd.fit, epochs)
    # Provide another type
    signal_cov = 42
    xd = Xdawn(n_components=2, correct_overlap=False,
               signal_cov=signal_cov)
    pytest.raises(ValueError, xd.fit, epochs)
    # Fit with baseline correction and overlap correction should throw an
    # error
    # XXX This is a buggy test, the epochs here don't overlap
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=(None, 0), verbose=False)

    xd = Xdawn(n_components=2, correct_overlap=True)
    pytest.raises(ValueError, xd.fit, epochs)


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
    pytest.raises(ValueError, xd.apply, 42)

    # Transform on Epochs
    xd.transform(epochs)
    # Transform on Evoked
    xd.transform(epochs.average())
    # Transform on ndarray
    xd.transform(epochs._data)
    xd.transform(epochs._data[0])
    # Transform on something else
    pytest.raises(ValueError, xd.transform, 42)

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
    # Get data, this time MEG so we can test proper reg/ch type support
    raw = read_raw_fif(raw_fname, verbose=False, preload=True)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, eeg=False, stim=False,
                       ecg=False, eog=False,
                       exclude='bads')[::8]
    raw.pick_channels([raw.ch_names[pick] for pick in picks])
    del picks
    raw.info.normalize_proj()
    epochs = Epochs(raw, events, event_id, tmin, tmax,
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
    assert xd.correct_overlap_
    evoked = epochs['cond2'].average()
    assert np.sum(np.abs(evoked.data - xd.evokeds_['cond2'].data))

    # With covariance regularization
    for reg in [.1, 0.1, 'ledoit_wolf', 'oas']:
        xd = Xdawn(n_components=2, correct_overlap=False,
                   signal_cov=np.eye(len(epochs.ch_names)), reg=reg)
        xd.fit(epochs)
    # With bad shrinkage
    xd = Xdawn(n_components=2, correct_overlap=False,
               signal_cov=np.eye(len(epochs.ch_names)), reg=2)
    with pytest.raises(ValueError, match='shrinkage must be'):
        xd.fit(epochs)
    # With rank-deficient input
    # this is a bit wacky because `epochs` has projectors on from the old raw
    # but it works as a rank-deficient test case
    xd = Xdawn(correct_overlap=False, reg=0.5)
    xd.fit(epochs)
    xd = Xdawn(correct_overlap=False, reg='diagonal_fixed')
    xd.fit(epochs)
    bad_eig = (sys.platform.startswith('win') and
               check_version('numpy', '1.16.5') and
               'mkl_rt' in _get_numpy_libs())  # some problem with MKL on Win
    if bad_eig:
        pytest.skip('Unknown MKL+Windows error fails for eig check')
    xd = Xdawn(correct_overlap=False, reg=None)
    with pytest.raises(ValueError, match='Could not compute eigenvalues'):
        xd.fit(epochs)


@requires_sklearn
def test_XdawnTransformer():
    """Test _XdawnTransformer."""
    # Get data
    raw, events, picks = _get_data()
    raw.del_proj()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)
    X = epochs._data
    y = epochs.events[:, -1]
    # Fit
    xdt = _XdawnTransformer()
    xdt.fit(X, y)
    pytest.raises(ValueError, xdt.fit, X, y[1:])
    pytest.raises(ValueError, xdt.fit, 'foo')

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
    pytest.raises(ValueError, xdt.fit, X, y)
    # Provide another type
    signal_cov = 42
    xdt = _XdawnTransformer(signal_cov=signal_cov)
    pytest.raises(ValueError, xdt.fit, X, y)

    # Fit with y as None
    xdt = _XdawnTransformer()
    xdt.fit(X)

    # Compare xdawn and _XdawnTransformer
    xd = Xdawn(correct_overlap=False)
    xd.fit(epochs)

    xdt = _XdawnTransformer()
    xdt.fit(X, y)
    assert_array_almost_equal(xd.filters_['cond2'][:2, :],
                              xdt.filters_.reshape(2, 2, 8)[0])

    # Transform testing
    xdt.transform(X[1:, ...])  # different number of epochs
    xdt.transform(X[:, :, 1:])  # different number of time
    pytest.raises(ValueError, xdt.transform, X[:, 1:, :])
    Xt = xdt.transform(X)
    pytest.raises(ValueError, xdt.transform, 42)

    # Inverse transform testing
    Xinv = xdt.inverse_transform(Xt)
    assert Xinv.shape == X.shape
    xdt.inverse_transform(Xt[1:, ...])
    xdt.inverse_transform(Xt[:, :, 1:])
    # should raise an error if not correct number of components
    pytest.raises(ValueError, xdt.inverse_transform, Xt[:, 1:, :])
    pytest.raises(ValueError, xdt.inverse_transform, 42)


def _simulate_erplike_mixed_data(n_epochs=100, n_channels=10):
    rng = np.random.RandomState(42)
    tmin, tmax = 0., 1.
    sfreq = 100.
    informative_ch_idx = 0

    y = rng.randint(0, 2, n_epochs)
    n_times = int((tmax - tmin) * sfreq)
    epoch_times = np.linspace(tmin, tmax, n_times)

    target_template = 1e-6 * (epoch_times - tmax) * np.sin(
        2 * np.pi * epoch_times)
    nontarget_template = 0.7e-6 * (epoch_times - tmax) * np.sin(
        2 * np.pi * (epoch_times - 0.1))

    epoch_data = rng.randn(n_epochs, n_channels, n_times) * 5e-7
    epoch_data[y == 0, informative_ch_idx, :] += nontarget_template
    epoch_data[y == 1, informative_ch_idx, :] += target_template

    mixing_mat = linalg.svd(rng.randn(n_channels, n_channels))[0]
    mixed_epoch_data = np.dot(mixing_mat.T, epoch_data).transpose((1, 0, 2))

    events = np.zeros((n_epochs, 3), dtype=int)
    events[:, 0] = np.arange(0, n_epochs * n_times, n_times)
    events[:, 2] = y

    info = create_info(
        ch_names=['C{:02d}'.format(i) for i in range(n_channels)],
        ch_types=['eeg'] * n_channels,
        sfreq=sfreq)
    epochs = EpochsArray(mixed_epoch_data, info, events,
                         tmin=tmin,
                         event_id={'nt': 0, 't': 1})

    return epochs, mixing_mat


@requires_sklearn
def test_xdawn_decoding_performance():
    """Test decoding performance and extracted pattern on synthetic data."""
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score

    n_xdawn_comps = 3
    expected_accuracy = 0.98

    epochs, mixing_mat = _simulate_erplike_mixed_data(n_epochs=100)
    y = epochs.events[:, 2]

    # results of Xdawn and _XdawnTransformer should match
    xdawn_pipe = make_pipeline(
        Xdawn(n_components=n_xdawn_comps),
        Vectorizer(),
        MinMaxScaler(),
        LogisticRegression(solver='liblinear'))
    xdawn_trans_pipe = make_pipeline(
        _XdawnTransformer(n_components=n_xdawn_comps),
        Vectorizer(),
        MinMaxScaler(),
        LogisticRegression(solver='liblinear'))

    cv = KFold(n_splits=3, shuffle=False)
    for pipe, X in (
            (xdawn_pipe, epochs),
            (xdawn_trans_pipe, epochs.get_data())):
        predictions = np.empty_like(y, dtype=float)
        for train, test in cv.split(X, y):
            pipe.fit(X[train], y[train])
            predictions[test] = pipe.predict(X[test])

        cv_accuracy_xdawn = accuracy_score(y, predictions)
        assert_allclose(cv_accuracy_xdawn, expected_accuracy, atol=0.01)

        # for both event types, the first component should "match" the mixing
        fitted_xdawn = pipe.steps[0][1]
        if isinstance(fitted_xdawn, Xdawn):
            relev_patterns = np.concatenate(
                [comps[[0]] for comps in fitted_xdawn.patterns_.values()])
        else:
            relev_patterns = fitted_xdawn.patterns_[::n_xdawn_comps]

        for i in range(len(relev_patterns)):
            r, _ = stats.pearsonr(relev_patterns[i, :], mixing_mat[0, :])
            assert np.abs(r) > 0.99


run_tests_if_main()
