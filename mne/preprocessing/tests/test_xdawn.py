# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import os.path as op
from nose.tools import (assert_equal, assert_raises)
from numpy.testing import assert_array_equal
from mne import (io, Epochs, read_events, pick_types,
                 compute_raw_covariance)
from mne.utils import requires_sklearn, run_tests_if_main
from mne.preprocessing.xdawn import Xdawn
from mne.decoding.transformer import EpochsVectorizer

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
    xd = Xdawn(info=epochs.info, n_components=2,
               signal_cov=None, reg=None)
    xd.fit(X, y)
    evoked = epochs[2].average()
    assert_array_equal(evoked.data, xd.evokeds_[2])

    # ========== with signal cov provided ====================
    # provide covariance object
    signal_cov = compute_raw_covariance(raw, picks=picks)
    xd = Xdawn(info=epochs.info, n_components=2,
               signal_cov=signal_cov, reg=None)
    xd.fit(X, y)
    # provide ndarray
    signal_cov = np.eye(len(picks))
    xd = Xdawn(info=epochs.info, n_components=2,
               signal_cov=signal_cov, reg=None)
    xd.fit(X, y)
    # provide ndarray of bad shape
    signal_cov = np.eye(len(picks) - 1)
    xd = Xdawn(info=epochs.info, n_components=2,
               signal_cov=signal_cov, reg=None)
    assert_raises(ValueError, xd.fit, X, y)
    # provide another type
    signal_cov = 42
    xd = Xdawn(info=epochs.info, n_components=2,
               signal_cov=signal_cov, reg=None)
    assert_raises(ValueError, xd.fit, X, y)
    # fit with y as None results in error
    xd = Xdawn(info=epochs.info, n_components=2,
               signal_cov=None, reg=None)
    assert_raises(ValueError, xd.fit, X, None)


def test_xdawn_apply_transform():
    """Test Xdawn apply and transform."""
    # get data
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)
    e = EpochsVectorizer()
    X, y = e.fit_transform(epochs)
    n_components = 2
    # Fit Xdawn
    xd = Xdawn(info=epochs.info, n_components=n_components)
    xd.fit(X, y)

    # apply on raw
    xd.apply(raw)
    # apply on epochs
    denoise = xd.apply(epochs)
    # apply on evoked
    xd.apply(epochs.average())
    # apply on other thing should raise an error
    assert_raises(ValueError, xd.apply, 42)

    # transform
    xd.transform(X, y)
    # transform on someting else
    assert_raises(ValueError, xd.transform, 42, 55)

    # check numerical results with shuffled epochs
    idx = np.arange(len(epochs))
    np.random.shuffle(idx)
    xd.fit(epochs[idx])
    denoise_shfl = xd.apply(epochs)
    assert_array_equal(denoise['cond2']._data, denoise_shfl['cond2']._data)


@requires_sklearn
def test_xdawn_regularization():
    """Test Xdawn with regularization."""
    # get data
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)

    e = EpochsVectorizer()
    X, y = e.fit_transform(epochs)
    # ========== with cov regularization ====================
    # ledoit-wolf
    xd = Xdawn(info=epochs.info, n_components=2,
               signal_cov=np.eye(len(picks)), reg='ledoit_wolf')
    xd.fit(X, y)
    # oas
    xd = Xdawn(info=epochs.info, n_components=2,
               signal_cov=np.eye(len(picks)), reg='oas')
    xd.fit(X, y)
    # with shrinkage
    xd = Xdawn(info=epochs.info, n_components=2,
               signal_cov=np.eye(len(picks)), reg=0.1)
    xd.fit(X, y)
    # with bad shrinkage
    xd = Xdawn(info=epochs.info, n_components=2,
               signal_cov=np.eye(len(picks)), reg=2)
    assert_raises(ValueError, xd.fit, X, y)

run_tests_if_main()
