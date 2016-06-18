import numpy as np
import os.path as op
from nose.tools import (assert_raises)
from numpy.testing import assert_array_equal
from mne import (io, Epochs, read_events, pick_types,
                 compute_raw_covariance)
from mne.utils import requires_sklearn, run_tests_if_main
from mne.decoding.transformer import EpochsVectorizer
from mne.decoding import XdawnTransformer
from mne.preprocessing import Xdawn

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked_nf_name = op.join(base_dir, 'test-nf-ave.fif')

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


@requires_sklearn
def test_xdawntransformer_fit():
    """Test Xdawn fit."""
    # get data
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)
    e = EpochsVectorizer()
    X = e.fit_transform(epochs)
    y = epochs.events[:, -1]
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

    # compare xdawn and xdawntransforer
    xd = Xdawn()
    xd.fit(epochs)

    xdt = XdawnTransformer(n_chan=epochs.info['nchan'])
    xdt.fit(X, y)
    assert_array_equal(xdt.filters_['cond2'], xd.filters_['cond2'])


@requires_sklearn
def test_xdawn_transform_and_inverse_transform():
    """Test Xdawn apply and transform."""
    # get data
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)
    e = EpochsVectorizer()
    X = e.fit_transform(epochs)
    y = epochs.events[:, -1]
    # Fit Xdawn
    xd = XdawnTransformer(n_chan=epochs.info['nchan'],
                          n_components=2)
    xd.fit(X, y)

    # transform
    xd.transform(X)
    # transform on someting else
    assert_raises(ValueError, xd.transform, 42)

    # inverse transform testing
    xd.inverse_transform(X)

    # should raise an error if not np.ndarray
    assert_raises(ValueError, xd.inverse_transform, 42)

run_tests_if_main()
