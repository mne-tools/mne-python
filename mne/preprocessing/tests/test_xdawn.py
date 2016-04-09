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


def test_xdawn_init():
    """Test init of xdawn."""
    # init xdawn with good parameters
    Xdawn(n_components=2, correct_overlap='auto', signal_cov=None, reg=None)
    # init xdawn with bad parameters
    assert_raises(ValueError, Xdawn, correct_overlap=42)


def test_xdawn_fit():
    """Test Xdawn fit."""
    # get data
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)
    # =========== Basic Fit test =================
    # test base xdawn
    xd = Xdawn(n_components=2, correct_overlap='auto',
               signal_cov=None, reg=None)
    xd.fit(epochs)
    # with this parameters, the overlapp correction must be False
    assert_equal(xd.correct_overlap, False)
    # no overlapp correction should give averaged evoked
    evoked = epochs['cond2'].average()
    assert_array_equal(evoked.data, xd.evokeds_['cond2'].data)

    # ========== with signal cov provided ====================
    # provide covariance object
    signal_cov = compute_raw_covariance(raw, picks=picks)
    xd = Xdawn(n_components=2, correct_overlap=False,
               signal_cov=signal_cov, reg=None)
    xd.fit(epochs)
    # provide ndarray
    signal_cov = np.eye(len(picks))
    xd = Xdawn(n_components=2, correct_overlap=False,
               signal_cov=signal_cov, reg=None)
    xd.fit(epochs)
    # provide ndarray of bad shape
    signal_cov = np.eye(len(picks) - 1)
    xd = Xdawn(n_components=2, correct_overlap=False,
               signal_cov=signal_cov, reg=None)
    assert_raises(ValueError, xd.fit, epochs)
    # provide another type
    signal_cov = 42
    xd = Xdawn(n_components=2, correct_overlap=False,
               signal_cov=signal_cov, reg=None)
    assert_raises(ValueError, xd.fit, epochs)
    # fit with baseline correction and ovverlapp correction should throw an
    # error
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=(None, 0), verbose=False)

    xd = Xdawn(n_components=2, correct_overlap=True)
    assert_raises(ValueError, xd.fit, epochs)


def test_xdawn_apply_transform():
    """Test Xdawn apply and transform."""
    # get data
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)
    n_components = 2
    # Fit Xdawn
    xd = Xdawn(n_components=n_components, correct_overlap='auto')
    xd.fit(epochs)

    # apply on raw
    xd.apply(raw)
    # apply on epochs
    xd.apply(epochs)
    # apply on evoked
    xd.apply(epochs.average())
    # apply on other thing should raise an error
    assert_raises(ValueError, xd.apply, 42)

    # transform on epochs
    xd.transform(epochs)
    # transform on ndarray
    xd.transform(epochs._data)
    # transform on someting else
    assert_raises(ValueError, xd.transform, 42)


@requires_sklearn
def test_xdawn_regularization():
    """Test Xdawn with regularization."""
    # get data
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)

    # test xdawn with overlap correction
    xd = Xdawn(n_components=2, correct_overlap=True,
               signal_cov=None, reg=0.1)
    xd.fit(epochs)
    # ========== with cov regularization ====================
    # ledoit-wolf
    xd = Xdawn(n_components=2, correct_overlap=False,
               signal_cov=np.eye(len(picks)), reg='ledoit_wolf')
    xd.fit(epochs)
    # oas
    xd = Xdawn(n_components=2, correct_overlap=False,
               signal_cov=np.eye(len(picks)), reg='oas')
    xd.fit(epochs)
    # with shrinkage
    xd = Xdawn(n_components=2, correct_overlap=False,
               signal_cov=np.eye(len(picks)), reg=0.1)
    xd.fit(epochs)
    # with bad shrinkage
    xd = Xdawn(n_components=2, correct_overlap=False,
               signal_cov=np.eye(len(picks)), reg=2)
    assert_raises(ValueError, xd.fit, epochs)

run_tests_if_main()
