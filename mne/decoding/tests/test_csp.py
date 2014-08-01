# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Romain Trachel <romain.trachel@inria.fr>
#
# License: BSD (3-clause)

import os.path as op

from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import assert_array_almost_equal

from mne import io, Epochs, read_events, pick_types
from mne.decoding.csp import CSP
from mne.utils import _TempDir, requires_sklearn

tempdir = _TempDir()

data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
start, stop = 0, 8  # if stop is too small pca may fail in some cases, but
                    # we're okay on this file


def test_csp():
    """Test Common Spatial Patterns algorithm on epochs
    """
    raw = io.Raw(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                            eog=False, exclude='bads')
    picks = picks[1:13:3]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    epochs_data = epochs.get_data()
    n_channels = epochs_data.shape[1]

    n_components = 3
    csp = CSP(n_components=n_components)

    csp.fit(epochs_data, epochs.events[:, -1])
    y = epochs.events[:, -1]
    X = csp.fit_transform(epochs_data, y)
    assert_true(csp.filters_.shape == (n_channels, n_channels))
    assert_true(csp.patterns_.shape == (n_channels, n_channels))
    assert_array_almost_equal(csp.fit(epochs_data, y).transform(epochs_data),
                              X)

    # test init exception
    assert_raises(ValueError, csp.fit, epochs_data,
                  np.zeros_like(epochs.events))
    assert_raises(ValueError, csp.fit, epochs, y)
    assert_raises(ValueError, csp.transform, epochs, y)

    csp.n_components = n_components
    sources = csp.transform(epochs_data)
    assert_true(sources.shape[1] == n_components)


@requires_sklearn
def test_regularized_csp():
    """Test Common Spatial Patterns algorithm using regularized covariance
    """
    raw = io.Raw(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                            eog=False, exclude='bads')
    picks = picks[1:13:3]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    epochs_data = epochs.get_data()
    n_channels = epochs_data.shape[1]

    n_components = 3
    reg_cov = [None, 0.05, 'lws', 'oas']
    for reg in reg_cov:
        csp = CSP(n_components=n_components, reg=reg)
        csp.fit(epochs_data, epochs.events[:, -1])
        y = epochs.events[:, -1]
        X = csp.fit_transform(epochs_data, y)
        assert_true(csp.filters_.shape == (n_channels, n_channels))
        assert_true(csp.patterns_.shape == (n_channels, n_channels))
        assert_array_almost_equal(csp.fit(epochs_data, y).
                                  transform(epochs_data), X)

        # test init exception
        assert_raises(ValueError, csp.fit, epochs_data,
                      np.zeros_like(epochs.events))
        assert_raises(ValueError, csp.fit, epochs, y)
        assert_raises(ValueError, csp.transform, epochs, y)

        csp.n_components = n_components
        sources = csp.transform(epochs_data)
        assert_true(sources.shape[1] == n_components)
