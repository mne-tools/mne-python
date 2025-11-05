# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_array_equal

pytest.importorskip("sklearn")

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mne import Epochs, create_info, io, pick_types, read_events
from mne.decoding import (
    CSP,
    LinearModel,
    SpatialFilter,
    Vectorizer,
    XdawnTransformer,
    get_spatial_filter_from_estimator,
)

data_dir = Path(__file__).parents[2] / "io" / "tests" / "data"
raw_fname = data_dir / "test_raw.fif"
event_name = data_dir / "test-eve.fif"
tmin, tmax = -0.1, 0.2
event_id = dict(aud_l=1, vis_l=3)
start, stop = 0, 8


def _get_X_y(event_id, return_info=False):
    raw = io.read_raw(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(
        raw.info, meg=True, stim=False, ecg=False, eog=False, exclude="bads"
    )
    picks = picks[2:12:3]  # subselect channels -> disable proj!
    raw.add_proj([], remove_existing=True)
    epochs = Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        picks=picks,
        baseline=(None, 0),
        preload=True,
        proj=False,
    )
    X = epochs.get_data(copy=False, units=dict(eeg="uV", grad="fT/cm", mag="fT"))
    y = epochs.events[:, -1]
    if return_info:
        return X, y, epochs.info
    return X, y


def test_spatial_filter_init():
    """Test the initialization of the SpatialFilter class."""
    # Test initialization and factory function
    rng = np.random.RandomState(0)
    n, n_features = 20, 3
    X = rng.rand(n, n_features)
    n_targets = 5
    y = rng.rand(n, n_targets)
    clf = LinearModel(LinearRegression())
    clf.fit(X, y)

    # test get_spatial_filter_from_estimator for LinearModel
    info = create_info(n_features, 1000.0, "eeg")
    sp_filter = get_spatial_filter_from_estimator(clf, info)
    assert sp_filter.patterns_method == "haufe"
    assert_array_equal(sp_filter.filters, clf.filters_)
    assert_array_equal(sp_filter.patterns, clf.patterns_)
    assert sp_filter.evals is None

    with pytest.raises(ValueError, match="can only include"):
        _ = get_spatial_filter_from_estimator(
            clf, info, get_coefs=("foo", "foo", "foo")
        )

    event_id = dict(aud_l=1, vis_l=3)
    X, y, info = _get_X_y(event_id, return_info=True)
    estimator = make_pipeline(Vectorizer(), StandardScaler(), CSP(n_components=4))
    estimator.fit(X, y)
    csp = estimator[-1]
    # test get_spatial_filter_from_estimator for GED
    sp_filter = get_spatial_filter_from_estimator(estimator, info, step_name="csp")
    assert sp_filter.patterns_method == "pinv"
    assert_array_equal(sp_filter.filters, csp.filters_)
    assert_array_equal(sp_filter.patterns, csp.patterns_)
    assert_array_equal(sp_filter.evals, csp.evals_)
    assert sp_filter.info is info

    # test without step_name
    sp_filter = get_spatial_filter_from_estimator(estimator, info)
    assert_array_equal(sp_filter.filters, csp.filters_)
    assert_array_equal(sp_filter.patterns, csp.patterns_)
    assert_array_equal(sp_filter.evals, csp.evals_)

    # test basic initialization
    sp_filter = SpatialFilter(
        info, filters=csp.filters_, patterns=csp.patterns_, evals=csp.evals_
    )
    assert_array_equal(sp_filter.filters, csp.filters_)
    assert_array_equal(sp_filter.patterns, csp.patterns_)
    assert_array_equal(sp_filter.evals, csp.evals_)
    assert sp_filter.info is info

    # test automatic pattern calculation via pinv
    sp_filter_pinv = SpatialFilter(info, filters=csp.filters_, evals=csp.evals_)
    patterns_pinv = np.linalg.pinv(csp.filters_.T)
    assert_array_equal(sp_filter_pinv.patterns, patterns_pinv)
    assert sp_filter_pinv.patterns_method == "pinv"

    # test shape mismatch error
    with pytest.raises(ValueError, match="Shape mismatch"):
        SpatialFilter(info, filters=csp.filters_, patterns=csp.patterns_[:-1])

    # test invalid patterns_method
    with pytest.raises(ValueError, match="patterns_method"):
        SpatialFilter(info, filters=csp.filters_, patterns_method="foo")

    # test n_components > n_channels error
    bad_filters = np.random.randn(31, 30)  # 31 components, 30 channels
    with pytest.raises(ValueError, match="Number of components can't be greater"):
        SpatialFilter(info, filters=bad_filters)


def test_spatial_filter_plotting():
    """Test the plotting methods of SpatialFilter."""
    event_id = dict(aud_l=1, vis_l=3)
    X, y, info = _get_X_y(event_id, return_info=True)
    csp = CSP(n_components=4)
    csp.fit(X, y)

    sp_filter = get_spatial_filter_from_estimator(csp, info)

    # test plot_filters
    fig_filters = sp_filter.plot_filters(components=[0, 1], show=False)
    assert isinstance(fig_filters, plt.Figure)
    plt.close("all")

    # test plot_patterns
    fig_patterns = sp_filter.plot_patterns(show=False)
    assert isinstance(fig_patterns, plt.Figure)
    plt.close("all")

    # test plot_scree
    fig_scree = sp_filter.plot_scree(show=False, add_cumul_evals=True)
    assert isinstance(fig_scree, plt.Figure)
    plt.close("all")
    _, axes = plt.subplots(figsize=(12, 7), layout="constrained")
    fig_scree = sp_filter.plot_scree(axes=axes, show=False)
    assert fig_scree == list()
    plt.close("all")

    # test plot_scree raises error if evals is None
    sp_filter_no_evals = SpatialFilter(info, filters=csp.filters_, evals=None)
    with pytest.raises(AttributeError, match="eigenvalues are not provided"):
        sp_filter_no_evals.plot_scree()

    # 3D case ('multi' GED decomposition)
    n_classes = 2
    event_id = dict(aud_l=1, vis_l=3)
    X, y, info = _get_X_y(event_id, return_info=True)
    xdawn = XdawnTransformer(n_components=4)
    xdawn.fit(X, y)
    sp_filter = get_spatial_filter_from_estimator(xdawn, info)

    fig_patterns = sp_filter.plot_patterns(show=False)
    assert len(fig_patterns) == n_classes
    plt.close("all")

    fig_scree = sp_filter.plot_scree(show=False)
    assert len(fig_scree) == n_classes
    plt.close("all")

    with pytest.raises(ValueError, match="but expected"):
        _, axes = plt.subplots(figsize=(12, 7), layout="constrained")
        _ = sp_filter.plot_scree(axes=axes, show=False)

    _, axes = plt.subplots(n_classes, figsize=(12, 7), layout="constrained")
    fig_scree = sp_filter.plot_scree(axes=axes, show=False)
    assert fig_scree == list()
    plt.close("all")
