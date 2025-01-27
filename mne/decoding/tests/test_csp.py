# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)

pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import parametrize_with_checks

from mne import Epochs, compute_proj_raw, io, pick_types, read_events
from mne.decoding import CSP, LinearModel, Scaler, SPoC, get_coef
from mne.decoding.csp import _ajd_pham
from mne.utils import catch_logging

data_dir = Path(__file__).parents[2] / "io" / "tests" / "data"
raw_fname = data_dir / "test_raw.fif"
event_name = data_dir / "test-eve.fif"
tmin, tmax = -0.1, 0.2
event_id = dict(aud_l=1, vis_l=3)
# if stop is too small pca may fail in some cases, but we're okay on this file
start, stop = 0, 8


def simulate_data(target, n_trials=100, n_channels=10, random_state=42):
    """Simulate data according to an instantaneous mixin model.

    Data are simulated in the statistical source space, where one source is
    modulated according to a target variable, before being mixed with a
    random mixing matrix.
    """
    rs = np.random.RandomState(random_state)

    # generate a orthogonal mixin matrix
    mixing_mat = np.linalg.svd(rs.randn(n_channels, n_channels))[0]

    S = rs.randn(n_trials, n_channels, 50)
    S[:, 0] *= np.atleast_2d(np.sqrt(target)).T
    S[:, 1:] *= 0.01  # less noise

    X = np.dot(mixing_mat, S).transpose((1, 0, 2))

    return X, mixing_mat


def deterministic_toy_data(classes=("class_a", "class_b")):
    """Generate a small deterministic toy data set.

    Four independent sources are modulated by the target class and mixed
    into signal space.
    """
    sources_a = (
        np.array(
            [
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=float,
        )
        * 2
        - 1
    )

    sources_b = (
        np.array(
            [
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=float,
        )
        * 2
        - 1
    )

    sources_a[0, :] *= 1
    sources_a[1, :] *= 2

    sources_b[2, :] *= 3
    sources_b[3, :] *= 4

    mixing = np.array(
        [
            [1.0, 0.8, 0.6, 0.4],
            [0.8, 1.0, 0.8, 0.6],
            [0.6, 0.8, 1.0, 0.8],
            [0.4, 0.6, 0.8, 1.0],
        ]
    )

    x_class_a = mixing @ sources_a
    x_class_b = mixing @ sources_b

    x = np.stack([x_class_a, x_class_b])
    y = np.array(classes)

    return x, y


@pytest.mark.slowtest
def test_csp():
    """Test Common Spatial Patterns algorithm on epochs."""
    raw = io.read_raw_fif(raw_fname, preload=False)
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
    epochs_data = epochs.get_data(copy=False)
    n_channels = epochs_data.shape[1]
    y = epochs.events[:, -1]

    # Init
    csp = CSP(n_components="foo")
    with pytest.raises(TypeError, match="must be an instance"):
        csp.fit(epochs_data, y)
    for reg in ["foo", -0.1, 1.1]:
        csp = CSP(reg=reg, norm_trace=False)
        pytest.raises(ValueError, csp.fit, epochs_data, epochs.events[:, -1])
    for reg in ["oas", "ledoit_wolf", 0, 0.5, 1.0]:
        CSP(reg=reg, norm_trace=False)
    csp = CSP(cov_est="foo", norm_trace=False)
    with pytest.raises(ValueError, match="Invalid value"):
        csp.fit(epochs_data, y)
    csp = CSP(norm_trace="foo")
    with pytest.raises(TypeError, match="instance of bool"):
        csp.fit(epochs_data, y)
    for cov_est in ["concat", "epoch"]:
        CSP(cov_est=cov_est, norm_trace=False).fit(epochs_data, y)

    n_components = 3
    # Fit
    for norm_trace in [True, False]:
        csp = CSP(n_components=n_components, norm_trace=norm_trace)
        csp.fit(epochs_data, epochs.events[:, -1])

    assert_equal(len(csp.mean_), n_components)
    assert_equal(len(csp.std_), n_components)

    # Transform
    X = csp.fit_transform(epochs_data, y)
    sources = csp.transform(epochs_data)
    assert sources.shape[1] == n_components
    assert csp.filters_.shape == (n_channels, n_channels)
    assert csp.patterns_.shape == (n_channels, n_channels)
    assert_array_almost_equal(sources, X)

    # Test data exception
    pytest.raises(ValueError, csp.fit, epochs_data, np.zeros_like(epochs.events))
    pytest.raises(ValueError, csp.fit, "foo", y)
    pytest.raises(ValueError, csp.transform, "foo")

    # Test plots
    epochs.pick(picks="mag")
    cmap = ("RdBu", True)
    components = np.arange(n_components)
    for plot in (csp.plot_patterns, csp.plot_filters):
        plot(epochs.info, components=components, res=12, show=False, cmap=cmap)

    # Test with more than 2 classes
    epochs = Epochs(
        raw,
        events,
        tmin=tmin,
        tmax=tmax,
        picks=picks,
        event_id=dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4),
        baseline=(None, 0),
        proj=False,
        preload=True,
    )
    epochs_data = epochs.get_data(copy=False)
    n_channels = epochs_data.shape[1]

    n_channels = epochs_data.shape[1]
    for cov_est in ["concat", "epoch"]:
        csp = CSP(n_components=n_components, cov_est=cov_est, norm_trace=False)
        csp.fit(epochs_data, epochs.events[:, 2]).transform(epochs_data)
        assert_equal(len(csp.classes_), 4)
        assert_array_equal(csp.filters_.shape, [n_channels, n_channels])
        assert_array_equal(csp.patterns_.shape, [n_channels, n_channels])

    # Test average power transform
    n_components = 2
    assert csp.transform_into == "average_power"
    feature_shape = [len(epochs_data), n_components]
    X_trans = dict()
    for log in (None, True, False):
        csp = CSP(n_components=n_components, log=log, norm_trace=False)
        assert csp.log is log
        Xt = csp.fit_transform(epochs_data, epochs.events[:, 2])
        assert_array_equal(Xt.shape, feature_shape)
        X_trans[str(log)] = Xt
    # log=None => log=True
    assert_array_almost_equal(X_trans["None"], X_trans["True"])
    # Different normalization return different transform
    assert np.sum((X_trans["True"] - X_trans["False"]) ** 2) > 1.0
    # Check wrong inputs
    csp = CSP(transform_into="average_power", log="foo")
    with pytest.raises(TypeError, match="must be an instance of bool"):
        csp.fit(epochs_data, epochs.events[:, 2])

    # Test csp space transform
    csp = CSP(transform_into="csp_space", norm_trace=False)
    assert csp.transform_into == "csp_space"
    for log in ("foo", True, False):
        csp = CSP(transform_into="csp_space", log=log, norm_trace=False)
        with pytest.raises(TypeError, match="must be an instance"):
            csp.fit(epochs_data, epochs.events[:, 2])
    n_components = 2
    csp = CSP(n_components=n_components, transform_into="csp_space", norm_trace=False)
    Xt = csp.fit(epochs_data, epochs.events[:, 2]).transform(epochs_data)
    feature_shape = [len(epochs_data), n_components, epochs_data.shape[2]]
    assert_array_equal(Xt.shape, feature_shape)

    # Check mixing matrix on simulated data
    y = np.array([100] * 50 + [1] * 50)
    X, A = simulate_data(y)

    for cov_est in ["concat", "epoch"]:
        # fit csp
        csp = CSP(n_components=1, cov_est=cov_est, norm_trace=False)
        csp.fit(X, y)

        # check the first pattern match the mixing matrix
        # the sign might change
        corr = np.abs(np.corrcoef(csp.patterns_[0, :].T, A[:, 0])[0, 1])
        assert np.abs(corr) > 0.99

        # check output
        out = csp.transform(X)
        corr = np.abs(np.corrcoef(out[:, 0], y)[0, 1])
        assert np.abs(corr) > 0.95


# Even the "reg is None and rank is None" case should pass now thanks to the
# do_compute_rank
@pytest.mark.parametrize("ch_type", ("mag", "eeg", ("mag", "eeg")))
@pytest.mark.parametrize("rank", (None, "full", "correct"))
@pytest.mark.parametrize("reg", [None, 0.001, "oas"])
def test_regularized_csp(ch_type, rank, reg):
    """Test Common Spatial Patterns algorithm using regularized covariance."""
    raw = io.read_raw_fif(raw_fname).pick(ch_type, exclude="bads").load_data()
    n_orig = len(raw.ch_names)
    ch_decim = 2
    raw.pick_channels(raw.ch_names[::ch_decim])
    raw.info.normalize_proj()
    if "eeg" in ch_type:
        raw.set_eeg_reference(projection=True)
        # TODO: for some reason we need to add a second EEG projector in order to get
        # the non-semidefinite error for EEG data. Hopefully this won't make much
        # difference in practice given our default is rank=None and regularization
        # is easy to use.
        raw.add_proj(compute_proj_raw(raw, n_eeg=1, n_mag=0, n_grad=0, n_jobs=1))
    n_eig = len(raw.ch_names) - len(raw.info["projs"])
    n_ch = n_orig // ch_decim
    if ch_type == "eeg":
        assert n_eig == n_ch - 2
    elif ch_type == "mag":
        assert n_eig == n_ch - 3
    else:
        assert n_eig == n_ch - 5
    if rank == "correct":
        if isinstance(ch_type, str):
            rank = {ch_type: n_eig}
        else:
            assert ch_type == ("mag", "eeg")
            rank = dict(
                mag=102 // ch_decim - 3,
                eeg=60 // ch_decim - 2,
            )
    else:
        assert rank is None or rank == "full", rank
    if rank == "full":
        n_eig = n_ch
    raw.filter(2, 40).apply_proj()
    events = read_events(event_name)
    # map make left and right events the same
    events[events[:, 2] == 2, 2] = 1
    events[events[:, 2] == 4, 2] = 3
    epochs = Epochs(raw, events, event_id, tmin, tmax, decim=5, preload=True)
    epochs.equalize_event_counts()
    assert 25 < len(epochs) < 30
    epochs_data = epochs.get_data(copy=False)
    n_channels = epochs_data.shape[1]
    assert n_channels == n_ch
    n_components = 3

    sc = Scaler(epochs.info)
    epochs_data_orig = epochs_data.copy()
    epochs_data = sc.fit_transform(epochs_data)
    csp = CSP(n_components=n_components, reg=reg, norm_trace=False, rank=rank)
    if rank == "full" and reg is None:
        with pytest.raises(np.linalg.LinAlgError, match="leading minor"):
            csp.fit(epochs_data, epochs.events[:, -1])
        return
    with catch_logging(verbose=True) as log:
        X = csp.fit_transform(epochs_data, epochs.events[:, -1])
    log = log.getvalue()
    assert "Setting small MAG" not in log
    if rank != "full":
        assert "Setting small data eigen" in log
    else:
        assert "Setting small data eigen" not in log
    if rank is None:
        assert "Computing rank from data" in log
        assert " mag: rank" not in log.lower()
        assert " data: rank" in log
        assert "rank (mag)" not in log.lower()
        assert "rank (data)" in log
    elif rank != "full":  # if rank is passed no computation is done
        assert "Computing rank" not in log
        assert ": rank" not in log
        assert "rank (" not in log
    assert "reducing mag" not in log.lower()
    assert f"Reducing data rank from {n_channels} " in log
    y = epochs.events[:, -1]
    assert csp.filters_.shape == (n_eig, n_channels)
    assert csp.patterns_.shape == (n_eig, n_channels)
    assert_array_almost_equal(csp.fit(epochs_data, y).transform(epochs_data), X)

    # test init exception
    pytest.raises(ValueError, csp.fit, epochs_data, np.zeros_like(epochs.events))
    pytest.raises(ValueError, csp.fit, "foo", y)
    pytest.raises(ValueError, csp.transform, "foo")

    csp.n_components = n_components
    sources = csp.transform(epochs_data)
    assert sources.shape[1] == n_components

    cv = StratifiedKFold(5)
    clf = make_pipeline(
        sc,
        csp,
        LinearModel(LogisticRegression(solver="liblinear")),
    )
    score = cross_val_score(clf, epochs_data_orig, y, cv=cv, scoring="roc_auc").mean()
    assert 0.75 <= score <= 1.0

    # Test get_coef on CSP
    clf.fit(epochs_data_orig, y)
    coef = csp.patterns_[:n_components]
    assert coef.shape == (n_components, n_channels), coef.shape
    coef = sc.inverse_transform(coef.T[np.newaxis])[0]
    assert coef.shape == (len(epochs.ch_names), n_components), coef.shape
    coef_mne = get_coef(clf, "patterns_", inverse_transform=True, verbose="debug")
    assert coef.shape == coef_mne.shape
    coef_mne /= np.linalg.norm(coef_mne, axis=0)
    coef /= np.linalg.norm(coef, axis=0)
    coef *= np.sign(np.sum(coef_mne * coef, axis=0))
    assert_allclose(coef_mne, coef)


def test_csp_pipeline():
    """Test if CSP works in a pipeline."""
    csp = CSP(reg=1, norm_trace=False)
    svc = SVC()
    pipe = Pipeline([("CSP", csp), ("SVC", svc)])
    pipe.set_params(CSP__reg=0.2)
    assert pipe.get_params()["CSP__reg"] == 0.2


def test_ajd():
    """Test approximate joint diagonalization."""
    # The implementation should obtain the same
    # results as the Matlab implementation by Pham Dinh-Tuan.
    # Generate a set of cavariances matrices for test purpose
    n_times, n_channels = 10, 3
    seed = np.random.RandomState(0)
    diags = 2.0 + 0.1 * seed.randn(n_times, n_channels)
    A = 2 * seed.rand(n_channels, n_channels) - 1
    A /= np.atleast_2d(np.sqrt(np.sum(A**2, 1))).T
    covmats = np.empty((n_times, n_channels, n_channels))
    for i in range(n_times):
        covmats[i] = np.dot(np.dot(A, np.diag(diags[i])), A.T)
    V, D = _ajd_pham(covmats)
    # Results obtained with original matlab implementation
    V_matlab = [
        [-3.507280775058041, -5.498189967306344, 7.720624541198574],
        [0.694689013234610, 0.775690358505945, -1.162043086446043],
        [-0.592603135588066, -0.598996925696260, 1.009550086271192],
    ]
    assert_array_almost_equal(V, V_matlab)


def test_spoc():
    """Test SPoC."""
    X = np.random.randn(10, 10, 20)
    y = np.random.randn(10)

    spoc = SPoC(n_components=4)
    spoc.fit(X, y)
    Xt = spoc.transform(X)
    assert_array_equal(Xt.shape, [10, 4])
    spoc = SPoC(n_components=4, transform_into="csp_space")
    spoc.fit(X, y)
    Xt = spoc.transform(X)
    assert_array_equal(Xt.shape, [10, 4, 20])
    assert_array_equal(spoc.filters_.shape, [10, 10])
    assert_array_equal(spoc.patterns_.shape, [10, 10])

    # check y
    pytest.raises(ValueError, spoc.fit, X, y * 0)

    # Check that doesn't take CSP-spcific input
    pytest.raises(TypeError, SPoC, cov_est="epoch")

    # Check mixing matrix on simulated data
    rs = np.random.RandomState(42)
    y = rs.rand(100) * 50 + 1
    X, A = simulate_data(y)

    # fit spoc
    spoc = SPoC(n_components=1)
    spoc.fit(X, y)

    # check the first patterns match the mixing matrix
    corr = np.abs(np.corrcoef(spoc.patterns_[0, :].T, A[:, 0])[0, 1])
    assert np.abs(corr) > 0.99

    # check output
    out = spoc.transform(X)
    corr = np.abs(np.corrcoef(out[:, 0], y)[0, 1])
    assert np.abs(corr) > 0.85


def test_csp_twoclass_symmetry():
    """Test that CSP is symmetric when swapping classes."""
    x, y = deterministic_toy_data(["class_a", "class_b"])
    csp = CSP(norm_trace=False, transform_into="average_power", log=True)
    log_power = csp.fit_transform(x, y)
    log_power_ratio_ab = log_power[0] - log_power[1]

    x, y = deterministic_toy_data(["class_b", "class_a"])
    csp = CSP(norm_trace=False, transform_into="average_power", log=True)
    log_power = csp.fit_transform(x, y)
    log_power_ratio_ba = log_power[0] - log_power[1]

    assert_array_almost_equal(log_power_ratio_ab, log_power_ratio_ba)


def test_csp_component_ordering():
    """Test that CSP component ordering works as expected."""
    x, y = deterministic_toy_data(["class_a", "class_b"])

    csp = CSP(component_order="invalid")
    with pytest.raises(ValueError, match="Invalid value"):
        csp.fit(x, y)

    # component_order='alternate' only works with two classes
    csp = CSP(component_order="alternate")
    with pytest.raises(ValueError):
        csp.fit(np.zeros((3, 0, 0)), ["a", "b", "c"])

    p_alt = CSP(component_order="alternate").fit(x, y).patterns_
    p_mut = CSP(component_order="mutual_info").fit(x, y).patterns_

    # This permutation of p_alt and p_mut is explained by the particular
    # eigenvalues of the toy data: [0.06, 0.1,   0.5,  0.8].
    # p_alt arranges them to [0.8, 0.06, 0.5, 0.1]
    # p_mut arranges them to [0.06, 0.1, 0.8, 0.5]
    assert_array_almost_equal(p_alt, p_mut[[2, 0, 3, 1]])


@pytest.mark.filterwarnings("ignore:.*Only one sample available.*")
@parametrize_with_checks([CSP(), SPoC()])
def test_sklearn_compliance(estimator, check):
    """Test compliance with sklearn."""
    check(estimator)
