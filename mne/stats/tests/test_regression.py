# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from scipy.signal.windows import hann

import mne
from mne import read_source_estimate
from mne.datasets import testing
from mne.io import RawArray
from mne.stats.regression import _fit_lm, linear_regression, linear_regression_raw

data_path = testing.data_path(download=False)
stc_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc-meg-lh.stc"
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
event_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw-eve.fif"


@testing.requires_testing_data
def test_regression():
    """Test Ordinary Least Squares Regression."""
    tmin, tmax = -0.2, 0.5
    event_id = dict(aud_l=1, aud_r=2)

    # Setup for reading the raw data
    raw = mne.io.read_raw_fif(raw_fname)
    events = mne.read_events(event_fname)[:10]
    epochs = mne.Epochs(
        raw, events, event_id, tmin, tmax, proj=True, baseline=(None, 0)
    )
    picks = np.arange(len(epochs.ch_names))
    evoked = epochs.average(picks=picks)
    design_matrix = epochs.events[:, 1:].astype(np.float64)
    # makes the intercept
    design_matrix[:, 0] = 1
    # creates contrast: aud_l=0, aud_r=1
    design_matrix[:, 1] -= 1
    with pytest.warns(RuntimeWarning, match="non-data"):
        lm = linear_regression(epochs, design_matrix, ["intercept", "aud"])

    for predictor, parameters in lm.items():
        for value in parameters:
            assert_equal(value.data.shape, evoked.data.shape)

    pytest.raises(ValueError, linear_regression, [epochs, epochs], design_matrix)

    stc = read_source_estimate(stc_fname).crop(0, 0.02)
    stc_list = [stc, stc, stc]
    stc_gen = (s for s in stc_list)
    with np.errstate(invalid="ignore"):  # divide by zero
        lm1 = linear_regression(stc_list, design_matrix[: len(stc_list)])
    lm2 = linear_regression(stc_gen, design_matrix[: len(stc_list)])
    for val in lm2.values():
        # all p values are 0 < p <= 1 to start, but get stored in float32
        # data, so can actually be truncated to 0. Thus the mlog10_p_val
        # actually maintains better precision for tiny p-values.
        assert np.isfinite(val.p_val.data).all()
        assert (val.p_val.data <= 1).all()
        assert (val.p_val.data >= 0).all()
        # all -log10(p) are non-negative
        assert np.isfinite(val.mlog10_p_val.data).all()
        assert (val.mlog10_p_val.data >= 0).all()
        assert (val.mlog10_p_val.data >= 0).all()

    for k in lm1:
        for v1, v2 in zip(lm1[k], lm2[k]):
            assert_array_equal(v1.data, v2.data)

    # Smoke test for fitting on epochs
    epochs.load_data()
    with pytest.warns(RuntimeWarning, match="non-data"):
        linear_regression(epochs, design_matrix)
    linear_regression(epochs.copy().pick("eeg"), design_matrix)


@pytest.mark.parametrize("sign", (1, -1))
def test_fit_lm_perfect_fit(sign):
    """Test that a perfect fit gives an infinite t-value of the correct sign."""
    design_matrix = np.eye(4)[:, :2]
    data = sign * 2.0 * design_matrix[:, :1] * np.ones((1, 3))
    beta, stderr, t_val, p_val, _ = _fit_lm(data, design_matrix, ["x0", "x1"])
    assert_allclose(beta["x0"], sign * 2.0)
    assert_array_equal(stderr["x0"], 0.0)
    assert_array_equal(t_val["x0"], sign * np.inf)
    assert (p_val["x0"] < 1e-300).all()
    # a coefficient of exactly zero is not significant
    assert_array_equal(t_val["x1"], 0.0)
    assert_array_equal(p_val["x1"], 1.0)


@testing.requires_testing_data
def test_continuous_regression_no_overlap():
    """Test regression without overlap correction, on real data."""
    tmin, tmax = -0.1, 0.5

    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.apply_proj()

    # a sampling of frequency where rounding and truncation yield
    # different results checks conversion from samples to times is
    # consistent across Epochs and linear_regression_raw
    with raw.info._unlock():
        raw.info["sfreq"] = 128

    events = mne.read_events(event_fname)
    event_id = dict(audio_l=1, audio_r=2)

    raw = raw.pick(raw.ch_names[:2])

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, reject=None)

    revokeds = linear_regression_raw(
        raw, events, event_id, tmin=tmin, tmax=tmax, reject=None
    )

    # Check that evokeds and revokeds are nearly equivalent
    for cond in event_id.keys():
        assert_allclose(revokeds[cond].data, epochs[cond].average().data, rtol=1e-15)

    # Test events that will lead to "duplicate" errors
    old_latency = events[1, 0]
    events[1, 0] = events[0, 0]
    pytest.raises(ValueError, linear_regression_raw, raw, events, event_id, tmin, tmax)

    events[1, 0] = old_latency
    events[:, 0] = range(len(events))
    pytest.raises(
        ValueError, linear_regression_raw, raw, events, event_id, tmin, tmax, decim=2
    )


@testing.requires_testing_data
def test_continuous_regression_with_overlap():
    """Test regression with overlap correction."""
    pytest.importorskip("sklearn")
    signal = np.zeros(100000)
    times = [1000, 2500, 3000, 5000, 5250, 7000, 7250, 8000]
    events = np.zeros((len(times), 3), int)
    events[:, 2] = 1
    events[:, 0] = times
    signal[events[:, 0]] = 1.0
    effect = hann(101)
    signal = np.convolve(signal, effect)[: len(signal)]
    raw = RawArray(signal[np.newaxis, :], mne.create_info(1, 100, "eeg"))

    assert_allclose(
        effect, linear_regression_raw(raw, events, {1: 1}, tmin=0)[1].data.flatten()
    )

    # test that sklearn solvers can be used
    from sklearn.linear_model import ridge_regression

    def solver(X, y):
        # Newer scikit-learn returns 1D array for ridge_regression, so ensure
        # 2D output
        return np.atleast_2d(
            ridge_regression(X, y, alpha=0.0, solver="cholesky", random_state=0)
        )

    assert_allclose(
        effect,
        linear_regression_raw(raw, events, tmin=0, solver=solver)["1"].data.flatten(),
    )

    # test bad solvers
    def solT(X, y):
        return ridge_regression(X, y, alpha=0.0, solver="cholesky", random_state=0).T

    pytest.raises(ValueError, linear_regression_raw, raw, events, solver=solT)
    pytest.raises(ValueError, linear_regression_raw, raw, events, solver="err")
    pytest.raises(TypeError, linear_regression_raw, raw, events, solver=0)


def test_continuous_regression_flat():
    """Test flat-only rejection without requiring an external dataset."""
    # rejection based on flat alone (reject=None) must not be ignored
    data = np.random.default_rng(0).standard_normal((1, 3000)) * 1e-5
    data[0, 1000:1300] = 0.0
    raw = RawArray(data, mne.create_info(1, 100, "eeg"))
    events = np.array([[sample, 0, 1] for sample in range(100, 2900, 200)])
    kwargs = dict(event_id={"a": 1}, tmin=0, tmax=0.5)
    evoked_both = linear_regression_raw(
        raw, events, reject=dict(eeg=1.0), flat=dict(eeg=1e-7), **kwargs
    )["a"]
    evoked_flat = linear_regression_raw(raw, events, flat=dict(eeg=1e-7), **kwargs)["a"]
    assert_allclose(evoked_both.data, evoked_flat.data)
