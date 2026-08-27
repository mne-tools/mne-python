"""Tests for epochs whose trials have different durations."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mne import EpochsArray, create_info
from mne.epochs import (
    _VARIABLE_FALLBACK,
    _VARIABLE_NEEDS_POLICY,
    _VARIABLE_NOT_IMPLEMENTED,
)

SFREQ = 100.0
CH_NAMES = ["a", "b", "c"]


def _events(n_epochs):
    """Return a simple event array."""
    return np.c_[
        np.arange(n_epochs) * 500 + 200,
        np.zeros(n_epochs, int),
        np.ones(n_epochs, int),
    ]


def _make(tmin, tmax, seed=0):
    """Build variable-duration epochs from per-event bounds."""
    rng = np.random.default_rng(seed)
    tmin = np.asarray(tmin, dtype=float)
    tmax = np.asarray(tmax, dtype=float)
    data = [
        rng.standard_normal((len(CH_NAMES), round((b - a) * SFREQ) + 1)) * 1e-6
        for a, b in zip(tmin, tmax)
    ]
    info = create_info(CH_NAMES, SFREQ, "eeg")
    return EpochsArray(
        data,
        info,
        events=_events(len(data)),
        tmin=tmin,
        baseline=None,
        verbose=False,
    )


@pytest.fixture
def variable():
    """Four epochs sharing tmin with different tmax."""
    return _make(np.full(4, -0.2), [0.5, 0.9, 0.7, 0.6])


# -- the scalar path must be untouched ------------------------------------
def test_scalar_path_unchanged():
    """Test that fixed-duration epochs behave exactly as before."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((5, len(CH_NAMES), 71)) * 1e-6
    info = create_info(CH_NAMES, SFREQ, "eeg")
    epochs = EpochsArray(data, info, events=_events(5), tmin=-0.2, verbose=False)

    assert not epochs.variable_duration
    assert isinstance(epochs.tmin, float)
    assert isinstance(epochs.tmax, float)
    assert_allclose(epochs.tmin, -0.2)
    assert epochs.get_data().shape == (5, len(CH_NAMES), 71)
    assert epochs.average().data.shape == (len(CH_NAMES), 71)
    # durations is new but must be defined for the fixed case too
    assert_allclose(epochs.durations, np.full(5, 0.7))


def test_equal_bounds_arrays_match_scalar():
    """Test that per-event bounds that happen to be equal match the scalar path."""
    rng = np.random.default_rng(3)
    data = rng.standard_normal((4, len(CH_NAMES), 71)) * 1e-6
    info = create_info(CH_NAMES, SFREQ, "eeg")

    scalar = EpochsArray(data, info, events=_events(4), tmin=-0.2, verbose=False)
    arrays = EpochsArray(
        list(data), info, events=_events(4), tmin=np.full(4, -0.2), verbose=False
    )

    # a list of equal-length arrays is not ragged, so this stays the scalar path
    assert not arrays.variable_duration
    assert_allclose(arrays.times, scalar.times)
    assert_allclose(arrays.get_data(), scalar.get_data())


# -- construction ---------------------------------------------------------
def test_bounds_and_durations(variable):
    """Test the per-epoch bounds derived from the data."""
    assert variable.variable_duration
    assert_allclose(variable.tmin, np.full(4, -0.2))
    assert_allclose(variable.tmax, [0.5, 0.9, 0.7, 0.6], atol=1e-12)
    assert_allclose(variable.durations, [0.7, 1.1, 0.9, 0.8], atol=1e-12)


def test_per_epoch_time_vectors(variable):
    """Test that each epoch reports its own axis, since none is shared."""
    per_epoch = variable.get_times()
    assert len(per_epoch) == 4
    for ii, epoch_times in enumerate(per_epoch):
        assert_allclose(epoch_times[0], variable.tmin[ii])
        assert_allclose(epoch_times[-1], variable.tmax[ii], atol=1e-12)
        assert len(epoch_times) == variable.get_data()[ii].shape[-1]


def test_get_data_returns_one_array_per_epoch(variable):
    """Test that data comes back per epoch, since no common length exists."""
    data = variable.get_data()
    assert isinstance(data, list)
    lengths = [epoch.shape[1] for epoch in data]
    assert lengths == [71, 111, 91, 81]
    assert all(epoch.shape[0] == len(CH_NAMES) for epoch in data)

    picked = variable.get_data(picks=["a", "c"])
    assert all(epoch.shape[0] == 2 for epoch in picked)


def test_only_time_may_vary():
    """Test that a varying channel count is rejected."""
    info = create_info(CH_NAMES, SFREQ, "eeg")
    data = [np.zeros((3, 71)), np.zeros((2, 111))]
    with pytest.raises(ValueError, match="same number of channels"):
        EpochsArray(data, info, events=_events(2), tmin=np.zeros(2), verbose=False)


def test_bounds_validation():
    """Test the checks on per-event bounds."""
    from mne.epochs import _check_variable_bounds

    tmin, tmax, variable = _check_variable_bounds(-0.2, 0.5, 3)
    assert not variable
    assert tmin == -0.2

    tmin, tmax, variable = _check_variable_bounds(-0.2, np.array([0.5, 0.9, 0.7]), 3)
    assert variable
    assert_array_equal(tmin, np.full(3, -0.2))

    with pytest.raises(ValueError, match="entries but there are"):
        _check_variable_bounds(np.zeros(2), 0.5, 3)
    with pytest.raises(ValueError, match="less than or equal to tmax"):
        _check_variable_bounds(np.array([0.5, 0.0]), np.array([0.1, 1.0]), 2)
    with pytest.raises(ValueError, match="must be finite"):
        _check_variable_bounds(np.array([np.nan, 0.0]), np.array([1.0, 1.0]), 2)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(baseline=(None, 0)), "Baseline correction is not implemented"),
        (dict(reject_tmin=-0.1), "reject_tmin is not implemented"),
        (dict(reject_tmax=0.1), "reject_tmax is not implemented"),
    ],
)
def test_unsupported_options_raise(kwargs, match):
    """Test that options outside this implementation are refused clearly."""
    rng = np.random.default_rng(0)
    tmin = np.full(2, -0.2)
    data = [rng.standard_normal((len(CH_NAMES), n)) for n in (71, 111)]
    info = create_info(CH_NAMES, SFREQ, "eeg")
    with pytest.raises(NotImplementedError, match=match):
        EpochsArray(data, info, events=_events(2), tmin=tmin, verbose=False, **kwargs)


# -- as_fixed -------------------------------------------------------------
def test_as_fixed_spans_union_and_reports_support(variable):
    """Test padding to a common window and the contributor count it implies."""
    fixed, n_contributing = variable.as_fixed()

    assert not fixed.variable_duration
    n_union = max(epoch.shape[-1] for epoch in variable.get_data())
    assert fixed.get_data().shape == (4, len(CH_NAMES), n_union)
    assert_allclose(fixed.times[0], variable.tmin.min())
    assert_allclose(fixed.times[-1], variable.tmax.max())

    # every epoch covers the start; only the longest reaches the end
    assert n_contributing[0] == 4
    assert n_contributing[-1] == 1
    assert n_contributing.min() < n_contributing.max()
    # support never grows once epochs start dropping out
    peak = int(np.argmax(n_contributing))
    assert np.all(np.diff(n_contributing[peak:]) <= 0)


def test_as_fixed_preserves_data_and_pads_the_rest(variable):
    """Test that real samples survive and the remainder is marked."""
    fixed, _ = variable.as_fixed()
    dense = fixed.get_data()
    for ii, epoch in enumerate(variable.get_data()):
        n = epoch.shape[1]
        assert_allclose(dense[ii, :, :n], epoch)
        assert np.isnan(dense[ii, :, n:]).all()


def test_as_fixed_on_fixed_epochs_is_a_copy():
    """Test that as_fixed is defined, and trivial, for fixed-duration epochs."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((3, len(CH_NAMES), 71)) * 1e-6
    info = create_info(CH_NAMES, SFREQ, "eeg")
    epochs = EpochsArray(data, info, events=_events(3), tmin=-0.2, verbose=False)

    fixed, n_contributing = epochs.as_fixed()
    assert_allclose(fixed.get_data(), epochs.get_data())
    assert_array_equal(n_contributing, np.full(len(epochs.times), 3))


# -- dispatch --------------------------------------------------------------
@pytest.mark.parametrize("meth", sorted(_VARIABLE_NEEDS_POLICY))
def test_reductions_ask_for_a_policy(variable, meth):
    """Test that combining epochs across a time axis they lack is refused.

    Padding first and reducing afterwards is not a slower answer, it is a
    different one: one short epoch turns a whole time point into NaN, and the
    scalar ``nave`` keeps reporting the full count.
    """
    with pytest.raises(NotImplementedError, match="explicit policy"):
        result = getattr(variable, meth)()
        list(result)  # iter_evoked is a generator


def test_policy_message_names_the_varying_count(variable):
    """Test that the refusal explains itself rather than just declining."""
    with pytest.raises(NotImplementedError) as excinfo:
        variable.average()
    message = str(excinfo.value)
    assert "varies across the window" in message
    assert "as_fixed" in message


def test_compute_tfr_does_not_silently_pad(variable):
    """Test that the transform is not quietly given padded data.

    Padding before a time-frequency transform is the opposite of the order this
    work argues for, which is to transform at native duration and warp the
    result. Doing it silently inside ``compute_tfr`` would ship the thing being
    argued against.
    """
    with pytest.raises(NotImplementedError, match="explicit policy"):
        variable.compute_tfr("morlet", freqs=np.arange(10.0, 20.0, 2.0), n_cycles=2)


@pytest.mark.parametrize("meth", sorted(_VARIABLE_NOT_IMPLEMENTED))
def test_per_trial_methods_raise_until_implemented(variable, meth):
    """Test that per-trial work refuses rather than running on a padded copy."""
    with pytest.raises(NotImplementedError, match="not implemented"):
        getattr(variable, meth)()


@pytest.mark.parametrize("meth", sorted(_VARIABLE_FALLBACK))
def test_display_methods_warn_and_fall_back(variable, meth):
    """Test that the remaining inspection method degrades rather than refuses."""
    if meth == "to_data_frame":
        pytest.importorskip("pandas")
    with pytest.warns(RuntimeWarning, match="ran on as_fixed"):
        assert getattr(variable, meth)() is not None


def test_plot_is_not_a_fallback():
    """Test that browsing is native, not padded (see mne/viz/tests/test_epochs)."""
    assert "plot" not in _VARIABLE_FALLBACK
    assert "plot" not in _VARIABLE_NEEDS_POLICY
    assert "plot" not in _VARIABLE_NOT_IMPLEMENTED


# -- operations that stay native -------------------------------------------
def test_pick_keeps_durations(variable):
    """Test that channel selection leaves the time axis alone."""
    before = variable.durations.copy()
    picked = variable.copy().pick(["a", "c"])
    assert picked.ch_names == ["a", "c"]
    assert_allclose(picked.durations, before)
    for epoch in picked.get_data():
        assert epoch.shape[0] == 2


def test_getitem_keeps_per_epoch_bounds(variable):
    """Test that indexing carries the bounds with the epochs."""
    subset = variable[[0, 2]]
    assert len(subset) == 2
    assert_allclose(subset.durations, variable.durations[[0, 2]])
    for got, want in zip(subset.get_data(), [variable.get_data()[i] for i in (0, 2)]):
        assert_array_equal(got, want)


def test_drop_keeps_per_epoch_bounds(variable):
    """Test that dropping an epoch drops its bounds too."""
    kept = variable.copy().drop([1])
    assert len(kept) == 3
    assert_allclose(kept.durations, variable.durations[[0, 2, 3]])


def test_shift_time_moves_bounds_not_samples(variable):
    """Test that shifting the origin does not resample anything."""
    before_lengths = [epoch.shape[1] for epoch in variable.get_data()]
    before_durations = variable.durations.copy()
    shifted = variable.copy().shift_time(0.1)
    assert_allclose(shifted.tmin, variable.tmin + 0.1)
    assert_allclose(shifted.durations, before_durations)
    assert [epoch.shape[1] for epoch in shifted.get_data()] == before_lengths


# -- the time axis ---------------------------------------------------------
def test_times_refuses_to_invent_a_shared_axis(variable):
    """Test that ``times`` raises rather than handing back the union.

    ``len(epochs.times) == data.shape[-1]`` has always held. Returning the union
    would leave it false while looking ordinary.
    """
    with pytest.raises(RuntimeError, match="no time axis they share"):
        variable.times
    with pytest.raises(RuntimeError, match="get_times"):
        variable.times


def test_union_axis_is_reachable_through_as_fixed(variable):
    """Test that the padded common axis is available when asked for."""
    fixed, _ = variable.as_fixed()
    assert_allclose(fixed.times[0], variable.tmin.min())
    assert_allclose(fixed.times[-1], variable.tmax.max())
    assert len(fixed.times) == fixed.get_data().shape[-1]


def test_fixed_epochs_still_have_times():
    """Test that the scalar path is untouched by any of this."""
    rng = np.random.default_rng(2)
    info = create_info(CH_NAMES, SFREQ, "eeg")
    epochs = EpochsArray(
        rng.standard_normal((3, len(CH_NAMES), 71)) * 1e-6,
        info,
        events=_events(3),
        tmin=-0.2,
        verbose=False,
    )
    assert len(epochs.times) == 71
    assert epochs.average().data.shape == (len(CH_NAMES), 71)


def test_nothing_reaches_the_user_as_an_internal_error(variable):
    """Test that no public method leaks a NumPy error about lists."""
    import warnings

    names = (
        sorted(_VARIABLE_FALLBACK)
        + sorted(_VARIABLE_NEEDS_POLICY)
        + sorted(_VARIABLE_NOT_IMPLEMENTED)
    )
    for name in names:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                getattr(variable.copy(), name)()
            except (NotImplementedError, RuntimeError):
                pass
            except TypeError as exc:
                assert "argument" in str(exc), f"{name}: {exc}"
            except (AttributeError, IndexError) as exc:
                raise AssertionError(f"{name} leaked an internal error: {exc}")
            except Exception:
                pass


# -- construction from Raw -------------------------------------------------
def _raw(n_seconds=30.0, sfreq=SFREQ):
    """Return a small continuous recording."""
    from mne.io import RawArray

    rng = np.random.default_rng(3)
    info = create_info(CH_NAMES, sfreq, "eeg")
    return RawArray(
        rng.standard_normal((len(CH_NAMES), int(n_seconds * sfreq))) * 1e-6,
        info,
        verbose=False,
    )


def test_from_raw_preserves_each_slice():
    """Test that epochs read from Raw match the samples they came from."""
    from mne import Epochs

    raw = _raw()
    onsets = np.array([100, 700, 1400, 2100])
    events = np.c_[onsets, np.zeros(4, int), np.ones(4, int)]
    tmax = np.array([0.5, 1.9, 0.9, 2.4])

    epochs = Epochs(
        raw,
        events,
        {"a": 1},
        tmin=np.zeros(4),
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    assert epochs.variable_duration
    assert len(epochs) == 4
    # +1 because both endpoints are inclusive
    want_n = np.round(tmax * SFREQ).astype(int) + 1
    data = epochs.get_data()
    assert [d.shape[-1] for d in data] == list(want_n)

    raw_data = raw.get_data()
    for i, start in enumerate(onsets):
        assert_array_equal(data[i], raw_data[:, start : start + want_n[i]])


def test_from_raw_load_data_leaves_raw_times_alone():
    """Test that loading from Raw does not ask a ragged object for one axis."""
    from mne import Epochs

    raw = _raw()
    events = np.c_[[100, 900], [0, 0], [1, 1]]
    # load_data() used to raise here from `self._raw_times = self.times`
    epochs = Epochs(
        raw,
        events,
        {"a": 1},
        tmin=np.zeros(2),
        tmax=np.array([0.5, 2.0]),
        baseline=None,
        preload=True,
        verbose=False,
    )
    assert epochs.preload
    # _raw_times spans the union, which is what as_fixed() lays epochs onto
    fixed, _ = epochs.as_fixed()
    assert len(epochs._raw_times) == fixed.get_data().shape[-1]


def test_from_raw_requires_preload():
    """Test that lazy reading is refused with a reason, not an internal error."""
    from mne import Epochs

    raw = _raw()
    events = np.c_[[100, 900], [0, 0], [1, 1]]
    with pytest.raises(NotImplementedError, match="must be preloaded"):
        Epochs(
            raw,
            events,
            {"a": 1},
            tmin=np.zeros(2),
            tmax=np.array([0.5, 2.0]),
            baseline=None,
            preload=False,
            verbose=False,
        )


def test_from_raw_scalar_bounds_still_scalar():
    """Test that equal per-event bounds do not switch on the ragged path."""
    from mne import Epochs

    raw = _raw()
    events = np.c_[[100, 900], [0, 0], [1, 1]]
    epochs = Epochs(
        raw,
        events,
        {"a": 1},
        tmin=np.zeros(2),
        tmax=np.full(2, 0.5),
        baseline=None,
        preload=True,
        verbose=False,
    )
    assert not epochs.variable_duration
    assert isinstance(epochs.tmin, float)
    assert epochs.get_data().shape == (2, len(CH_NAMES), 51)


@pytest.mark.parametrize(
    "item", [slice(None, 2), slice(1, None), slice(None, None, 2), slice(None)]
)
def test_getitem_slice_selects_epochs_not_the_slice(variable, item):
    """Test that slicing subsets the epochs rather than wrapping the slice."""
    want = np.arange(len(variable))[item]
    subset = variable[item]

    assert len(subset) == len(want)
    # a slice used to survive into the data list as a single nested element
    assert all(isinstance(d, np.ndarray) and d.ndim == 2 for d in subset.get_data())
    assert_allclose(subset.durations, variable.durations[want])
    for got, idx in zip(subset.get_data(), want):
        assert_array_equal(got, variable.get_data()[idx])


def test_apply_function_refuses(variable):
    """Test that apply_function refuses instead of indexing a list with a tuple."""
    with pytest.raises(NotImplementedError, match="not implemented"):
        variable.apply_function(lambda x: x * 2)


def test_pick_does_not_reach_back_into_the_parent(variable):
    """Test that picking replaces one object's epochs and no other's."""
    before = [epoch.shape for epoch in variable.get_data()]

    # _pick_drop_channels replaces the list contents in place, so anything
    # sharing that list would be picked too
    variable.copy().pick(["a", "c"])
    assert [epoch.shape for epoch in variable.get_data()] == before

    subset = variable[:2]
    assert subset._data is not variable._data
    subset.pick(["a"])
    assert [epoch.shape for epoch in variable.get_data()] == before
    assert all(epoch.shape[0] == 1 for epoch in subset.get_data())
