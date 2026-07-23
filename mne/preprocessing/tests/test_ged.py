"""Tests for generalized eigendecomposition preprocessing."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.preprocessing.ged import GED


def _make_data(n_channels=4, n_times=2000, random_state=0):
    """Create foreground/background data with a dominant foreground component."""
    rng = np.random.RandomState(random_state)

    background = rng.randn(n_channels, n_times)

    foreground = background.copy()

    artifact = 10 * rng.randn(n_times)

    foreground[0] += artifact

    foreground[1] += 0.5 * artifact

    return foreground, background


def test_ged_fit():
    """Test fitting GED on foreground/background data."""
    foreground, background = _make_data()

    ged = GED(n_components=2).fit(foreground, background)

    assert ged.filters_.shape == (2, 4)

    assert ged.patterns_.shape == (2, 4)

    assert ged.eigenvalues_.shape == (2,)

    assert ged.foreground_cov_.shape == (4, 4)

    assert ged.background_cov_.shape == (4, 4)

    assert ged.background_cov_reg_.shape == (4, 4)

    assert ged.n_components_ == 2

    assert np.all(np.diff(ged.eigenvalues_) <= 0)


def test_ged_transform():
    """Test GED transform."""
    foreground, background = _make_data()

    ged = GED(n_components=2).fit(foreground, background)

    transformed = ged.transform(foreground)

    assert transformed.shape == (2, foreground.shape[1])


def test_ged_transform_epochs_like_array():
    """Test GED transform on epochs-like arrays."""
    foreground, background = _make_data()

    epochs_data = np.stack([foreground, foreground], axis=0)

    ged = GED(n_components=2).fit(foreground, background)

    transformed = ged.transform(epochs_data)

    assert transformed.shape == (2, 2, foreground.shape[1])


def test_ged_apply():
    """Test GED reconstruction after excluding components."""
    foreground, background = _make_data()

    ged = GED(n_components=4).fit(foreground, background)

    cleaned = ged.apply(foreground, exclude=[0])

    assert cleaned.shape == foreground.shape

    assert np.isfinite(cleaned).all()


def test_ged_normalize():
    """Test GED with channel-wise normalization."""
    foreground, background = _make_data()

    ged = GED(n_components=2, normalize=True).fit(foreground, background)

    assert ged.filters_.shape == (2, 4)


def test_ged_invalid_n_components():
    """Test invalid n_components."""
    foreground, background = _make_data()

    with pytest.raises(ValueError, match="between 1 and n_channels"):
        GED(n_components=5).fit(foreground, background)


def test_ged_invalid_reg():
    """Test invalid regularization."""
    foreground, background = _make_data()

    with pytest.raises(ValueError, match="reg must be between 0 and 1"):
        GED(reg=2).fit(foreground, background)


def test_ged_channel_mismatch():
    """Test channel mismatch between foreground and background."""
    foreground, background = _make_data()

    with pytest.raises(ValueError, match="same number of channels"):
        GED().fit(foreground[:3], background)


def test_ged_transform_before_fit():
    """Test transform before fit."""
    foreground, _ = _make_data()

    with pytest.raises(RuntimeError, match="has not been fitted"):
        GED().transform(foreground)


def test_ged_apply_invalid_exclude():
    """Test invalid exclude indices."""
    foreground, background = _make_data()

    ged = GED(n_components=2).fit(foreground, background)

    with pytest.raises(ValueError, match="invalid component indices"):
        ged.apply(foreground, exclude=[2])


def test_ged_plot_components():
    """Test plotting GED component topographies."""
    foreground, background = _make_data(n_channels=4)
    info = create_info(
        ["Fz", "Cz", "Pz", "Oz"],
        sfreq=100.0,
        ch_types="eeg",
    )
    info.set_montage(make_standard_montage("colin27_1020"))

    ged = GED(n_components=2).fit(foreground, background)
    fig = ged.plot_components(info=info, picks=[0, 1], show=False)

    assert len(fig.axes) >= 2
    plt.close(fig)


def test_ged_plot_components_without_info():
    """Test plotting GED component weights without channel locations."""
    foreground, background = _make_data(n_channels=4)

    ged = GED(n_components=2).fit(foreground, background)
    fig = ged.plot_components(picks=[0], show=False)

    assert len(fig.axes) >= 1
    plt.close(fig)


def test_ged_plot_properties_raw():
    """Test plotting GED properties from Raw."""
    foreground, background = _make_data(n_channels=4)
    info = create_info(
        ["Fz", "Cz", "Pz", "Oz"],
        sfreq=100.0,
        ch_types="eeg",
    )
    info.set_montage(make_standard_montage("colin27_1020"))

    raw_foreground = RawArray(foreground, info)
    raw_background = RawArray(background, info)

    ged = GED(n_components=2).fit(raw_foreground, raw_background)
    figs = ged.plot_properties(raw_foreground, picks=[0, 1], show=False)

    assert len(figs) == 2
    assert all(len(fig.axes) >= 3 for fig in figs)
    for fig in figs:
        plt.close(fig)


def test_ged_plot_properties_array_requires_sfreq():
    """Test that ndarray property plotting requires sfreq."""
    foreground, background = _make_data(n_channels=4)

    ged = GED(n_components=2).fit(foreground, background)

    with pytest.raises(ValueError, match="sfreq is required"):
        ged.plot_properties(foreground, picks=[0], show=False)


def test_ged_plot_properties_array_with_sfreq():
    """Test plotting GED properties from ndarray with sfreq."""
    foreground, background = _make_data(n_channels=4)

    ged = GED(n_components=2).fit(foreground, background)
    figs = ged.plot_properties(
        foreground,
        picks=[0],
        sfreq=100.0,
        show=False,
    )

    assert len(figs) == 1
    assert len(figs[0].axes) >= 3
    plt.close(figs[0])


def test_ged_fit_raw():
    """Test fitting GED from Raw objects."""
    foreground, background = _make_data(n_channels=4)
    info = create_info(
        ["Fz", "Cz", "Pz", "Oz"],
        sfreq=100.0,
        ch_types="eeg",
    )

    raw_foreground = RawArray(foreground, info)
    raw_background = RawArray(background, info)

    ged = GED(n_components=2).fit(raw_foreground, raw_background)

    assert ged.filters_.shape == (2, 4)
    assert ged.patterns_.shape == (2, 4)
    assert ged.eigenvalues_.shape == (2,)
    assert ged.ch_names_ == ["Fz", "Cz", "Pz", "Oz"]


def test_ged_fit_raw_channel_mismatch():
    """Test that Raw channel mismatch is rejected."""
    foreground, background = _make_data(n_channels=4)
    info_foreground = create_info(
        ["Fz", "Cz", "Pz", "Oz"],
        sfreq=100.0,
        ch_types="eeg",
    )
    info_background = create_info(
        ["Fz", "Cz", "Pz", "P4"],
        sfreq=100.0,
        ch_types="eeg",
    )

    raw_foreground = RawArray(foreground, info_foreground)
    raw_background = RawArray(background, info_background)

    with pytest.raises(ValueError, match="same channel names in the same order"):
        GED(n_components=2).fit(raw_foreground, raw_background)


def test_ged_explained_ratio():
    """Test normalized GED eigenvalues."""
    foreground, background = _make_data()

    ged = GED(n_components=4).fit(foreground, background)

    assert ged.explained_ratio_.shape == (4,)
    np.testing.assert_allclose(ged.explained_ratio_.sum(), 1.0)


def test_ged_apply_raw():
    """Test GED application to Raw objects."""
    foreground, background = _make_data(n_channels=4)
    info = create_info(
        ["Fz", "Cz", "Pz", "Oz"],
        sfreq=100.0,
        ch_types="eeg",
    )

    raw_foreground = RawArray(foreground, info)
    raw_background = RawArray(background, info)

    ged = GED(n_components=4).fit(raw_foreground, raw_background)
    raw_clean = ged.apply(raw_foreground, exclude=[0])

    assert isinstance(raw_clean, RawArray)
    assert raw_clean.get_data().shape == raw_foreground.get_data().shape
    assert np.isfinite(raw_clean.get_data()).all()
    assert raw_clean is not raw_foreground


def test_ged_plot_eigenvalues():
    """Test plotting GED eigenvalues."""
    foreground, background = _make_data(n_channels=4)

    ged = GED(n_components=4).fit(foreground, background)
    fig = ged.plot_eigenvalues(show=False)

    assert len(fig.axes) == 2
    plt.close(fig)


def test_ged_apply_uses_full_reconstruction():
    """Test that apply reconstructs from all components, not only n_components."""
    foreground, background = _make_data(n_channels=8)

    ged = GED(n_components=2).fit(foreground, background)

    cleaned = ged.apply(foreground, exclude=[])

    np.testing.assert_allclose(cleaned, foreground, rtol=1e-10, atol=1e-10)


def test_ged_apply_raw_empty_exclude_identity():
    """Test Raw output is unchanged when no components are excluded."""
    foreground, background = _make_data(n_channels=4)
    info = create_info(
        ["Fz", "Cz", "Pz", "Oz"],
        sfreq=100.0,
        ch_types="eeg",
    )

    raw_foreground = RawArray(foreground, info)
    raw_background = RawArray(background, info)

    ged = GED(n_components=2).fit(raw_foreground, raw_background)
    raw_clean = ged.apply(raw_foreground, exclude=[])

    np.testing.assert_allclose(
        raw_clean.get_data(),
        raw_foreground.get_data(),
        rtol=1e-10,
        atol=1e-10,
    )
