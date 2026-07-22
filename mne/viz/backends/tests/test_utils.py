# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from colorsys import rgb_to_hls

import numpy as np
import pytest

from mne import create_info
from mne.io import RawArray
from mne.viz.backends._utils import (
    _check_color,
    _get_colormap_from_array,
    _pixmap_to_ndarray,
    _qt_is_dark,
)


def test_get_colormap_from_array():
    """Test setting a colormap."""
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    cmap = _get_colormap_from_array()
    assert isinstance(cmap, LinearSegmentedColormap)
    cmap = _get_colormap_from_array(colormap="viridis")
    assert isinstance(cmap, ListedColormap)
    cmap = _get_colormap_from_array(colormap=[1, 1, 1], normalized_colormap=True)
    assert isinstance(cmap, ListedColormap)
    cmap = _get_colormap_from_array(colormap=[255, 255, 255], normalized_colormap=False)
    assert isinstance(cmap, ListedColormap)


def test_check_color():
    """Test color format."""
    assert _check_color("red") == (1.0, 0.0, 0.0)
    assert _check_color((0.0, 1.0, 0.0, 1.0)) == (0.0, 1.0, 0.0, 1.0)
    assert _check_color((0, 0, 255, 255)) == (0, 0, 255, 255)
    with pytest.raises(ValueError, match="RGB or RGBA"):
        _check_color([255, 0])
    with pytest.raises(ValueError, match="out of range"):
        _check_color([256, 0, 0])
    with pytest.raises(ValueError, match="out of range"):
        _check_color([-1.0, 0.0, 0.0])
    with pytest.raises(TypeError, match="Expected data type"):
        _check_color(["foo", "bar", "foo"])
    with pytest.raises(TypeError, match="Expected type"):
        _check_color(None)


def _assert_correct_darkness(widget, want_dark):
    __tracebackhide__ = True  # noqa
    # The override propagates to children, so both palette and pixels should match.
    bgcolor = widget.palette().color(widget.backgroundRole()).getRgbF()[:3]
    dark = rgb_to_hls(*bgcolor)[1] < 0.5
    assert dark == want_dark, f"{widget} palette dark={dark} want_dark={want_dark}"
    colors = _pixmap_to_ndarray(widget.grab())[:, :, :3]
    dark = colors.mean() < 0.5
    assert dark == want_dark, f"{widget} pixmap dark={dark} want_dark={want_dark}"


@pytest.mark.pgtest
@pytest.mark.parametrize("theme", ("auto", "light", "dark"))
def test_theme_colors(pg_backend, theme, monkeypatch, tmp_path):
    """Test that theme colors propagate properly."""
    darkdetect = pytest.importorskip("darkdetect")
    monkeypatch.setenv("_MNE_FAKE_HOME_DIR", str(tmp_path))
    monkeypatch.delenv("MNE_BROWSER_THEME", raising=False)
    # A qdarkstyle stylesheet is only applied when the requested theme differs from
    # the system, so fake the system as the opposite of the request
    if theme == "auto":
        want_dark = (darkdetect.theme() or "light").lower() == "dark"
    else:
        want_dark = theme == "dark"
        fake_system = "light" if want_dark else "dark"
        monkeypatch.setattr(darkdetect, "theme", lambda: fake_system)
    raw = RawArray(np.zeros((1, 1000)), create_info(1, 1000.0, "eeg"))
    fig = raw.plot(theme=theme)
    is_dark = _qt_is_dark(fig)
    assert is_dark == want_dark, theme

    for widget in (fig.mne.toolbar, fig.statusBar()):
        _assert_correct_darkness(widget, is_dark)
