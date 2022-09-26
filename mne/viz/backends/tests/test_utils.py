# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import sys
from colorsys import rgb_to_hls
from contextlib import nullcontext

import numpy as np
import pytest

from mne import create_info
from mne.io import RawArray
from mne.viz.backends._utils import (_get_colormap_from_array, _check_color,
                                     _qt_is_dark, _pixmap_to_ndarray)
from mne.utils import _check_qt_version


def test_get_colormap_from_array():
    """Test setting a colormap."""
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    cmap = _get_colormap_from_array()
    assert isinstance(cmap, LinearSegmentedColormap)
    cmap = _get_colormap_from_array(colormap='viridis')
    assert isinstance(cmap, ListedColormap)
    cmap = _get_colormap_from_array(colormap=[1, 1, 1],
                                    normalized_colormap=True)
    assert isinstance(cmap, ListedColormap)
    cmap = _get_colormap_from_array(colormap=[255, 255, 255],
                                    normalized_colormap=False)
    assert isinstance(cmap, ListedColormap)


def test_check_color():
    """Test color format."""
    assert _check_color('red') == (1., 0., 0.)
    assert _check_color((0., 1., 0., 1.)) == (0., 1., 0., 1.)
    assert _check_color((0, 0, 255, 255)) == (0, 0, 255, 255)
    with pytest.raises(ValueError, match='RGB or RGBA'):
        _check_color([255, 0])
    with pytest.raises(ValueError, match='out of range'):
        _check_color([256, 0, 0])
    with pytest.raises(ValueError, match='out of range'):
        _check_color([-1.0, 0.0, 0.0])
    with pytest.raises(TypeError, match='Expected data type'):
        _check_color(['foo', 'bar', 'foo'])
    with pytest.raises(TypeError, match='Expected type'):
        _check_color(None)


@pytest.mark.pgtest
@pytest.mark.parametrize('theme', ('auto', 'light', 'dark'))
def test_theme_colors(pg_backend, theme, monkeypatch, tmp_path):
    """Test that theme colors propagate properly."""
    darkdetect = pytest.importorskip('darkdetect')
    monkeypatch.setenv('_MNE_FAKE_HOME_DIR', str(tmp_path))
    monkeypatch.delenv('MNE_BROWSER_THEME', raising=False)
    # make it seem like the system is always in light mode
    monkeypatch.setattr(darkdetect, 'theme', lambda: 'light')
    raw = RawArray(np.zeros((1, 1000)), create_info(1, 1000., 'eeg'))
    _, api = _check_qt_version(return_api=True)
    if api in ('PyQt6', 'PySide6') and theme == 'dark':
        ctx = pytest.warns(RuntimeWarning, match='not yet supported')
        return_early = True
    else:
        ctx = nullcontext()
        return_early = False
    with ctx:
        fig = raw.plot(theme=theme)
    if return_early:
        return  # we could add a ton of conditionals below, but KISS
    is_dark = _qt_is_dark(fig)
    # on Darwin these checks get complicated, so don't bother for now
    if sys.platform != 'darwin':
        if theme == 'dark':
            assert is_dark, theme
        elif theme == 'light':
            assert not is_dark, theme
        else:
            got_dark = darkdetect.theme().lower() == 'dark'
            assert is_dark is got_dark

    def assert_correct_darkness(widget, want_dark):
        __tracebackhide__ = True  # noqa
        # This should work, but it just picks up the parent in the errant case!
        bgcolor = widget.palette().color(widget.backgroundRole()).getRgbF()[:3]
        dark = rgb_to_hls(*bgcolor)[1] < 0.5
        assert dark == want_dark, f'{widget} dark={dark} want_dark={want_dark}'
        # ... so we use a more direct test
        colors = _pixmap_to_ndarray(widget.grab())[:, :, :3]
        dark = colors.mean() < 0.5
        assert dark == want_dark, f'{widget} dark={dark} want_dark={want_dark}'

    for widget in (fig.mne.toolbar, fig.statusBar()):
        assert_correct_darkness(widget, is_dark)
