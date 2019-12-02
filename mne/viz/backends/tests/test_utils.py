# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import pytest
from mne.viz.backends._utils import _get_colormap_from_array, _check_color


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
