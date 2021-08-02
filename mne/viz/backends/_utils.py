# -*- coding: utf-8 -*-
#
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

from contextlib import contextmanager
import numpy as np
import collections.abc
from ...externals.decorator import decorator

VALID_BROWSE_BACKENDS = (
    'matplotlib',
    'pyqtgraph'
)

VALID_3D_BACKENDS = (
    'pyvistaqt',  # default 3d backend
    'mayavi',
    'notebook',
)
ALLOWED_QUIVER_MODES = ('2darrow', 'arrow', 'cone', 'cylinder', 'sphere',
                        'oct')


def _get_colormap_from_array(colormap=None, normalized_colormap=False,
                             default_colormap='coolwarm'):
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    if colormap is None:
        cmap = cm.get_cmap(default_colormap)
    elif isinstance(colormap, str):
        cmap = cm.get_cmap(colormap)
    elif normalized_colormap:
        cmap = ListedColormap(colormap)
    else:
        cmap = ListedColormap(np.array(colormap) / 255.0)
    return cmap


def _check_color(color):
    from matplotlib.colors import colorConverter
    if isinstance(color, str):
        color = colorConverter.to_rgb(color)
    elif isinstance(color, collections.abc.Iterable):
        np_color = np.array(color)
        if np_color.size % 3 != 0 and np_color.size % 4 != 0:
            raise ValueError("The expected valid format is RGB or RGBA.")
        if np_color.dtype in (np.int64, np.int32):
            if (np_color < 0).any() or (np_color > 255).any():
                raise ValueError("Values out of range [0, 255].")
        elif np_color.dtype == np.float64:
            if (np_color < 0.0).any() or (np_color > 1.0).any():
                raise ValueError("Values out of range [0.0, 1.0].")
        else:
            raise TypeError("Expected data type is `np.int64`, `np.int32`, or "
                            "`np.float64` but {} was given."
                            .format(np_color.dtype))
    else:
        raise TypeError("Expected type is `str` or iterable but "
                        "{} was given.".format(type(color)))
    return color


def _alpha_blend_background(ctable, background_color):
    alphas = ctable[:, -1][:, np.newaxis] / 255.
    use_table = ctable.copy()
    use_table[:, -1] = 255.
    return (use_table * alphas) + background_color * (1 - alphas)


@decorator
def run_once(fun, *args, **kwargs):
    """Run the function only once."""
    if not hasattr(fun, "_has_run"):
        fun._has_run = True
        return fun(*args, **kwargs)


@run_once
def _init_qt_resources():
    from ...icons import resources
    resources.qInitResources()


@contextmanager
def _qt_disable_paint(widget):
    paintEvent = widget.paintEvent
    widget.paintEvent = lambda *args, **kwargs: None
    try:
        yield
    finally:
        widget.paintEvent = paintEvent
