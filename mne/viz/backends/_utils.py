# -*- coding: utf-8 -*-
#
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import numpy as np
import collections.abc
from ...utils import get_config
from ...utils.check import _check_option

DEFAULT_3D_BACKEND = 'mayavi'
VALID_3D_BACKENDS = ['mayavi', 'pyvista']


def _get_backend_based_on_env_and_defaults():
    backend = get_config(key='MNE_3D_BACKEND', default=DEFAULT_3D_BACKEND)
    _check_option('MNE_3D_BACKEND', backend, VALID_3D_BACKENDS)

    return backend


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
        if np_color.dtype == np.int:
            if (np_color < 0).any() or (np_color > 255).any():
                raise ValueError("Values out of range [0, 255].")
        elif np_color.dtype == np.float:
            if (np_color < 0.0).any() or (np_color > 1.0).any():
                raise ValueError("Values out of range [0.0, 1.0].")
        else:
            raise TypeError("Expected data type is `np.int` or `np.float` but "
                            "{} was given.".format(np_color.dtype))
    else:
        raise TypeError("Expected type is `str` or iterable but "
                        "{} was given.".format(type(color)))
    return color
