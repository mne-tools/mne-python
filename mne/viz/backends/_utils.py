# -*- coding: utf-8 -*-
#
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

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
        cmap = ListedColormap(colormap / 255.0)
    return cmap
