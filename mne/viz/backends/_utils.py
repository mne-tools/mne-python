# -*- coding: utf-8 -*-
#
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#
# License: Simplified BSD
from enum import Enum

from ...utils import get_config
from ...utils.check import _check_option


DEFAULT_3D_BACKEND = 'mayavi'


class Backends3D(str, Enum):
    """Enumeration of valid 3D backends."""

    ipyvolume = 'ipyvolume'
    mayavi = 'mayavi'
    pyvista = 'pyvista'

    @classmethod
    def get_backend_based_on_env_and_defaults(cls):
        """Read MNE-Python preferences from environment or config file."""
        backend = get_config(key='MNE_3D_BACKEND', default=DEFAULT_3D_BACKEND)
        cls.check_backend(backend)

        return backend

    @classmethod
    def check_backend(cls, backend_value):
        """Check the value of the backend against a list of valid options.

        Parameters
        ----------
        backend_value: str
            Provided by user backend value.
        """
        valid_values = tuple(b.value for b in cls)
        _check_option('MNE_3D_BACKEND', backend_value, valid_values)


def _get_colormap_from_array(colormap=None, normalized_colormap=False,
                             default_colormap='coolwarm'):
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    if colormap is None:
        cmap = cm.get_cmap(default_colormap)
    elif normalized_colormap:
        cmap = ListedColormap(colormap)
    else:
        cmap = ListedColormap(colormap / 255.0)
    return cmap


def _get_color_from_scalars(cmap, scalars=None, vmin=None, vmax=None):
    color = None
    if scalars is not None:
        if vmin is None:
            vmin = min(scalars)
        if vmax is None:
            vmax = max(scalars)
        nscalars = (scalars - vmin) / (vmax - vmin)
        color = cmap(nscalars)
    return color
