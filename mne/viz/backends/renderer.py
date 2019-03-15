"""Core visualization operations."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import importlib
from ._utils import _get_backend_based_on_env_and_defaults, VALID_3D_BACKENDS
from ...utils import logger
from ...utils.check import _check_option

try:
    MNE_3D_BACKEND
except NameError:
    MNE_3D_BACKEND = _get_backend_based_on_env_and_defaults()
finally:
    logger.info('Using %s 3d backend.\n' % MNE_3D_BACKEND)

if MNE_3D_BACKEND == 'mayavi':
    from ._pysurfer_mayavi import _Renderer, _Projection  # noqa: F401
elif MNE_3D_BACKEND == 'vispy':
    from ._vispy import _Renderer, _Projection  # noqa: F401
else:
    raise RuntimeError('This should never happen, there was some issue with'
                       ' MNE_3D_BACKEND check %s' % __file__)


def set_3d_backend(backend_name):
    """Set the backend for MNE.

    The backend will be set as specified and operations will use
    that backend

    Parameters
    ----------
    backend_name : str
    """
    _check_option('backend_name', backend_name, VALID_3D_BACKENDS)
    global MNE_3D_BACKEND
    MNE_3D_BACKEND = backend_name
    from . import renderer
    importlib.reload(renderer)


def get_3d_backend():
    """Return the backend currently used.

    Returns
    -------
    backend_used : str
        the backend currently in use
    """
    global MNE_3D_BACKEND
    return MNE_3D_BACKEND
