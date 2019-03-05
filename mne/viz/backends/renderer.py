"""Core visualization operations."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import importlib
from ...utils import logger, get_config, _validate_type

default_3d_backend = 'mayavi'


try:
    if MNE_3D_BACKEND is None:
        MNE_3D_BACKEND = get_config('MNE_3D_BACKEND', default_3d_backend)
except NameError:
    MNE_3D_BACKEND = get_config('MNE_3D_BACKEND', default_3d_backend)

if MNE_3D_BACKEND == 'mayavi':
    from ._pysurfer_mayavi import _Renderer, _Projection  # noqa
else:
    raise ValueError('MNE_3D_BACKEND should be "mayavi" : '
                     '{} was given.'.format(MNE_3D_BACKEND))

logger.info('Using %s 3d backend.\n' % MNE_3D_BACKEND)


def set_3d_backend(backend_name):
    """Set the backend for MNE.

    The backend will be set as specified and operations will use
    that backend

    Parameters
    ----------
    backend_name : {'mlab'}, default is 'mlab'
    """
    _validate_type(backend_name, "str")

    global MNE_3D_BACKEND
    if backend_name == 'mayavi':
        MNE_3D_BACKEND = backend_name
    else:
        raise ValueError('backend_name should be "mayavi" : '
                         '{} was given.'.format(MNE_3D_BACKEND))
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
