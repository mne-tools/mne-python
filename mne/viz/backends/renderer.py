"""Core visualization operations."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import sys
import os
import importlib

default_3d_backend = 'mayavi'


try:
    if MNE_3D_BACKEND is None:
        MNE_3D_BACKEND = os.environ.get('MNE_3D_BACKEND',
                                        default_3d_backend)
except NameError:
    MNE_3D_BACKEND = os.environ.get('MNE_3D_BACKEND', default_3d_backend)

if MNE_3D_BACKEND == 'mayavi':
    from ._pysurfer_mayavi import Renderer, _Projection  # noqa: F401
else:
    import warnings
    warnings.warn('MNE_3D_BACKEND should be "mayavi" : '
                  '{} was given.'.format(MNE_3D_BACKEND))

sys.stderr.write('Using {} backend.\n'.format(MNE_3D_BACKEND))


def set_3d_backend(backend_name):
    """Set the backend for MNE.

    The backend will be set as specified and operations will use
    that backend

    Parameters
    ----------
    backend_name : {'mlab'}, default is 'mlab'
    """
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
    backend_used = MNE_3D_BACKEND
    return backend_used
