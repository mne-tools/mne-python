"""Core visualization operations."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import importlib
from contextlib import contextmanager
import sys

from ._utils import _get_backend_based_on_env_and_defaults, VALID_3D_BACKENDS
from ...utils import logger
from ...utils.check import _check_option

try:
    MNE_3D_BACKEND
    MNE_3D_BACKEND_TEST_DATA
except NameError:
    MNE_3D_BACKEND = _get_backend_based_on_env_and_defaults()
    MNE_3D_BACKEND_TEST_DATA = None

logger.info('Using %s 3d backend.\n' % MNE_3D_BACKEND)

if MNE_3D_BACKEND == 'mayavi':
    from ._pysurfer_mayavi import _Renderer, _Projection  # lgtm # noqa: F401
elif MNE_3D_BACKEND == 'pyvista':
    from ._pyvista import _Renderer, _Projection  # lgtm # noqa: F401


def set_3d_backend(backend_name):
    """Set the backend for MNE.

    The backend will be set as specified and operations will use
    that backend

    Parameters
    ----------
    backend_name : str
        The 3d backend to select.
    """
    _check_option('backend_name', backend_name, VALID_3D_BACKENDS)
    global MNE_3D_BACKEND
    MNE_3D_BACKEND = backend_name
    importlib.reload(sys.modules[__name__])


def get_3d_backend():
    """Return the backend currently used.

    Returns
    -------
    backend_used : str
        The 3d backend currently in use.
    """
    global MNE_3D_BACKEND
    return MNE_3D_BACKEND


@contextmanager
def use_3d_backend(backend_name):
    """Create a viz context.

    Parameters
    ----------
    backend_name : str
        The 3d backend to use in the context.
    """
    old_backend = get_3d_backend()
    set_3d_backend(backend_name)
    try:
        yield
    finally:
        set_3d_backend(old_backend)


@contextmanager
def _use_test_3d_backend(backend_name):
    """Create a testing viz context.

    Parameters
    ----------
    backend_name : str
        The 3d backend to use in the context.
    """
    with use_3d_backend(backend_name):
        global MNE_3D_BACKEND_TEST_DATA
        if backend_name == 'pyvista':
            MNE_3D_BACKEND_TEST_DATA = True
        yield
