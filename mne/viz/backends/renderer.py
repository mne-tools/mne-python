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

from ._utils import Backends3D
from ...utils import logger

try:
    MNE_3D_BACKEND
    MNE_3D_BACKEND_TEST_DATA
except NameError:
    MNE_3D_BACKEND = Backends3D.get_backend_based_on_env_and_defaults()
    MNE_3D_BACKEND_TEST_DATA = None

logger.info('Using %s 3d backend.\n' % MNE_3D_BACKEND)

if MNE_3D_BACKEND == Backends3D.mayavi:
    from ._pysurfer_mayavi import _Renderer, _Projection  # lgtm # noqa: F401
elif MNE_3D_BACKEND == Backends3D.ipyvolume:
    from ._ipyvolume import _Renderer  # lgtm # noqa: F401
elif MNE_3D_BACKEND == Backends3D.vtki:
    from ._vtki import _Renderer, _Projection  # lgtm # noqa: F401


def set_3d_backend(backend_name):
    """Set the backend for MNE.

    The backend will be set as specified and operations will use
    that backend

    Parameters
    ----------
    backend_name : str
        The 3d backend to select.
    """
    Backends3D.check_backend(backend_name)
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
    yield
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
        if backend_name == Backends3D.vtki:
            MNE_3D_BACKEND_TEST_DATA = True
        yield
